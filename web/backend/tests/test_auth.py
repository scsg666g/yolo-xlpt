from concurrent.futures import ThreadPoolExecutor
import asyncio

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from app import auth
from app.main import app

PASSWORD = "test-password-123!"
NEW_PASSWORD = "new-password-456!"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ROOT", tmp_path / "private-account")
    monkeypatch.setenv("PLATFORM_COOKIE_SECURE", "false")
    auth.initialize()
    # No lifespan: authentication tests must not start real training workers.
    instance = TestClient(app, base_url="http://testserver:18088")
    yield instance
    instance.close()


def setup(client):
    return client.post("/api/auth/setup", json={"username": "admin", "password": PASSWORD,
        "code": auth.setup_code()})


def login(client, password=PASSWORD):
    return client.post("/api/auth/login", json={"username": "admin", "password": password})


def csrf(client):
    return {"X-CSRF-Token": client.get("/api/auth/status").json()["csrf_token"]}


def test_uninitialized_platform_is_closed_and_no_secrets_leak(client):
    response = client.get("/api/auth/status")
    assert response.json() == {"initialized": False, "authenticated": False, "username": None, "csrf_token": None}
    for path in ["/api/tasks", "/api/resources", "/api/health", "/api/compute/agent/download",
                 "/api/model-tests/any/images/0/original", "/api/artifacts/1/download",
                 "/api/stream", "/api/unknown", "/docs", "/openapi.json"]:
        assert client.get(path).status_code == 401, path
    assert response.headers["cache-control"] == "no-store"
    assert len(auth.setup_code()) >= 32
    assert (auth.AUTH_ROOT / "setup-code").stat().st_mode & 0o777 == 0o600


def test_setup_requires_server_code_and_strong_password(client):
    body = {"username": "admin", "password": PASSWORD, "code": "wrong"}
    assert client.post("/api/auth/setup", json=body).status_code == 403
    body["code"] = "无效校验码"
    assert client.post("/api/auth/setup", json=body).status_code == 403
    body.update(code=auth.setup_code(), password="short")
    assert client.post("/api/auth/setup", json=body).status_code == 422
    body.update(password=PASSWORD, username="   ")
    assert client.post("/api/auth/setup", json=body).status_code == 422
    result = setup(client)
    assert result.status_code == 201
    assert not (auth.AUTH_ROOT / "setup-code").exists()
    assert auth.setup_code() is None
    body.update(password=PASSWORD, username="admin")
    assert client.post("/api/auth/setup", json=body).status_code == 409
    assert client.get("/api/auth/status").json()["authenticated"]
    with auth.store() as db:
        encoded = db.execute("SELECT password FROM account").fetchone()[0]
        token = db.execute("SELECT digest FROM sessions").fetchone()[0]
    assert PASSWORD not in encoded
    assert auth.password_matches(PASSWORD, encoded)
    assert token not in list(client.cookies.values())
    assert (auth.AUTH_ROOT / "account.sqlite3").stat().st_mode & 0o777 == 0o600


def test_cookie_security_and_port_isolation(client):
    response = setup(client)
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "max-age=" in cookie
    assert "secure" not in cookie
    assert not client.get("http://testserver:18089/api/auth/status").json()["authenticated"]
    secure = TestClient(app, base_url="https://testserver:18088")
    assert "secure" in login(secure).headers["set-cookie"].lower()
    secure.close()


def test_login_wrong_credentials_and_persistent_account(client):
    setup(client)
    client.cookies.clear()
    assert login(client, "bad-password").status_code == 401
    assert client.post("/api/auth/login", json={"username": "other", "password": PASSWORD}).status_code == 401
    assert not client.get("/api/auth/status").json()["authenticated"]
    assert login(client).status_code == 200
    assert client.get("/api/auth/status").json()["username"] == "admin"
    assert client.get("/api/unknown").status_code == 404
    assert client.get("/openapi.json").status_code == 200


def test_csrf_and_origin_protect_writes_including_login(client):
    assert client.post("/api/auth/login", headers={"Origin": "https://evil.example"}, json={
        "username": "admin", "password": PASSWORD}).status_code == 403
    setup(client)
    assert client.post("/api/auth/logout").status_code == 403
    assert client.post("/api/auth/logout", headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert client.post("/api/auth/logout", headers={**csrf(client), "Origin": "http://testserver:18089"}).status_code == 403
    assert client.post("/api/auth/logout", headers={**csrf(client), "Origin": "http://testserver:18088"}).status_code == 204
    assert client.get("/api/unknown").status_code == 401


def test_password_change_revokes_all_sessions_but_logout_only_current(client):
    setup(client)
    second = TestClient(app, base_url="http://testserver:18088")
    login(second)
    assert client.post("/api/auth/logout", headers=csrf(client)).status_code == 204
    assert second.get("/api/auth/status").json()["authenticated"]
    login(client)
    headers = csrf(client)
    assert client.post("/api/auth/password", headers=headers, json={"current_password": "bad", "new_password": NEW_PASSWORD}).status_code == 400
    assert client.post("/api/auth/password", headers=headers, json={"current_password": PASSWORD, "new_password": PASSWORD}).status_code == 422
    assert client.post("/api/auth/password", headers=headers, json={"current_password": PASSWORD, "new_password": NEW_PASSWORD}).status_code == 204
    assert not second.get("/api/auth/status").json()["authenticated"]
    assert client.get("/api/unknown").status_code == 401
    assert login(client).status_code == 401
    assert login(client, NEW_PASSWORD).status_code == 200
    second.close()


def test_expired_and_revoked_sessions_are_rejected(client):
    setup(client)
    with auth.store() as db:
        db.execute("UPDATE sessions SET expires = 0")
    assert client.get("/api/unknown").status_code == 401
    assert not client.get("/api/auth/status").json()["authenticated"]
    login(client)
    with auth.store() as db:
        db.execute("UPDATE account SET revision = revision + 1")
    assert client.get("/api/unknown").status_code == 401


def test_throttle_does_not_trust_forwarded_ip(client):
    setup(client)
    for index in range(11):
        assert client.post("/api/auth/login", headers={"X-Forwarded-For": f"192.0.2.{index}"}, json={
            "username": "admin", "password": "bad"}).status_code == 401
    response = login(client)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "300"
    with auth.store() as db:
        db.execute("UPDATE attempts SET started = 0")
    assert login(client).status_code == 200


def test_only_existing_token_authenticated_agent_routes_are_exempt(client):
    expected = {
        route.path for route in app.routes if getattr(route, "path", "").startswith("/api/")
        and ("/agent/" in route.path or route.path.endswith("/report"))
        and route.path != "/api/compute/agent/download"
    }
    for route in expected:
        assert auth.is_agent_path(route), route
    for path in ["/api/compute/agent/download", "/api/compute/nodes/n/token", "/api/compute/nodes/n",
                 "/api/compute/nodes/n/agent/runs/r/delete", "/api/tasks/agent/poll",
                 "/api/dataset-storage/agent/n/anything/delete"]:
        assert not auth.is_agent_path(path)


def test_agent_poll_reaches_its_token_boundary_without_browser_login(client, monkeypatch):
    from app import compute_nodes, remote_operations
    def token_check(*args, **kwargs):
        raise HTTPException(401, "agent-token-checked")
    monkeypatch.setattr(compute_nodes, "authenticated_agent", token_check)
    monkeypatch.setattr(remote_operations, "authenticated_agent", token_check)
    for path in ["/api/compute/nodes/test/agent/poll", "/api/compute/nodes/test/agent/operations/poll",
                 "/api/dataset-storage/agent/test/poll"]:
        response = client.post(path, json={})
        assert response.status_code == 401
        assert response.json()["detail"] == "agent-token-checked", path


def test_open_task_stream_closes_after_session_revocation(client, monkeypatch):
    from app import main
    setup(client)
    monkeypatch.setattr(main, "load_task_snapshot", lambda: "[]")
    cookie = "; ".join(f"{name}={value}" for name, value in client.cookies.items())
    request = Request({"type": "http", "scheme": "http", "method": "GET", "path": "/api/stream",
        "server": ("testserver", 18088), "query_string": b"", "headers": [
            (b"host", b"testserver:18088"), (b"cookie", cookie.encode())]})
    async def exercise():
        baseline = set(main._task_stream_subscribers)
        response = await main.stream(request)
        iterator = response.body_iterator
        assert "event: tasks" in await anext(iterator)
        subscriber, = main._task_stream_subscribers - baseline
        with auth.store() as db:
            db.execute("DELETE FROM sessions")
        subscriber.put_nowait('[{"secret":"must-not-be-sent"}]')
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        assert main._task_stream_subscribers == baseline
    asyncio.run(exercise())


def test_concurrent_setup_only_creates_one_administrator(client):
    code = auth.setup_code()
    # Avoid entering lifespan (training workers). Independent HTTP clients only.
    def create(index):
        other = TestClient(app, base_url="http://testserver:18088")
        try:
            return other.post("/api/auth/setup", json={"username": f"admin{index}", "password": PASSWORD, "code": code}).status_code
        finally:
            other.close()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, range(2)))
    assert sorted(results) == [201, 409]
    with auth.store() as db:
        assert db.execute("SELECT count(*) FROM account").fetchone()[0] == 1
