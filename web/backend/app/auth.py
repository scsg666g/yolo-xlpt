"""Single administrator browser authentication; Agent tokens remain independent.

The small private database lives outside the training database so login/password
operations never hold training transactions. No credentials are shipped in Git.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import DATA_ROOT, project_path

AUTH_ROOT = project_path(os.getenv("PLATFORM_AUTH_ROOT", str(DATA_ROOT / "auth")))
SESSION_SECONDS = max(300, min(30 * 86400, int(os.getenv("PLATFORM_SESSION_SECONDS", "43200"))))
router = APIRouter(prefix="/api/auth", tags=["account"])
_DUMMY_HASH = "scrypt$" + "00" * 16 + "$" + "00" * 32


def password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def password_matches(password: str, encoded: str) -> bool:
    _, salt, expected = encoded.split("$")
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1, dklen=32)
    return hmac.compare_digest(digest.hex(), expected)


def validate_password(password: str) -> None:
    if len(password) < 10 or len(password) > 128 or not password.strip():
        raise HTTPException(422, "密码须为 10～128 个字符，不能全为空格")


@contextmanager
def store():
    AUTH_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = AUTH_ROOT / "account.sqlite3"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.close(descriptor)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY CHECK (id = 1), username TEXT NOT NULL,
                password TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS sessions (
                digest TEXT PRIMARY KEY, csrf TEXT NOT NULL,
                revision INTEGER NOT NULL, expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                key TEXT PRIMARY KEY, started REAL NOT NULL, count INTEGER NOT NULL
            );
        """)
        yield connection
        connection.commit()
    finally:
        connection.close()


def setup_code() -> str | None:
    with store() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM account").fetchone():
            return None
        path = AUTH_ROOT / "setup-code"
        try:
            # The database lock keeps other workers from reading an incomplete
            # setup-code file or racing the administrator creation transaction.
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return path.read_text(encoding="ascii").strip()
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            code = secrets.token_urlsafe(24)
            handle.write(code + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return code


def initialize() -> None:
    setup_code()


def cookie_name(request: Request) -> str:
    # Cookies ignore ports: installation + public port isolates dev/prod even
    # when a deployment intentionally shares one data mount.
    identity = f"{AUTH_ROOT.resolve()}:{request.url.port or (443 if request.url.scheme == 'https' else 80)}"
    return "yolo_session_" + hashlib.sha256(identity.encode()).hexdigest()[:12]


def session_for(request: Request) -> dict | None:
    token = request.cookies.get(cookie_name(request), "")
    if not token or len(token) > 128:
        return None
    with store() as db:
        row = db.execute("""SELECT account.username, sessions.csrf, sessions.digest
            FROM sessions JOIN account ON account.revision = sessions.revision
            WHERE sessions.digest = ? AND sessions.expires > ?""",
            (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
        return dict(row) if row else None


def trusted_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return request.headers.get("sec-fetch-site") != "cross-site"
    return origin == f"{request.url.scheme}://{request.url.netloc}"


def is_agent_path(path: str) -> bool:
    # Only endpoints that already call authenticated_agent() are exempt.
    return bool(re.fullmatch(
        r"/api/compute/nodes/[^/]+/(?:report|agent/(?:poll|runs/[^/]+/(?:confirm|reconcile|update|trace|control|artifacts)|operations/(?:poll|[^/]+/(?:result|chunk))))"
        r"|/api/dataset-storage/agent/[^/]+/(?:reconcile|poll|[^/]+/(?:heartbeat|finish|chunk))", path))


async def authorize(request: Request) -> JSONResponse | None:
    path = request.url.path.rstrip("/") or "/"
    private = path == "/api" or path.startswith("/api/") or path in {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
    if not private or is_agent_path(path):
        return None
    public = {"/api/auth/status", "/api/auth/login", "/api/auth/setup"}
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not trusted_origin(request):
        return JSONResponse({"detail": "请求来源不匹配，请从平台页面操作"}, status_code=403)
    if request.method == "OPTIONS" or path in public:
        return None
    session = await asyncio.to_thread(session_for, request)
    if session is None:
        return JSONResponse({"detail": "登录已过期，请重新登录；后台训练仍会继续"}, status_code=401,
                            headers={"Cache-Control": "no-store"})
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not hmac.compare_digest(
        request.headers.get("x-csrf-token", "").encode(), session["csrf"].encode(),
    ):
        return JSONResponse({"detail": "安全校验已失效，请刷新页面后重试"}, status_code=403)
    request.state.account = session
    return None


class BrowserAuthMiddleware:
    """Pure ASGI middleware: no buffering uploads, downloads or SSE responses."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        rejected = await authorize(request)
        if rejected is not None:
            return await rejected(scope, receive, send)

        async def private_headers(message):
            if message["type"] == "http.response.start" and (request.url.path.startswith("/api") or request.url.path in {"/docs", "/redoc", "/openapi.json"}):
                message.setdefault("headers", []).extend([
                    (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"),
                ])
            await send(message)
        await self.app(scope, receive, private_headers)


def throttle(request: Request) -> None:
    # Both a peer budget and a global budget limit password hashing work. Do
    # not trust user-supplied forwarding headers for client identification.
    peer = request.client.host if request.client else "unknown"
    now = time.time()
    with store() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM attempts WHERE started < ? OR started > ?", (now - 300, now + 300))
        for key, limit in (("peer:" + peer, 12), ("global", 100)):
            row = db.execute("SELECT count FROM attempts WHERE key = ?", (key,)).fetchone()
            if row and row["count"] >= limit:
                raise HTTPException(429, "尝试次数过多，请 5 分钟后重试", headers={"Retry-After": "300"})
            db.execute("INSERT INTO attempts VALUES (?, ?, 1) ON CONFLICT(key) DO UPDATE SET count = count + 1", (key, now))


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=128)


class Setup(Credentials):
    code: str = Field(min_length=1, max_length=128)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=10, max_length=128)


def set_session(response: Response, request: Request, revision: int) -> dict:
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with store() as db:
        db.execute("DELETE FROM sessions WHERE expires <= ?", (time.time(),))
        db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                   (hashlib.sha256(token.encode()).hexdigest(), csrf, revision, time.time() + SESSION_SECONDS))
    secure = request.url.scheme == "https" or os.getenv("PLATFORM_COOKIE_SECURE", "").lower() in {"1", "true"}
    response.set_cookie(cookie_name(request), token, max_age=SESSION_SECONDS, httponly=True,
                        secure=secure, samesite="lax", path="/")
    return {"csrf_token": csrf}


@router.get("/status")
def status(request: Request):
    with store() as db:
        initialized = bool(db.execute("SELECT 1 FROM account").fetchone())
    session = session_for(request)
    return {"initialized": initialized, "authenticated": bool(session),
            "username": session["username"] if session else None,
            "csrf_token": session["csrf"] if session else None}


@router.post("/setup", status_code=201)
def setup(payload: Setup, request: Request, response: Response):
    throttle(request)
    code = setup_code()
    if not code:
        raise HTTPException(409, "管理员已设置，请登录")
    if not hmac.compare_digest(payload.code.strip().encode(), code.encode()):
        raise HTTPException(403, "服务器校验码不正确")
    validate_password(payload.password)
    username = payload.username.strip()
    if not username or any(ord(c) < 32 for c in username):
        raise HTTPException(422, "请输入有效账号")
    encoded = password_hash(payload.password)
    with store() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM account").fetchone():
            raise HTTPException(409, "管理员已设置，请登录")
        db.execute("INSERT INTO account VALUES (1, ?, ?, 1)", (username, encoded))
    (AUTH_ROOT / "setup-code").unlink(missing_ok=True)
    return {"username": username, **set_session(response, request, 1)}


@router.post("/login")
def login(payload: Credentials, request: Request, response: Response):
    throttle(request)
    with store() as db:
        account = db.execute("SELECT * FROM account").fetchone()
    if not account:
        raise HTTPException(409, "请先设置管理员账号")
    valid = password_matches(payload.password, account["password"] if payload.username == account["username"] else _DUMMY_HASH)
    if not valid or payload.username != account["username"]:
        raise HTTPException(401, "账号或密码不正确")
    return {"username": account["username"], **set_session(response, request, account["revision"])}


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response):
    with store() as db:
        db.execute("DELETE FROM sessions WHERE digest = ?", (request.state.account["digest"],))
    response.delete_cookie(cookie_name(request), path="/", httponly=True, samesite="lax")


@router.post("/password", status_code=204)
def change_password(payload: PasswordChange, request: Request, response: Response):
    throttle(request)
    validate_password(payload.new_password)
    with store() as db:
        account = db.execute("SELECT * FROM account").fetchone()
    if not password_matches(payload.current_password, account["password"]):
        raise HTTPException(400, "当前密码不正确")
    if payload.current_password == payload.new_password:
        raise HTTPException(422, "新密码不能与当前密码相同")
    encoded = password_hash(payload.new_password)
    with store() as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("SELECT revision FROM account").fetchone()
        if current["revision"] != account["revision"]:
            raise HTTPException(409, "账号已变更，请重新登录后操作")
        db.execute("UPDATE account SET password = ?, revision = revision + 1", (encoded,))
        db.execute("DELETE FROM sessions")
    response.delete_cookie(cookie_name(request), path="/", httponly=True, samesite="lax")


def main():
    parser = argparse.ArgumentParser(description="平台管理员维护（不停止训练）")
    parser.add_argument("command", choices=["setup-code", "reset-password"])
    args = parser.parse_args()
    if args.command == "setup-code":
        code = setup_code()
        print(code if code else "管理员已设置；忘记密码可执行 reset-password")
        return
    with store() as db:
        if not db.execute("SELECT 1 FROM account").fetchone():
            parser.error("尚未设置管理员，请先执行 setup-code 并在网页完成设置")
    password = getpass.getpass("新密码（10～128 个字符）：")
    if password != getpass.getpass("确认新密码："):
        parser.error("两次密码不一致")
    try:
        validate_password(password)
    except HTTPException as exc:
        parser.error(exc.detail)
    encoded = password_hash(password)
    with store() as db:
        db.execute("UPDATE account SET password = ?, revision = revision + 1", (encoded,))
        db.execute("DELETE FROM sessions")
    print("密码已更新，所有网页会话已退出；后台训练不受影响。")


if __name__ == "__main__":
    main()
