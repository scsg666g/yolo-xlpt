"""Existing endpoint tests exercise domain logic independently of browser auth.

Authentication and actual middleware boundaries are tested without this
isolation in test_auth.py. This is a pytest-only override, never a server flag.
"""
import pytest


@pytest.fixture(autouse=True)
def domain_auth_boundary(request, monkeypatch, tmp_path):
    if request.node.path.name == "test_auth.py":
        return
    from app import auth
    monkeypatch.setattr(auth, "AUTH_ROOT", tmp_path / "private-account")

    async def domain_request(_request):
        return None

    monkeypatch.setattr(auth, "authorize", domain_request)
