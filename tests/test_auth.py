"""End-to-end auth tests via TestClient (httpx is pinned <0.28 so it works again).

Google is mocked, so these run offline with no credentials. The app's lifespan is
NOT triggered (no `with` block), so the real app.db / heavy RAG service are never
touched; get_db is overridden to a shared in-memory SQLite DB.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.api.auth_routes as auth_routes_module
from app.config import settings
from app.database import Base, User, get_db
from main import app


@pytest.fixture
def client(monkeypatch):
    # One shared in-memory DB across all sessions (StaticPool).
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    # Default to enforced auth unless a test opts into the bypass.
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", False)
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


# ---------- enforcement ----------

def test_me_401_when_logged_out(client):
    assert client.get("/auth/me").status_code == 401


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/documents/list",
        "/api/v1/documents/stats/overview",  # was an unauthenticated aggregate leak
        "/api/v1/chat/status",               # same class — now gated
    ],
)
def test_protected_api_401_when_logged_out(client, path):
    # get_current_user 401s before the route body (so the heavy RAG service,
    # imported inside the route, is never instantiated).
    assert client.get(path).status_code == 401


def test_login_503_when_google_not_configured(client, monkeypatch):
    monkeypatch.setattr(auth_routes_module, "google_oauth_configured", lambda: False)
    r = client.get("/auth/login", follow_redirects=False)
    assert r.status_code == 503


# ---------- dev bypass ----------

def test_dev_bypass_returns_dev_user(client, monkeypatch):
    monkeypatch.setattr(settings, "DEV_AUTH_BYPASS", True)
    r = client.get("/auth/me")
    assert r.status_code == 200
    assert r.json()["email"] == "dev@example.com"


# ---------- OAuth callback (Google mocked) ----------

class _FakeGoogle:
    def __init__(self, userinfo):
        self._userinfo = userinfo

    async def authorize_access_token(self, request):
        return {"userinfo": self._userinfo}


class _FakeOAuth:
    def __init__(self, userinfo):
        self.google = _FakeGoogle(userinfo)


def _mock_google(monkeypatch, userinfo):
    monkeypatch.setattr(auth_routes_module, "google_oauth_configured", lambda: True)
    monkeypatch.setattr(auth_routes_module, "oauth", _FakeOAuth(userinfo))


def test_callback_creates_user_and_logs_in(client, monkeypatch):
    _mock_google(
        monkeypatch,
        {"email": "alice@example.com", "email_verified": True, "sub": "g-alice"},
    )

    r = client.get("/auth/callback", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"

    # The session cookie now identifies the user.
    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"


def test_callback_rejects_unverified_email(client, monkeypatch):
    _mock_google(
        monkeypatch,
        {"email": "mallory@example.com", "email_verified": False, "sub": "g-mal"},
    )
    r = client.get("/auth/callback", follow_redirects=False)
    assert r.status_code == 403
    # Not logged in.
    assert client.get("/auth/me").status_code == 401


def test_callback_rejects_missing_email_verified(client, monkeypatch):
    """A missing email_verified claim must fail closed (not be treated as verified)."""
    _mock_google(monkeypatch, {"email": "nev@example.com", "sub": "g-nev"})  # no email_verified
    r = client.get("/auth/callback", follow_redirects=False)
    assert r.status_code == 403
    assert client.get("/auth/me").status_code == 401


def test_logout_clears_session(client, monkeypatch):
    _mock_google(
        monkeypatch,
        {"email": "bob@example.com", "email_verified": True, "sub": "g-bob"},
    )
    client.get("/auth/callback", follow_redirects=False)
    assert client.get("/auth/me").status_code == 200

    out = client.get("/auth/logout", follow_redirects=False)
    assert out.status_code == 303
    assert client.get("/auth/me").status_code == 401


def test_same_email_logs_into_same_account(client, monkeypatch):
    """Logging in twice with the same verified email reuses the one account."""
    _mock_google(
        monkeypatch,
        {"email": "carol@example.com", "email_verified": True, "sub": "g-carol"},
    )
    client.get("/auth/callback", follow_redirects=False)
    uid1 = client.get("/auth/me").json()["id"]
    client.get("/auth/logout", follow_redirects=False)
    client.get("/auth/callback", follow_redirects=False)
    uid2 = client.get("/auth/me").json()["id"]
    assert uid1 == uid2
