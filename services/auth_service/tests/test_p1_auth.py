"""Tests for username handles (bug A7), refresh-token hashing and refresh claims (A4).

Run from ``services/auth_service``: ``.venv\\Scripts\\python.exe -m pytest tests``.
No DB: repositories are replaced by in-memory fakes and the session is a stub.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest

# Imports in this service are relative to src/ (see .claude/rules/backend.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.security import SecurityUtil  # noqa: E402
from repositories import refresh_token_repository as rt_repo  # noqa: E402
from services import auth_service as auth_mod  # noqa: E402
from services.auth_service import AuthService, _handle_base  # noqa: E402

SECRET = "test-only-secret"


def _service() -> AuthService:
    """AuthService with a real SecurityUtil using a throwaway secret."""
    return AuthService(
        security_util=SecurityUtil(
            jwt_secret_key=SECRET,
            access_token_expire_minutes=30,
            refresh_token_expire_days=7,
        )
    )


# --- A7: username handles ------------------------------------------------------------


@pytest.mark.parametrize(
    "email, expected",
    [
        ("Nguyen.Duc+work@example.com", "nguyen.duc"),
        ("a b!c@example.com", "abc"),
        ("...@example.com", "user"),
        ("x" * 80 + "@example.com", "x" * 30),
    ],
)
def test_handle_base(email, expected):
    """Local part is lower-cased, cleaned, stripped of +tags and capped at 30 chars."""
    assert _handle_base(email) == expected


class _FakeUsers:
    """UserRepository stand-in: ``taken`` is the set of usernames already in use."""

    def __init__(self, taken):
        self.taken = set(taken)

    async def get_user_by_username(self, username):
        return object() if username in self.taken else None


def test_free_handle_is_used_as_is():
    """No collision: the plain handle is returned."""
    name = asyncio.run(_service()._unique_username("duc@x.com", _FakeUsers([])))
    assert name == "duc"


def test_taken_handle_gets_a_suffix():
    """Same local part as an existing user: a different, suffixed handle is returned."""
    name = asyncio.run(_service()._unique_username("duc@y.com", _FakeUsers(["duc"])))
    assert name != "duc" and name.startswith("duc-")


def test_two_users_with_same_full_name_get_different_usernames():
    """The A7 scenario: same full name, different emails, no UNIQUE collision."""
    users = _FakeUsers([])
    first = asyncio.run(_service()._unique_username("an.nguyen@x.com", users))
    users.taken.add(first)
    second = asyncio.run(_service()._unique_username("an.nguyen@y.com", users))
    assert first != second


def test_everything_taken_falls_back_to_hex(monkeypatch):
    """When every random suffix is taken the hex fallback is used."""
    monkeypatch.setattr(auth_mod.secrets, "randbelow", lambda n: 0)  # always "duc-1000"
    name = asyncio.run(
        _service()._unique_username("duc@x.com", _FakeUsers(["duc", "duc-1000"]))
    )
    assert name.startswith("duc-") and len(name) == len("duc-") + 8


# --- Refresh-token hashing -------------------------------------------------------------


def test_hash_is_prefixed_and_not_the_token():
    """Stored value is "sha256:<64 hex>" and never contains the raw token."""
    raw = SecurityUtil.generate_random_refresh_token()
    stored = SecurityUtil.hash_refresh_token(raw)
    assert stored.startswith("sha256:") and len(stored) == 7 + 64
    assert raw not in stored
    assert stored == SecurityUtil.hash_refresh_token(raw)  # deterministic for lookup


class _FakeSession:
    """Async session stub that records added rows."""

    def __init__(self):
        self.added = []

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        pass

    async def refresh(self, row):
        pass

    async def rollback(self):
        pass


def test_repository_stores_only_the_hash():
    """create() persists the digest, not the raw token."""
    session = _FakeSession()
    raw = "raw-token-value"
    asyncio.run(
        rt_repo.RefreshTokenRepository(session).create(1, raw, datetime(2030, 1, 1))
    )
    assert session.added[0].token == SecurityUtil.hash_refresh_token(raw)


def test_lookup_matches_hashed_and_legacy_rows():
    """The WHERE clause accepts the hashed value and the legacy plain value."""
    raw = "raw-token-value"
    compiled = rt_repo._matches(raw).compile(compile_kwargs={"literal_binds": True})
    sql = str(compiled)
    assert SecurityUtil.hash_refresh_token(raw) in sql
    assert f"'{raw}'" in sql
    # The legacy branch must exclude hashed rows (see the next test).
    assert "NOT LIKE 'sha256:'" in sql


def _lookup(stored_values, presented):
    """Run the real ``_matches`` condition on in-memory SQLite; return matching tokens."""
    from sqlalchemy import create_engine, insert, select

    from models.user import RefreshToken

    engine = create_engine("sqlite://")
    RefreshToken.__table__.create(engine)
    with engine.begin() as conn:
        for i, value in enumerate(stored_values, start=1):
            conn.execute(
                insert(RefreshToken.__table__).values(
                    token_id=i, user_id=1, token=value,
                    expires_at=datetime(2030, 1, 1), revoked=False,
                    created_at=datetime(2026, 1, 1),
                )
            )
        rows = conn.execute(
            select(RefreshToken.__table__.c.token).where(rt_repo._matches(presented))
        )
        return [r[0] for r in rows]


def test_raw_token_finds_hashed_row():
    """The normal case: the cookie's raw token finds its hashed row."""
    stored = SecurityUtil.hash_refresh_token("raw-a")
    assert _lookup([stored], "raw-a") == [stored]


def test_raw_token_finds_legacy_row():
    """A token issued before hashing (stored raw) still works."""
    assert _lookup(["legacy-raw"], "legacy-raw") == ["legacy-raw"]


def test_stored_digest_is_not_a_valid_token():
    """A digest copied from a DB leak must not match its own row (reviewer finding)."""
    stored = SecurityUtil.hash_refresh_token("raw-a")
    assert _lookup([stored, "legacy-raw"], stored) == []


# --- Refresh logic (claims and user checks) ---------------------------------------------


class _FakeRefreshRepo:
    """RefreshTokenRepository stand-in with one stored, valid token."""

    record = None
    created = []

    def __init__(self, session):
        pass

    async def get_by_token(self, token):
        return self.record if token == "good" else None

    async def create(self, user_id, token, expires_at):
        _FakeRefreshRepo.created.append(token)

    async def update_last_used(self, token):
        pass

    async def revoke_by_token(self, token):
        pass


class _FakeUserRepo:
    """UserRepository stand-in returning ``user`` for any id."""

    user = None

    def __init__(self, session):
        pass

    async def get_user_by_id(self, user_id):
        return self.user


@pytest.fixture
def fakes(monkeypatch):
    """Patch both repositories used by AuthService.refresh_token."""
    _FakeRefreshRepo.record = SimpleNamespace(
        user_id=42,
        token="sha256:stored",
        revoked=False,
        expires_at=datetime.utcnow() + timedelta(days=1),
    )
    _FakeRefreshRepo.created = []
    _FakeUserRepo.user = SimpleNamespace(user_id=42, email="u@example.com", is_active=True)
    monkeypatch.setattr(auth_mod, "RefreshTokenRepository", _FakeRefreshRepo)
    monkeypatch.setattr(auth_mod, "UserRepository", _FakeUserRepo)


def test_refreshed_token_has_login_claims(fakes):
    """The new access token carries user_id, email and sub like a login token."""
    result = asyncio.run(_service().refresh_token("good", session=None))
    claims = jwt.decode(result.token.access_token, SECRET, algorithms=["HS256"])
    assert claims["user_id"] == "42"
    assert claims["sub"] == "42"
    assert claims["email"] == "u@example.com"
    assert claims["type"] == "access"
    assert len(_FakeRefreshRepo.created) == 1  # rotated


def test_disabled_user_cannot_refresh(fakes):
    """A disabled account is rejected even with a valid refresh token."""
    _FakeUserRepo.user.is_active = False
    with pytest.raises(ValueError, match="disabled"):
        asyncio.run(_service().refresh_token("good", session=None))
    assert _FakeRefreshRepo.created == []


def test_deleted_user_cannot_refresh(fakes):
    """A token whose user no longer exists is rejected."""
    _FakeUserRepo.user = None
    with pytest.raises(ValueError, match="User not found"):
        asyncio.run(_service().refresh_token("good", session=None))


def test_unknown_token_is_rejected(fakes):
    """An unknown token never reaches the user lookup."""
    with pytest.raises(ValueError, match="Invalid refresh token"):
        asyncio.run(_service().refresh_token("bad", session=None))
