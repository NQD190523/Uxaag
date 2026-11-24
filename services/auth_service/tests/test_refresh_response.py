"""Tests for the refresh response shape (bug A4).

The router blanks ``token.refresh_token`` (the rotated token only travels in the
HttpOnly cookie), so ``TokenResponse.refresh_token`` is ``Optional``. These tests pin
that contract through a real ``response_model`` round trip and through validation
(``model_validate``), which rejects ``None`` if the field goes back to plain ``str``.
No DB or JWT code runs.

Run from ``services/auth_service``: ``.venv\\Scripts\\python.exe -m pytest tests``.
"""

import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Imports in this service are relative to src/ (see .claude/rules/backend.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schemas.auth_schema import AuthResponse, TokenResponse, UserProfile  # noqa: E402


def _app() -> FastAPI:
    """Tiny app whose route mirrors what the refresh router returns."""
    app = FastAPI()

    @app.post("/refresh", response_model=AuthResponse)
    async def refresh():
        result = AuthResponse(
            success=True,
            message="Token refreshed successfully",
            user=None,
            token=TokenResponse(
                access_token="a", refresh_token="r", expires_in=1800.0
            ),
        )
        # Same mutation as endpoints.refresh_token_endpoint.
        result.token.refresh_token = None
        return result

    return app


def test_refresh_response_with_blank_refresh_token_is_valid():
    """The response serialises with refresh_token null instead of raising a 500."""
    res = TestClient(_app()).post("/refresh")
    assert res.status_code == 200
    assert res.json()["token"]["refresh_token"] is None
    assert res.json()["token"]["access_token"] == "a"


def test_login_shape_still_carries_refresh_token():
    """Login/register still serialise the refresh token when it is set."""
    token = TokenResponse(access_token="a", refresh_token="r", expires_in=1.0)
    profile = UserProfile(user_id="1", email="u@example.com", created_at=datetime(2026, 1, 1))
    body = AuthResponse(success=True, message="ok", user=profile, token=token)
    assert body.model_dump()["token"]["refresh_token"] == "r"


def test_schema_accepts_null_refresh_token():
    """Validation (not just the router's mutation) must accept a null refresh token."""
    token = TokenResponse.model_validate(
        {"access_token": "a", "refresh_token": None, "expires_in": 1.0}
    )
    assert token.refresh_token is None
