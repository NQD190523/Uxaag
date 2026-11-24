"""Tests for the chat WebSocket's token close code (4401, ``WS_CLOSE_UNAUTHORIZED``).

The browser refreshes its token and reconnects on 4401, but gives up on 1008, so
token failures must never be reported as 1008.

Run from ``services/chat_service``: ``.venv\\Scripts\\python.exe -m pytest tests``.
No DB: both cases are rejected before the membership query runs.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# Imports in this service are relative to src/ (see .claude/rules/backend.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://test:test@127.0.0.1:1/test-only"
)
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret")

from api import websocket as ws_api  # noqa: E402


@pytest.fixture
def client():
    """App with only the WebSocket router mounted like main.py does."""
    app = FastAPI()
    app.include_router(ws_api.router, prefix="/api/v1")
    return TestClient(app)


def _close_code(client, url):
    """Connect, read until the server closes, and return the close code."""
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(url) as ws:
            ws.receive_text()
    return exc.value.code


def test_missing_token_closes_with_4401(client):
    """No token: the client should refresh, not give up."""
    assert _close_code(client, "/api/v1/ws/chat/5") == ws_api.WS_CLOSE_UNAUTHORIZED == 4401


def test_expired_token_closes_with_4401(client):
    """Expired token: 4401, so the UI refreshes and reconnects instead of going dead."""
    expired = jwt.encode(
        {
            "user_id": "7",
            "sub": "7",
            "type": "access",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        },
        os.environ["JWT_SECRET_KEY"],
        algorithm="HS256",
    )
    assert _close_code(client, f"/api/v1/ws/chat/5?token={expired}") == 4401
