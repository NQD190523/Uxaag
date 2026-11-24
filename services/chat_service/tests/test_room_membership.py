"""Tests for ``GET /api/v1/chat/rooms/{chat_room_id}/membership``.

Run from ``services/chat_service``: ``.venv\\Scripts\\python.exe -m pytest tests``.
The DB is never touched: the session dependency is overridden and
``ChatService.is_participant`` is replaced by a fake. A dummy ``DATABASE_URL`` is set
only because ``db.database`` builds its (lazy, never connected) engine at import.
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Imports in this service are relative to src/ (see .claude/rules/backend.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://test:test@127.0.0.1:1/test-only"
)

from api import endpoints  # noqa: E402
from db.database import get_async_session  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    """App with only the chat router, no DB, and a steerable membership answer."""
    members = {(5, 7)}  # (room, user) pairs that are members

    async def fake_is_participant(chat_room_id, user_id, session):
        return (int(chat_room_id), int(user_id)) in members

    async def no_session():
        yield None

    monkeypatch.setattr(endpoints.chat_service, "is_participant", fake_is_participant)
    app = FastAPI()
    app.include_router(endpoints.router, prefix="/api/v1")
    app.dependency_overrides[get_async_session] = no_session
    return TestClient(app)


def test_member(client):
    """An active participant gets is_member=true."""
    res = client.get("/api/v1/chat/rooms/5/membership", headers={"x-user-id": "7"})
    assert res.status_code == 200
    assert res.json() == {"chat_room_id": 5, "user_id": 7, "is_member": True}


def test_non_member_is_200_false(client):
    """A non-member gets 200 with is_member=false (not 403), see the endpoint docstring."""
    res = client.get("/api/v1/chat/rooms/5/membership", headers={"x-user-id": "8"})
    assert res.status_code == 200
    assert res.json()["is_member"] is False


def test_missing_identity_header(client):
    """Without the gateway's x-user-id the request is invalid."""
    assert client.get("/api/v1/chat/rooms/5/membership").status_code == 422


def test_non_numeric_room(client):
    """Room id must be numeric."""
    res = client.get("/api/v1/chat/rooms/abc/membership", headers={"x-user-id": "7"})
    assert res.status_code == 422
