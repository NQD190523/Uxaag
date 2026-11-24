"""Tests for the document-service room-membership guard (bug D1).

Run from ``services/document_service``: ``.venv\\Scripts\\python.exe -m pytest tests``.
chat-service and the gateway are never called: ``fetch_room_membership`` is replaced
by a fake, and the guard is mounted on a tiny app so no storage or PDF code runs.
"""

import sys
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

# Imports in this service are relative to src/ (see .claude/rules/backend.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.dependencies import require_room_member  # noqa: E402
from clients.chat_membership_client import (  # noqa: E402
    MembershipAuthError,
    MembershipCheckError,
)
from services import room_access  # noqa: E402

AUTH = {"x-user-id": "7", "Authorization": "Bearer test-token"}


def _app() -> FastAPI:
    """Return an app with one guarded route shaped like the real room routes."""
    app = FastAPI()

    @app.get("/chat-room/{chat_room_id}/tree")
    async def tree(chat_room_id: int, x_user_id: str = Depends(require_room_member)):
        return {"room": chat_room_id, "user": x_user_id}

    return app


@pytest.fixture
def fake_chat(monkeypatch):
    """Replace the chat-service call; returns a dict to steer it and count calls."""
    state = {"answer": True, "error": None, "calls": 0}

    async def fake_fetch(*, chat_room_id, authorization):
        state["calls"] += 1
        assert authorization == "Bearer test-token"  # token is forwarded unchanged
        if state["error"]:
            raise state["error"]
        return state["answer"]

    room_access.clear_membership_cache()
    monkeypatch.setattr(room_access, "fetch_room_membership", fake_fetch)
    monkeypatch.setenv("MEMBERSHIP_CACHE_SECONDS", "30")
    yield state
    room_access.clear_membership_cache()


def test_member_is_allowed(fake_chat):
    """A member gets the handler's response and their user id."""
    res = TestClient(_app()).get("/chat-room/5/tree", headers=AUTH)
    assert res.status_code == 200
    assert res.json() == {"room": 5, "user": "7"}


def test_non_member_gets_403(fake_chat):
    """A non-member is refused."""
    fake_chat["answer"] = False
    res = TestClient(_app()).get("/chat-room/5/tree", headers=AUTH)
    assert res.status_code == 403


def test_chat_unavailable_fails_closed(fake_chat):
    """If chat-service cannot answer, the request is refused with 503 (not allowed)."""
    fake_chat["error"] = MembershipCheckError("ConnectError")
    res = TestClient(_app()).get("/chat-room/5/tree", headers=AUTH)
    assert res.status_code == 503


def test_rejected_token_gives_401(fake_chat):
    """A 401 from the gateway is passed on so the UI can refresh its token."""
    fake_chat["error"] = MembershipAuthError("expired")
    res = TestClient(_app()).get("/chat-room/5/tree", headers=AUTH)
    assert res.status_code == 401


def test_missing_bearer_gives_401_without_calling_chat(fake_chat):
    """No token means no lookup at all."""
    res = TestClient(_app()).get("/chat-room/5/tree", headers={"x-user-id": "7"})
    assert res.status_code == 401
    assert fake_chat["calls"] == 0


def test_positive_answer_is_cached(fake_chat):
    """Two calls for the same user and room cost one chat-service lookup."""
    client = TestClient(_app())
    assert client.get("/chat-room/5/tree", headers=AUTH).status_code == 200
    assert client.get("/chat-room/5/tree", headers=AUTH).status_code == 200
    assert fake_chat["calls"] == 1


def test_negative_answer_is_not_cached(fake_chat):
    """A user added to the room after a 403 gets access on the next call."""
    client = TestClient(_app())
    fake_chat["answer"] = False
    assert client.get("/chat-room/5/tree", headers=AUTH).status_code == 403
    fake_chat["answer"] = True
    assert client.get("/chat-room/5/tree", headers=AUTH).status_code == 200
    assert fake_chat["calls"] == 2


def test_cache_disabled_with_zero_ttl(fake_chat, monkeypatch):
    """MEMBERSHIP_CACHE_SECONDS=0 asks chat-service every time."""
    monkeypatch.setenv("MEMBERSHIP_CACHE_SECONDS", "0")
    client = TestClient(_app())
    client.get("/chat-room/5/tree", headers=AUTH)
    client.get("/chat-room/5/tree", headers=AUTH)
    assert fake_chat["calls"] == 2
