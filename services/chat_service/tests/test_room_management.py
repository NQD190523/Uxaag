"""Tests for room management (P2): leave/archive, rename, re-add and history paging.

Run from ``services/chat_service``: ``.venv\\Scripts\\python.exe -m pytest tests``.

The real router -> service -> repository code runs against an in-memory SQLite DB
(never Supabase). The repositories expect an async session, and aiosqlite is not
installed, so ``_AsyncSessionAdapter`` wraps a sync SQLModel session in coroutines.
A minimal ``"user"`` table stands in for the auth table that the participants query
reads with raw SQL.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

# Imports in this service are relative to src/ (see .claude/rules/backend.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://test:test@127.0.0.1:1/test-only"
)
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret")

from contextlib import asynccontextmanager  # noqa: E402

import jwt  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from api import endpoints  # noqa: E402
from api import websocket as ws_api  # noqa: E402
from services import ws_auth  # noqa: E402
from db.database import get_async_session  # noqa: E402
from models.chat import chatMessage, chatParticipant, chatRoom  # noqa: E402

ROOM = 5
ALICE, BOB, CAROL = 7, 8, 9


class _AsyncSessionAdapter:
    """Expose a sync SQLModel ``Session`` through the async methods the code awaits."""

    def __init__(self, session: Session):
        self._s = session

    def add(self, obj):
        self._s.add(obj)

    async def execute(self, *args, **kwargs):
        return self._s.execute(*args, **kwargs)

    async def exec(self, *args, **kwargs):
        return self._s.exec(*args, **kwargs)

    async def commit(self):
        self._s.commit()

    async def rollback(self):
        self._s.rollback()

    async def refresh(self, obj):
        self._s.refresh(obj)


@pytest.fixture
def db():
    """Fresh in-memory DB with one room (Alice + Bob active) and 120 messages."""
    # StaticPool: every connection is the same in-memory DB (TestClient uses threads).
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    for model in (chatRoom, chatMessage, chatParticipant):
        model.__table__.create(engine)
    with engine.begin() as conn:
        conn.execute(text(
            'CREATE TABLE "user" (user_id INTEGER PRIMARY KEY, username TEXT, '
            "email TEXT, full_name TEXT, last_login TIMESTAMP)"
        ))
    session = Session(engine, expire_on_commit=False)
    session.add(chatRoom(chat_room_id=ROOM, chat_room_name="Old name"))
    for uid in (ALICE, BOB):
        session.add(chatParticipant(chat_room_id=ROOM, user_id=uid))
    start = datetime(2026, 1, 1)
    for i in range(120):
        # Pairs share a timestamp so the (created_at, id) tie-break is exercised.
        session.add(chatMessage(
            chat_message_id=1000 + i,
            chat_room_id=ROOM,
            user_id=ALICE,
            role="user",
            message=f"m{i}",
            created_at=start + timedelta(seconds=i // 2),
        ))
    session.commit()
    yield session
    session.close()


@pytest.fixture
def client(db):
    """App with the chat router; the session dependency yields the SQLite adapter."""

    async def sqlite_session():
        yield _AsyncSessionAdapter(db)

    app = FastAPI()
    app.include_router(endpoints.router, prefix="/api/v1")
    app.dependency_overrides[get_async_session] = sqlite_session
    return TestClient(app)


def _as(uid):
    """Headers the gateway would inject for user ``uid``."""
    return {"x-user-id": str(uid)}


# --- Paging ---------------------------------------------------------------------------


def test_history_returns_newest_page_with_has_more(client):
    """Opening a room loads the newest 50 messages, oldest first, and flags more."""
    res = client.get(f"/api/v1/chat/messages/{ROOM}", headers=_as(ALICE))
    assert res.status_code == 200
    body = res.json()
    assert [m["message"] for m in body["messages"]] == [f"m{i}" for i in range(70, 120)]
    assert body["has_more"] is True


def test_older_pages_walk_back_without_gaps_or_duplicates(client):
    """Following ``before`` cursors returns every message exactly once."""
    seen = [m["message"] for m in client.get(
        f"/api/v1/chat/messages/{ROOM}", headers=_as(ALICE)).json()["messages"]]
    oldest_id = None
    first = client.get(f"/api/v1/chat/messages/{ROOM}", headers=_as(ALICE)).json()
    oldest_id = first["messages"][0]["id"]
    has_more = first["has_more"]
    while has_more:
        page = client.get(
            f"/api/v1/chat/rooms/{ROOM}/messages",
            params={"before": oldest_id, "limit": 50},
            headers=_as(ALICE),
        ).json()
        seen = [m["message"] for m in page["messages"]] + seen
        oldest_id = page["messages"][0]["id"]
        has_more = page["has_more"]
    assert seen == [f"m{i}" for i in range(120)]


def test_before_from_another_room_is_400(client):
    """A cursor that is not a message of this room is rejected."""
    res = client.get(
        f"/api/v1/chat/rooms/{ROOM}/messages", params={"before": 1}, headers=_as(ALICE)
    )
    assert res.status_code == 400


def test_paging_requires_membership(client):
    """Non-members cannot page a room's history."""
    res = client.get(
        f"/api/v1/chat/rooms/{ROOM}/messages", params={"before": 1100}, headers=_as(CAROL)
    )
    assert res.status_code == 403


def test_limit_is_bounded(client):
    """``limit`` outside 1..200 is a validation error."""
    res = client.get(
        f"/api/v1/chat/messages/{ROOM}", params={"limit": 500}, headers=_as(ALICE)
    )
    assert res.status_code == 422


# --- Rename ---------------------------------------------------------------------------


def test_name_is_trimmed_before_length_check(client):
    """Padding does not count toward the 100-char limit; 101 real chars is too long."""
    padded = "  " + "x" * 100 + "  "
    ok = client.patch(f"/api/v1/chat/rooms/{ROOM}", json={"chat_room_name": padded},
                      headers=_as(ALICE))
    assert ok.status_code == 200 and ok.json()["chat_room_name"] == "x" * 100
    too_long = client.patch(f"/api/v1/chat/rooms/{ROOM}", json={"chat_room_name": "x" * 101},
                            headers=_as(ALICE))
    assert too_long.status_code == 422


def test_member_can_rename(client, db):
    """A member renames the room; the name is trimmed and stored."""
    res = client.patch(
        f"/api/v1/chat/rooms/{ROOM}", json={"chat_room_name": "  New name "}, headers=_as(BOB)
    )
    assert res.status_code == 200
    assert res.json()["chat_room_name"] == "New name"
    assert db.get(chatRoom, ROOM).chat_room_name == "New name"


def test_blank_name_and_empty_body_are_400(client):
    """Blank names and no-op bodies are rejected."""
    assert client.patch(f"/api/v1/chat/rooms/{ROOM}", json={"chat_room_name": "  "},
                        headers=_as(ALICE)).status_code == 400
    assert client.patch(f"/api/v1/chat/rooms/{ROOM}", json={},
                        headers=_as(ALICE)).status_code == 400


def test_non_member_cannot_rename(client):
    """Only members can edit the room."""
    res = client.patch(
        f"/api/v1/chat/rooms/{ROOM}", json={"chat_room_name": "x"}, headers=_as(CAROL)
    )
    assert res.status_code == 403


# --- Leave / archive / re-add -----------------------------------------------------------


def test_leave_then_last_leave_archives(client, db):
    """First leave keeps the room; the last member leaving archives it."""
    first = client.post(f"/api/v1/chat/rooms/{ROOM}/leave", headers=_as(ALICE))
    assert first.status_code == 200 and first.json()["room_archived"] is False
    assert db.get(chatRoom, ROOM).is_active is True

    last = client.post(f"/api/v1/chat/rooms/{ROOM}/leave", headers=_as(BOB))
    assert last.status_code == 200 and last.json()["room_archived"] is True
    assert db.get(chatRoom, ROOM).is_active is False


def test_leaving_twice_is_403(client):
    """A user who already left is no longer a participant."""
    client.post(f"/api/v1/chat/rooms/{ROOM}/leave", headers=_as(ALICE))
    assert client.post(f"/api/v1/chat/rooms/{ROOM}/leave",
                       headers=_as(ALICE)).status_code == 403


def test_left_user_loses_access_and_room_leaves_their_list(client):
    """After leaving, history is forbidden and the room is gone from the user's list."""
    client.post(f"/api/v1/chat/rooms/{ROOM}/leave", headers=_as(ALICE))
    assert client.get(f"/api/v1/chat/messages/{ROOM}",
                      headers=_as(ALICE)).status_code == 403
    rooms = client.get("/api/v1/chat/user-chat-rooms", headers=_as(ALICE)).json()
    assert rooms == []


def test_user_who_left_can_be_added_back(client, db):
    """Re-adding reactivates the old row instead of 'User already in chat room'."""
    client.post(f"/api/v1/chat/rooms/{ROOM}/leave", headers=_as(ALICE))
    res = client.post(
        f"/api/v1/chat/add-user-to-chat-room/{ROOM}", json={"user_id": ALICE}, headers=_as(BOB)
    )
    assert res.status_code == 200
    rows = db.exec(select(chatParticipant).where(chatParticipant.user_id == ALICE)).all()
    assert len(rows) == 1 and rows[0].is_active is True and rows[0].left_at is None


def test_adding_an_active_member_is_still_400(client):
    """The existing duplicate check still applies to active members."""
    res = client.post(
        f"/api/v1/chat/add-user-to-chat-room/{ROOM}", json={"user_id": BOB}, headers=_as(ALICE)
    )
    assert res.status_code == 400


# --- WebSocket after leaving ----------------------------------------------------------------


def test_socket_of_user_who_left_cannot_send(client, db, monkeypatch):
    """A socket opened while a member is closed with 1008 on its next message after leaving.

    Only the refusal path runs (no message is accepted), so no Redis or background DB
    writes are triggered.
    """

    @asynccontextmanager
    async def sqlite_background_session():
        yield _AsyncSessionAdapter(db)

    monkeypatch.setattr(ws_api, "get_background_session", sqlite_background_session)
    app = FastAPI()
    app.include_router(ws_api.router, prefix="/api/v1")
    app.include_router(endpoints.router, prefix="/api/v1")

    async def sqlite_session():
        yield _AsyncSessionAdapter(db)

    app.dependency_overrides[get_async_session] = sqlite_session
    ws_client = TestClient(app)
    token = jwt.encode(
        {"user_id": str(ALICE), "sub": str(ALICE), "type": "access",
         "exp": datetime.utcnow() + timedelta(minutes=5)},
        ws_auth.JWT_SECRET_KEY,
        algorithm="HS256",
    )
    with pytest.raises(WebSocketDisconnect) as exc:
        with ws_client.websocket_connect(f"/api/v1/ws/chat/{ROOM}?token={token}") as ws:
            # Alice leaves from "another tab" while this socket stays open.
            assert ws_client.post(f"/api/v1/chat/rooms/{ROOM}/leave",
                                  headers=_as(ALICE)).status_code == 200
            ws.send_text('{"action": "send_message", "content": "still here?"}')
            ws.receive_text()
    assert exc.value.code == 1008
    assert db.exec(select(chatMessage).where(chatMessage.message == "still here?")).all() == []


# --- Own membership status -------------------------------------------------------------------


def test_own_status_is_active_without_recent_login(client):
    """A member with no last_login (e.g. just signed up) is not labelled as having left.

    The fixture's "user" table is empty, so the old presence-based rule returned
    is_active=False ("You left this room") for every member.
    """
    history = client.get(f"/api/v1/chat/messages/{ROOM}", headers=_as(ALICE)).json()
    assert history["current_user_status"]["is_member"] is True
    assert history["current_user_status"]["is_active"] is True
    rooms = client.get("/api/v1/chat/user-chat-rooms", headers=_as(ALICE)).json()
    assert rooms[0]["current_user_status"]["is_active"] is True
