"""Tests for the 2026-11-26 security review fixes in chat-service.

1. Room-list updates (``last_message_updated``, which carry message text) reach only
   members of the room they are about, never every open socket.
2. The profile rooms list hides ``latest_message`` and ``description`` of rooms the
   caller is not in.

Runs on the in-memory SQLite harness from ``test_room_management`` (no Redis, no
Supabase). Room ``ROOM`` has Alice and Bob as members; Carol is an outsider.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import pytest

from test_room_management import (  # noqa: F401  (pytest fixtures used by name)
    ALICE,
    BOB,
    CAROL,
    ROOM,
    _as,
    _AsyncSessionAdapter,
    client,
    db,
)

from models.chat import chatParticipant, chatRoom
from services import room_list_fanout
from services.websocket import WebSocketService

OTHER_ROOM = 6


class _FakeSocket:
    """Stand-in WebSocket that records the frames sent to it."""

    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(text)


@pytest.fixture
def fanout(db, monkeypatch):
    """Fresh socket registry, with the fan-out's DB session pointed at SQLite."""
    registry = WebSocketService()

    @asynccontextmanager
    async def session():
        yield _AsyncSessionAdapter(db)

    monkeypatch.setattr(room_list_fanout, "websocket_service", registry)
    monkeypatch.setattr(room_list_fanout, "get_background_session", session)
    return registry


def _frame(room_id, text="secret text"):
    return json.dumps({"action": "last_message_updated", "chat_room_id": str(room_id),
                       "user_id": str(ALICE), "last_message_text": text})


def _connect(registry, room_id, user_id):
    sock = _FakeSocket()
    asyncio.run(registry.connect(sock, str(room_id), user_id=user_id))
    return sock


def test_update_reaches_members_in_any_room_but_not_outsiders(fanout):
    """Bob (member, viewing another room) gets it; Carol (outsider) does not."""
    alice = _connect(fanout, ROOM, str(ALICE))
    bob_elsewhere = _connect(fanout, OTHER_ROOM, str(BOB))
    carol = _connect(fanout, OTHER_ROOM, str(CAROL))

    delivered = asyncio.run(room_list_fanout.fan_out_room_list_update(_frame(ROOM)))

    assert delivered == 2
    assert alice.sent and bob_elsewhere.sent
    assert carol.sent == []


def test_member_who_left_no_longer_receives(fanout, db):
    bob = _connect(fanout, OTHER_ROOM, str(BOB))
    row = db.query(chatParticipant).filter_by(chat_room_id=ROOM, user_id=BOB).one()
    row.is_active = False
    db.commit()
    asyncio.run(room_list_fanout.fan_out_room_list_update(_frame(ROOM)))
    assert bob.sent == []


def test_socket_without_user_and_malformed_frames_get_nothing(fanout):
    """Fail closed: an anonymous socket and frames without a room id are skipped."""
    anon = _FakeSocket()
    asyncio.run(fanout.connect(anon, str(ROOM)))  # registered without user_id
    alice = _connect(fanout, ROOM, str(ALICE))
    for bad in ("not json", json.dumps({"action": "x"}), json.dumps({"chat_room_id": "abc"})):
        assert asyncio.run(room_list_fanout.fan_out_room_list_update(bad)) == 0
    asyncio.run(room_list_fanout.fan_out_room_list_update(_frame(ROOM)))
    assert anon.sent == [] and len(alice.sent) == 1


def test_disconnect_forgets_the_socket_user(fanout):
    alice = _connect(fanout, ROOM, str(ALICE))
    fanout.disconnect(alice, str(ROOM))
    assert alice not in fanout.socket_users
    assert asyncio.run(room_list_fanout.fan_out_room_list_update(_frame(ROOM))) == 0


# --- Profile rooms --------------------------------------------------------------------


@pytest.fixture
def room_content(db):
    """Give the room a description and a last message preview."""
    room = db.get(chatRoom, ROOM)
    room.description = "private description"
    room.last_message_id = uuid.uuid4()  # UUID columns (see models/chat.py)
    room.last_message_text = "private last message"
    room.last_message_sender_id = uuid.uuid4()
    room.last_message_at = datetime(2026, 1, 2)
    db.commit()


def _profile_rooms(client, profile, caller):
    res = client.get(f"/api/v1/chat/users/{profile}/chat-rooms", headers=_as(caller))
    assert res.status_code == 200
    return res.json()


def test_outsider_sees_room_listed_but_no_content(client, room_content):
    rooms = _profile_rooms(client, ALICE, CAROL)
    assert [r["chat_room_id"] for r in rooms] == [ROOM]
    assert rooms[0]["latest_message"] is None
    assert rooms[0]["description"] is None
    assert rooms[0]["chat_room_name"] == "Old name"  # the list itself is unchanged


@pytest.mark.parametrize("caller", [ALICE, BOB])
def test_self_and_fellow_member_see_content(client, room_content, caller):
    rooms = _profile_rooms(client, ALICE, caller)
    assert rooms[0]["latest_message"]["message"] == "private last message"
    assert rooms[0]["description"] == "private description"
