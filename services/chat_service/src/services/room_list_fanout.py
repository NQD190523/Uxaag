"""Room-list update fan-out (service layer): deliver ``last_message_updated`` frames
only to members of the room they are about.

Frames arrive from the global Redis channel (``db/redis_client.py``). They carry the
message text, so sending them to every open socket leaked messages to users outside
the room (security review, 2026-11-26). Each frame is now sent only to sockets whose
user is an active participant of the frame's ``chat_room_id``, in whatever room those
sockets have open. The frame format is unchanged, so the UI needs no change.
"""

from __future__ import annotations

import json

from shared.logger import get_logger

from db.database import get_background_session
from repositories.chat_repository import ChatRepository
from services.websocket import websocket_service

logger = get_logger("chat-service")


def _room_id_of(raw: str) -> int | None:
    """Return the frame's ``chat_room_id`` as int, or None if the frame is malformed."""
    try:
        data = json.loads(raw)
        return int(data["chat_room_id"])
    except (ValueError, TypeError, KeyError):
        return None


async def fan_out_room_list_update(raw: str) -> int:
    """Send one chat-list update to the local sockets of the room's members.

    Fails closed: a frame without a valid ``chat_room_id`` is dropped, and a database
    error propagates (the listener logs it) instead of falling back to everyone.

    Args:
        raw: JSON string published by ``websocket_chat_actions`` (a
            ``last_message_updated`` payload).

    Returns:
        Number of sockets the frame was sent to.
    """
    # No open sockets on this worker: nothing to deliver, skip the DB round trip.
    if not websocket_service.has_connections():
        return 0
    chat_room_id = _room_id_of(raw)
    if chat_room_id is None:
        logger.warning("Room-list update dropped: no valid chat_room_id")
        return 0
    async with get_background_session() as session:
        member_ids = await ChatRepository(session=session).get_active_participant_user_ids(
            chat_room_id=chat_room_id
        )
    return await websocket_service.broadcast_to_users(
        message=raw, user_ids={str(uid) for uid in member_ids}
    )
