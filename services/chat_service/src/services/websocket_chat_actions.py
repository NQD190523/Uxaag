"""Service layer: orchestration when a user sends a chat message over WebSocket
(local broadcast, DB persistence, Redis chat-list hint, optional LLM follow-up)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from shared.logger import get_logger

from services.background_tasks import spawn_background_task
from db.redis_client import redis_client, CHAT_LIST_UPDATES_CHANNEL
from services.chat_service import ChatService
from services.websocket import websocket_service
from services.ws_llm_followup_service import run_llm_followup_and_broadcast
from services.ws_payloads import (
    build_chat_message_payload,
    build_last_message_payload,
    llm_followup_enabled,
)

logger = get_logger("chat-service")


async def handle_ws_user_send_message(
    *,
    chat_room_id: str,
    user_id: Any,
    content: str,
    raw_token: str,
    chat_service: ChatService,
    is_model_prompt: bool = False,
) -> None:
    """Fan out a user message: local WS broadcast, persist, Redis chat-list hint.

    Handles inbound ``action="send_message"``. Emits ``chat_message`` to the room
    (fields in ``ws_payloads.build_chat_message_payload``) and publishes
    ``last_message_updated`` on the Redis chat-list channel.

    The LLM gateway is used only when ``is_model_prompt`` is true and
    ``CHAT_LLM_ENABLED`` is on.

    Args:
        chat_room_id: Room id string from the URL path.
        user_id: Authenticated user id (JWT ``sub``/``user_id`` claim).
        content: Non-empty message text.
        raw_token: The user's JWT, forwarded to the gateway for the LLM call.
        chat_service: Service used for the background DB writes.
        is_model_prompt: True if the message is a prompt for the model.
    """
    now = datetime.now()
    outbound = build_chat_message_payload(
        chat_room_id=chat_room_id,
        user_id=str(user_id),
        content=content,
        created_at=now,
        updated_at=now,
        is_model_prompt=is_model_prompt,
        role="user",
    )
    last_message_outbound = build_last_message_payload(
        chat_room_id=chat_room_id,
        user_id=str(user_id),
        message=content,
        updated_at=now,
    )

    # Broadcast first so the room sees the message without waiting on the DB.
    await websocket_service.broadcast(message=outbound, room_id=chat_room_id)

    # Fire-and-forget DB writes: keep the socket loop fast. Tasks are tracked by
    # spawn_background_task (strong reference + failure logging).
    # The ids come from the URL / JWT claim; guard the int conversions so a
    # malformed id skips persistence instead of crashing the socket loop.
    try:
        room_id_int = int(chat_room_id)
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        logger.warning(
            "Skipping DB persistence: non-numeric ids (room=%r, user=%r)",
            chat_room_id,
            user_id,
        )
    else:
        spawn_background_task(
            chat_service.save_message_background(
                chat_room_id=chat_room_id,
                content=content,
                user_id=user_id_int,
                role="user",
            ),
            name=f"save_message:{chat_room_id}",
        )
        spawn_background_task(
            chat_service.update_chat_room_last_message(
                chat_room_id=room_id_int,
                message=content,
                user_id=user_id_int,
            ),
            name=f"update_last_message:{chat_room_id}",
        )

    # Publish to Redis so every worker pushes the room-list update to its sockets.
    # Failure is non-fatal: the message was already broadcast to this room locally.
    try:
        await redis_client.publish(CHAT_LIST_UPDATES_CHANNEL, last_message_outbound)
    except Exception as pub_err:
        logger.warning(
            "Redis publish failed (chat list updates will not propagate): %s",
            pub_err,
        )

    if is_model_prompt and llm_followup_enabled():
        # The LLM call can take many seconds; run it in the background so the
        # user's socket keeps receiving other frames meanwhile.
        spawn_background_task(
            run_llm_followup_and_broadcast(
                chat_room_id=chat_room_id,
                user_message=content,
                access_token=raw_token,
                chat_service=chat_service,
            ),
            name=f"llm_followup:{chat_room_id}",
        )
