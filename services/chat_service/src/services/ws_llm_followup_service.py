"""Service layer: call the RAG LLM through the API gateway and broadcast the
assistant reply to the room."""

from __future__ import annotations

from datetime import datetime

from shared.logger import get_logger

from services.background_tasks import spawn_background_task
from clients.rag_llm_gateway_client import post_llm_chat_via_api_gateway
from services.chat_service import ChatService
from services.websocket import websocket_service
from services.ws_payloads import (
    build_llm_error_payload,
    build_llm_reply_payload,
    default_llm_system_prompt,
)

logger = get_logger("chat-service")


async def run_llm_followup_and_broadcast(
    *,
    chat_room_id: str,
    user_message: str,
    access_token: str,
    chat_service: ChatService,
) -> None:
    """POST to gateway ``/rag/llm/chat``, then broadcast the result to the room.

    Cross-service call: chat-service -> API gateway -> rag-service. Emits
    ``llm_reply`` on success (fields in ``ws_payloads.build_llm_reply_payload``) or
    ``llm_error`` on failure. An empty model reply is treated as a failure and also emits ``llm_error``.
    Never raises; errors are logged and reported to the room.

    Args:
        chat_room_id: Room id string.
        user_message: The prompt text the user sent.
        access_token: The user's JWT, forwarded so the gateway can authenticate.
        chat_service: Service used to persist the assistant message.
    """
    try:
        data = await post_llm_chat_via_api_gateway(
            user_message=user_message,
            access_token=access_token,
            system_prompt=default_llm_system_prompt(),
        )
        reply = (data.get("reply") or "").strip()
        if not reply:
            logger.warning("LLM returned empty reply room_id=%s", chat_room_id)
            # Do not leave the room waiting for a reply that will never come.
            await websocket_service.broadcast(
                message=build_llm_error_payload(
                    chat_room_id=chat_room_id,
                    detail="Assistant returned an empty reply. Try again.",
                ),
                room_id=chat_room_id,
            )
            return
        now = datetime.now()
        outbound = build_llm_reply_payload(
            chat_room_id=chat_room_id,
            content=reply,
            created_at=now,
        )
        await websocket_service.broadcast(message=outbound, room_id=chat_room_id)
        logger.info("LLM reply broadcast room_id=%s reply_length=%d", chat_room_id, len(reply))
        # Persist assistant response asynchronously, same non-blocking style as user
        # messages (tracked fire-and-forget; failures are logged by the helper).
        try:
            room_id_int = int(chat_room_id)
        except ValueError:
            logger.warning(
                "LLM reply not persisted: non-numeric room_id=%s",
                chat_room_id,
            )
            return
        spawn_background_task(
            chat_service.save_message_background(
                chat_room_id=chat_room_id,
                content=reply,
                user_id=None,
                role="assistant",
            ),
            name=f"save_llm_reply:{chat_room_id}",
        )
        spawn_background_task(
            chat_service.update_chat_room_last_message(
                chat_room_id=room_id_int,
                message=reply,
                # Assistant has no user id; 0 is a placeholder (the method ignores it).
                user_id=0,
            ),
            name=f"update_last_message_llm:{chat_room_id}",
        )
    except Exception:
        logger.exception("LLM follow-up failed room_id=%s", chat_room_id)
        # Tell the room the assistant failed; the detail is generic on purpose so
        # internal errors (URLs, tokens) are not leaked to clients.
        try:
            await websocket_service.broadcast(
                message=build_llm_error_payload(
                    chat_room_id=chat_room_id,
                    detail="Assistant could not be reached. Try again later.",
                ),
                room_id=chat_room_id,
            )
        except Exception as broadcast_err:
            logger.warning(
                "Failed to broadcast llm_error room_id=%s error=%s",
                chat_room_id,
                type(broadcast_err).__name__,
            )
