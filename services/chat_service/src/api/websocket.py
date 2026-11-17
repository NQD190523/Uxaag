"""WebSocket router (router layer): the chat socket endpoint only.

Business logic lives under ``services/``. Mounted at ``/api/v1/ws``; browsers connect
directly to this service (WebSockets do not go through the API gateway).
"""

import json
from typing import Optional

from fastapi import APIRouter, Query, WebSocket
from starlette.websockets import WebSocketDisconnect

from shared.logger import get_logger

from db.database import get_background_session
from repositories.chat_repository import ChatRepository
from services.chat_service import ChatService
from services.websocket import websocket_service
from services.ws_auth import (
    JWT_SECRET_KEY,
    InvalidTokenError,
    decode_user_access_token,
    extract_bearer_token,
)
from services.websocket_chat_actions import handle_ws_user_send_message
from services.ws_payloads import coalesce_inbound_message_text, parse_is_model_prompt

logger = get_logger("chat-service")
chat_service = ChatService(logger=logger)

router = APIRouter(prefix="/ws")


@router.websocket("/chat/{chat_room_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    chat_room_id: str,
    token: Optional[str] = Query(default=None),
):
    """WS /api/v1/ws/chat/{chat_room_id}?token=<JWT>: realtime chat for one room.

    Auth: JWT access token from the ``token`` query param (browsers cannot set
    headers on WebSockets) or an ``Authorization: Bearer`` header. The user id is the
    token's ``sub``/``user_id`` claim. The user must be an active participant of the
    room, otherwise the socket is closed with 1008.

    Inbound text frames:
        - ``"close"``: client asks to disconnect.
        - JSON ``{"action": "send_message", "content" | "message": str,
          "is_model_prompt": bool}``: broadcast and persist the message; when
          ``is_model_prompt`` is true an LLM reply follows. Other actions and
          invalid JSON are ignored.

    Outbound frames (``action``): ``chat_message``, ``llm_reply``, ``llm_error``,
    ``last_message_updated`` (payload builders live in ``services/ws_payloads.py``).

    The handshake is always accepted first and rejections are sent as a close frame
    (browsers only see the close code/reason after a completed handshake; closing
    before ``accept()`` surfaces as an opaque 1006).

    Close codes: 1008 missing/invalid/expired token, bad room id or not a room
    member; 1011 JWT secret not configured, membership check failure
    or runtime error.
    """
    client = websocket.client
    client_addr = f"{client.host}:{client.port}" if client else "unknown-client"
    logger.info("WS connect attempt from %s to room %s", client_addr, chat_room_id)

    # Accept exactly once, up front, so every rejection below can deliver a real
    # close code to the browser. ``websocket_service.connect`` only registers.
    await websocket.accept()

    raw_token = extract_bearer_token(websocket, token)
    if not raw_token:
        logger.warning("WS reject (%s): missing token", client_addr)
        await websocket.close(code=1008, reason="Missing token")
        return
    if not JWT_SECRET_KEY:
        logger.error("WS reject: JWT secret is not configured in chat-service env")
        await websocket.close(code=1011, reason="JWT secret is not configured")
        return
    try:
        _payload, user_id = decode_user_access_token(raw_token)
    except InvalidTokenError:
        logger.warning("WS reject (%s): invalid or expired token", client_addr)
        await websocket.close(code=1008, reason="Invalid or expired token")
        return

    # SECURITY: authentication alone is not enough; the user must belong to the room.
    # A short-lived background session is used because the socket outlives any
    # request-scoped session.
    try:
        async with get_background_session() as db_session:
            is_member = await ChatRepository(session=db_session).is_participant(
                chat_room_id=int(chat_room_id), user_id=int(user_id)
            )
    except ValueError:
        logger.warning("WS reject (%s): invalid room or user id", client_addr)
        await websocket.close(code=1008, reason="Invalid room id")
        return
    except Exception as e:
        logger.error("WS membership check failed for room %s: %s", chat_room_id, e)
        await websocket.close(code=1011, reason="Membership check failed")
        return
    if not is_member:
        logger.warning(
            "WS reject (%s): user %s is not a member of room %s",
            client_addr,
            user_id,
            chat_room_id,
        )
        await websocket.close(code=1008, reason="Not a member of this chat room")
        return

    try:
        await websocket_service.connect(websocket=websocket, room_id=chat_room_id)
        logger.info("WS connected to room %s from %s", chat_room_id, client_addr)

        while True:
            try:
                data = await websocket.receive_text()

                if data == "close":
                    logger.info("WS close requested in room %s", chat_room_id)
                    websocket_service.disconnect(websocket, chat_room_id)
                    break

                try:
                    # Frames are best-effort JSON; bad frames are skipped, not fatal.
                    message = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from client in room %s", chat_room_id)
                    continue

                if message.get("action") != "send_message":
                    continue

                content = coalesce_inbound_message_text(message)
                if not content:
                    continue

                # True when the user addressed the model (frontend marks ``@model`` prompts).
                is_model_prompt = parse_is_model_prompt(message.get("is_model_prompt"))

                await handle_ws_user_send_message(
                    chat_room_id=chat_room_id,
                    user_id=user_id,
                    content=content,
                    raw_token=raw_token,
                    chat_service=chat_service,
                    is_model_prompt=is_model_prompt,
                )

            except WebSocketDisconnect as e:
                # Raised by receive_text() when the peer closes; not an error.
                logger.info(
                    "WS client disconnected in room %s (code=%s)", chat_room_id, e.code
                )
                websocket_service.disconnect(websocket, chat_room_id)
                break

            except Exception as e:
                logger.error("WS runtime error in room %s: %s", chat_room_id, e)
                websocket_service.disconnect(websocket, chat_room_id)

                # close() raises RuntimeError if the socket is already closed.
                try:
                    await websocket.close(code=1011, reason="Websocket error")
                except RuntimeError:
                    pass
                break
    except Exception as e:
        logger.error("WS runtime error in room %s: %s", chat_room_id, e)
        websocket_service.disconnect(websocket, chat_room_id)
        try:
            await websocket.close(code=1011, reason="Websocket error")
        except RuntimeError:
            pass
