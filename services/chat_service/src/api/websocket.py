"""WebSocket router (router layer): the chat socket endpoint only.

Business logic lives under ``services/``. Mounted at ``/api/v1/ws``; browsers connect
directly to this service (WebSockets do not go through the API gateway).
"""

import json
import time
from typing import Optional

from fastapi import APIRouter, Query, WebSocket
from starlette.websockets import WebSocketDisconnect

from shared.log_context import bind_context, bind_user, reset_context
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

# Close code for "token missing, invalid or expired" (4000-4999 is the application
# range). Kept apart from 1008 so the browser can tell "refresh your token and
# reconnect" from "you are not allowed in this room, stop retrying". Mirrored as
# WS_CLOSE_UNAUTHORIZED in uxaag_UI/app/api/chat/useChatWebSocket.ts.
WS_CLOSE_UNAUTHORIZED = 4401
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

    Close codes: 4401 (``WS_CLOSE_UNAUTHORIZED``) missing/invalid/expired token, so
    the client refreshes and reconnects; 1008 bad room id or not a room member (final);
    1011 JWT secret not configured, membership check failure or runtime error.
    """
    # One request id per connection (all lines of this socket share it); the user id
    # is added as soon as the token is decoded.
    conn_tokens = bind_context(None, None)
    started = time.monotonic()
    user_id: Optional[str] = None
    close_code: Optional[int] = None  # set once the socket was accepted as a chat socket

    try:
        # Accept exactly once, up front, so every rejection below can deliver a real
        # close code to the browser. ``websocket_service.connect`` only registers.
        await websocket.accept()

        raw_token = extract_bearer_token(websocket, token)
        if not raw_token:
            logger.warning(
                "WebSocket rejected room_id=%s reason=missing_token", chat_room_id
            )
            await websocket.close(code=WS_CLOSE_UNAUTHORIZED, reason="Missing token")
            return
        if not JWT_SECRET_KEY:
            logger.error(
                "WebSocket rejected room_id=%s reason=jwt_secret_not_configured",
                chat_room_id,
            )
            await websocket.close(code=1011, reason="JWT secret is not configured")
            return
        try:
            _payload, user_id = decode_user_access_token(raw_token)
        except InvalidTokenError:
            logger.warning(
                "WebSocket rejected room_id=%s reason=invalid_token", chat_room_id
            )
            await websocket.close(
                code=WS_CLOSE_UNAUTHORIZED, reason="Invalid or expired token"
            )
            return
        bind_user(user_id)

        # SECURITY: authentication alone is not enough; the user must belong to the
        # room. A short-lived background session is used because the socket outlives
        # any request-scoped session.
        try:
            async with get_background_session() as db_session:
                is_member = await ChatRepository(session=db_session).is_participant(
                    chat_room_id=int(chat_room_id), user_id=int(user_id)
                )
        except ValueError:
            logger.warning(
                "WebSocket rejected room_id=%s user_id=%s reason=invalid_id",
                chat_room_id,
                user_id,
            )
            await websocket.close(code=1008, reason="Invalid room id")
            return
        except Exception:
            logger.exception(
                "WebSocket membership check failed room_id=%s user_id=%s",
                chat_room_id,
                user_id,
            )
            await websocket.close(code=1011, reason="Membership check failed")
            return
        if not is_member:
            logger.warning(
                "WebSocket rejected room_id=%s user_id=%s reason=not_a_member",
                chat_room_id,
                user_id,
            )
            await websocket.close(code=1008, reason="Not a member of this chat room")
            return

        await websocket_service.connect(websocket=websocket, room_id=chat_room_id)
        close_code = 1000  # default if the loop ends without a more specific code
        logger.info("WebSocket connected room_id=%s user_id=%s", chat_room_id, user_id)

        while True:
            try:
                data = await websocket.receive_text()

                if data == "close":
                    close_code = 1000
                    websocket_service.disconnect(websocket, chat_room_id)
                    break

                try:
                    # Frames are best-effort JSON; bad frames are skipped, not fatal.
                    message = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(
                        "WebSocket frame ignored room_id=%s reason=invalid_json length=%d",
                        chat_room_id,
                        len(data),
                    )
                    continue

                if message.get("action") != "send_message":
                    continue

                content = coalesce_inbound_message_text(message)
                if not content:
                    continue

                # True when the user addressed the model (frontend marks ``@model`` prompts).
                is_model_prompt = parse_is_model_prompt(message.get("is_model_prompt"))

                # Fresh request id per inbound message: one @model flow (chat -> gateway
                # -> rag) can be traced end to end. Background tasks spawned while it is
                # bound copy it (contextvars), so their lines carry the same id.
                msg_tokens = bind_context(None, user_id)
                try:
                    # Length only: message text is never logged.
                    logger.info(
                        "WebSocket message received room_id=%s length=%d is_model_prompt=%s",
                        chat_room_id,
                        len(content),
                        is_model_prompt,
                    )
                    await handle_ws_user_send_message(
                        chat_room_id=chat_room_id,
                        user_id=user_id,
                        content=content,
                        raw_token=raw_token,
                        chat_service=chat_service,
                        is_model_prompt=is_model_prompt,
                    )
                finally:
                    # Back to the connection's request id for the next frame.
                    reset_context(msg_tokens)

            except WebSocketDisconnect as e:
                # Raised by receive_text() when the peer closes; not an error.
                close_code = e.code
                websocket_service.disconnect(websocket, chat_room_id)
                break

            except Exception:
                logger.exception(
                    "WebSocket runtime error room_id=%s user_id=%s",
                    chat_room_id,
                    user_id,
                )
                close_code = 1011
                websocket_service.disconnect(websocket, chat_room_id)

                # close() raises RuntimeError if the socket is already closed.
                try:
                    await websocket.close(code=1011, reason="Websocket error")
                except RuntimeError:
                    pass
                break
    except Exception:
        logger.exception(
            "WebSocket runtime error room_id=%s user_id=%s", chat_room_id, user_id
        )
        if close_code is not None:
            close_code = 1011
        websocket_service.disconnect(websocket, chat_room_id)
        try:
            await websocket.close(code=1011, reason="Websocket error")
        except RuntimeError:
            pass
    finally:
        # Only sockets that passed auth were logged as connected, so only they get
        # a matching "disconnected" line.
        if close_code is not None:
            logger.info(
                "WebSocket disconnected room_id=%s user_id=%s code=%s duration_s=%.0f",
                chat_room_id,
                user_id,
                close_code,
                time.monotonic() - started,
            )
        reset_context(conn_tokens)
