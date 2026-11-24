"""Chat REST endpoints (router layer), mounted at ``/api/v1/chat`` on port 8005.

Thin HTTP handlers that delegate to ``ChatService``. The API gateway forwards
``/chat/...`` here and injects the ``x-user-id`` header on protected routes.
Services raise ``ValueError`` for bad input; handlers map it to 400/404 here.
"""

from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from shared.logger import get_logger

from db.database import get_async_session
from models.chat import chatRoom
from schemas.chat import (
    ChatMessageHistoryResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatParticipantResponse,
    ChatRoom,
    ChatRoomCreateRequest,
    LatestChatMessageResponse,
    UserChatRoomResponse,
)
from services.chat_service import ChatService

# Get configured logger
logger = get_logger("chat-service")

# Create router
router = APIRouter(prefix="/chat", tags=["chat"])

# Create chat service instance
chat_service: ChatService = ChatService(logger=logger)


async def _require_participant(
    chat_room_id: int, user_id: int, session: AsyncSession
) -> None:
    """Raise HTTP 403 unless the user is an active participant of the room.

    SECURITY: the gateway only authenticates; room-level authorization is
    enforced here.

    Args:
        chat_room_id: Room to check.
        user_id: Caller id from ``x-user-id``.
        session: Request-scoped DB session.

    Raises:
        HTTPException: 403 if the caller is not an active participant.
    """
    if not await chat_service.is_participant(chat_room_id, user_id, session):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a participant of this chat room",
        )


async def _build_user_chat_room_response(
    room: chatRoom,
    user_id: int,
    session: AsyncSession,
) -> UserChatRoomResponse:
    """Map a ``chatRoom`` row plus the user's membership status to the list-item schema.

    ``latest_message`` is built from the room's denormalized last-message columns
    and is None while ``last_message_id`` is unset. The UUID columns are converted
    with ``str()`` to match ``LatestChatMessageResponse``.
    """
    current_user_status = await chat_service.get_current_user_chat_room_status(
        chat_room_id=int(room.chat_room_id),
        user_id=int(user_id),
        session=session,
    )
    latest_message = None
    if room.last_message_id:
        latest_message = LatestChatMessageResponse(
            chat_message_id=str(room.last_message_id),
            chat_room_id=str(room.chat_room_id),
            user_id=(
                str(room.last_message_sender_id)
                if room.last_message_sender_id
                else None
            ),
            message=room.last_message_text or "",
            created_at=room.last_message_at or room.created_at,
        )
    return UserChatRoomResponse(
        chat_room_id=room.chat_room_id,
        chat_room_name=room.chat_room_name,
        description=room.description,
        room_created_at=room.created_at,
        room_updated_at=room.updated_at,
        room_is_active=room.is_active,
        latest_message=latest_message,
        current_user_status=current_user_status,
    )


@router.post(
    "/messages",
    status_code=status.HTTP_200_OK,
    response_model=ChatMessageResponse,
)
async def send_message(
    message_request: ChatMessageRequest,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
) -> ChatMessageResponse:
    """POST /chat/messages: save a message over plain HTTP (no WebSocket broadcast).

    Auth: ``x-user-id`` header (injected by the gateway); the caller must be an
    active participant of the room named by ``conversation_id``.
    Body: ``ChatMessageRequest``; ``conversation_id`` is the chat room id and the
    body ``user_id`` must equal the header user.
    Response: 200 the saved ``ChatMessageResponse``.
    Errors: 400 missing/invalid ``conversation_id``; 403 not a participant or
    body ``user_id`` differs from the caller; 422 invalid body; 500 otherwise.
    """
    try:
        if message_request.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="user_id does not match the authenticated user",
            )
        try:
            chat_room_id = int(message_request.conversation_id or "")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="conversation_id must be a numeric chat room id",
            )
        await _require_participant(chat_room_id, user_id, session)
        return await chat_service.process_message(
            message_request, user_id=user_id, session=session
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Traceback goes to the log only; the client gets a generic message.
        logger.exception("Failed to send message")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/rooms", status_code=status.HTTP_200_OK)
async def create_chat_room(
    chat_room_request: ChatRoomCreateRequest,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
):
    """POST /chat/rooms: create a room and add the caller as its first participant.

    Auth: ``x-user-id`` header (injected by the gateway).
    Body: ``ChatRoomCreateRequest`` (``chat_room_name``, 1-100 chars).
    Response: 200 ``{message, chat_room_id, chat_room_name, participant_id}``.
    Errors: 400 empty name; 422 on invalid body; 500 on any other failure.
    """
    try:
        if not chat_room_request.chat_room_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chat room name is required",
            )

        # Create the room and register the caller as a participant.
        room, participant = await chat_service.create_chat_room(
            ChatRoom(chat_room_name=chat_room_request.chat_room_name),
            user_id=user_id,
            session=session,
        )

        return {
            "message": "Chat room created successfully",
            "chat_room_id": room.chat_room_id,
            "chat_room_name": room.chat_room_name,
            "participant_id": getattr(participant, "chat_participant_id", None),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create chat room")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/user-chat-rooms",
    status_code=status.HTTP_200_OK,
    response_model=List[UserChatRoomResponse],
)
async def get_user_chat_rooms(
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
) -> List[UserChatRoomResponse]:
    """GET /chat/user-chat-rooms: list the caller's active rooms.

    Auth: ``x-user-id`` header (injected by the gateway).
    Response: 200 list of ``UserChatRoomResponse`` (room info, latest message,
    caller's membership status).
    Errors: 500 on any failure.
    """
    try:
        logger.info("Getting user chat rooms for user_id: %s", user_id)
        rooms: List[chatRoom] = await chat_service.get_all_user_chat_rooms(
            user_id=user_id, session=session
        )
        return [
            await _build_user_chat_room_response(room, int(user_id), session)
            for room in rooms
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list rooms")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/users/{profile_user_id}/chat-rooms",
    status_code=status.HTTP_200_OK,
    response_model=List[UserChatRoomResponse],
)
async def get_profile_user_chat_rooms(
    profile_user_id: int,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
) -> List[UserChatRoomResponse]:
    """GET /chat/users/{profile_user_id}/chat-rooms: list another user's active rooms.

    Auth: ``x-user-id`` header (injected by the gateway) is required, but the caller
    may view any user's rooms. This is intentional: the endpoint backs the profile
    view, which lists the rooms a user belongs to. NOTE: restricting it to rooms
    shared with the caller would be a product decision and is not applied here.
    Path: ``profile_user_id`` int user id.
    Response: 200 list of ``UserChatRoomResponse`` (status is relative to that user).
    Errors: 422 if the id is not an int; 500 on any failure.
    """
    try:
        rooms: List[chatRoom] = await chat_service.get_all_user_chat_rooms(
            user_id=profile_user_id,
            session=session,
        )
        return [
            await _build_user_chat_room_response(room, profile_user_id, session)
            for room in rooms
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list profile user rooms")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/messages/{chat_room_id}", response_model=ChatMessageHistoryResponse)
async def get_chat_messages(
    chat_room_id: str,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
):
    """GET /chat/messages/{chat_room_id}: room history, participants and preview.

    Auth: ``x-user-id`` header (injected by the gateway); the caller must be an
    active participant of the room.
    Path: ``chat_room_id`` numeric room id as a string.
    Response: 200 ``ChatMessageHistoryResponse`` (messages oldest first,
    participants, last_message, current_user_status).
    Errors: 400 non-numeric room id; 403 caller is not a participant; 500 otherwise.
    """
    logger.info(f"[history] start room={chat_room_id}")
    try:
        try:
            room_id = int(chat_room_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="chat_room_id must be numeric",
            )
        await _require_participant(room_id, user_id, session)

        messages: List[ChatMessageResponse] = (
            await chat_service.get_chat_messages_by_chat_room_id(
                chat_room_id=room_id, session=session
            )
        )
        logger.info(f"[history] messages_count room={room_id} count={len(messages)}")
        participants: List[ChatParticipantResponse] = (
            await chat_service.get_chat_room_participants(
                chat_room_id=room_id, session=session
            )
        )
        logger.info(
            f"[history] participants_count room={room_id} count={len(participants)}"
        )
        current_user_status = await chat_service.get_current_user_chat_room_status(
            chat_room_id=room_id,
            user_id=int(user_id),
            session=session,
        )
        last_message = await chat_service.get_chat_room_last_message_preview(
            chat_room_id=room_id,
            session=session,
        )
        # The preview is message content, so only log whether one exists.
        logger.info(
            f"[history] last_message room={room_id} present={last_message is not None}"
        )
        return {
            "messages": messages,
            "participants": participants,
            "last_message": last_message,
            "current_user_status": current_user_status,
        }
    except HTTPException:
        raise
    except NoResultFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to get chat messages")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/add-user-to-chat-room/{chat_room_id}", status_code=status.HTTP_200_OK)
async def add_user_to_chat_room(
    chat_room_id: str,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    caller_id: int = Header(..., alias="x-user-id"),
):
    """POST /chat/add-user-to-chat-room/{chat_room_id}: add a user to a room.

    Auth: ``x-user-id`` header (injected by the gateway); the caller must be an
    active participant of the room (only members can invite).
    Path: ``chat_room_id`` numeric room id as a string.
    Body: JSON ``{"user_id": <int>}`` (the user to add).
    Response: 200 ``{"message": "User added to chat room successfully"}``.
    Errors: 400 missing/invalid ``user_id``, non-numeric room id, or user already
    a member; 403 caller is not a participant; 500 on any other failure.
    """
    try:
        try:
            room_id = int(chat_room_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="chat_room_id must be numeric",
            )
        await _require_participant(room_id, caller_id, session)

        payload = await request.json()
        user_id = payload.get("user_id") if isinstance(payload, dict) else None
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="User ID is required"
            )
        await chat_service.add_user_to_chat_room(
            chat_room_id=room_id, user_id=int(user_id), session=session
        )
        return {"message": "User added to chat room successfully"}
    except HTTPException:
        raise
    except ValueError as e:
        # Covers "User already in chat room" and a non-integer body user_id.
        logger.error(f"Invalid add-user request: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to add user to chat room")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
