"""Chat REST endpoints (router layer), mounted at ``/api/v1/chat`` on port 8005.

Thin HTTP handlers that delegate to ``ChatService``. The API gateway forwards
``/chat/...`` here and injects the ``x-user-id`` header on protected routes.
Services raise ``ValueError`` for bad input; handlers map it to 400/404 here.
"""

from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from shared.logger import get_logger

from db.database import get_async_session
from models.chat import chatRoom
from schemas.chat import (
    ChatMessageHistoryResponse,
    ChatMessagePageResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatParticipantResponse,
    ChatRoom,
    ChatRoomCreateRequest,
    ChatRoomUpdateRequest,
    ChatRoomUpdateResponse,
    LeaveChatRoomResponse,
    LatestChatMessageResponse,
    RoomMembershipResponse,
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

    Auth: ``x-user-id`` header (injected by the gateway) is required, and the caller
    may view any user's room list: it backs the profile view. SECURITY: for rooms
    the caller is not an active member of, ``latest_message`` and ``description``
    are returned as null, because they are room content (security review,
    2026-11-26). Viewing your own profile returns everything.
    Path: ``profile_user_id`` int user id.
    Response: 200 list of ``UserChatRoomResponse`` (status is relative to that user).
    Errors: 422 if the id is not an int; 500 on any failure.
    """
    try:
        rooms: List[chatRoom] = await chat_service.get_all_user_chat_rooms(
            user_id=profile_user_id,
            session=session,
        )
        if int(user_id) == int(profile_user_id):
            caller_room_ids = {room.chat_room_id for room in rooms}
        else:
            # One query for the caller's own rooms instead of a check per room.
            caller_rooms = await chat_service.get_all_user_chat_rooms(
                user_id=int(user_id), session=session
            )
            caller_room_ids = {room.chat_room_id for room in caller_rooms}
        responses = []
        for room in rooms:
            item = await _build_user_chat_room_response(room, profile_user_id, session)
            if room.chat_room_id not in caller_room_ids:
                item.latest_message = None
                item.description = None
            responses.append(item)
        return responses
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list profile user rooms")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/rooms/{chat_room_id}/membership",
    status_code=status.HTTP_200_OK,
    response_model=RoomMembershipResponse,
)
async def get_room_membership(
    chat_room_id: int,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
) -> RoomMembershipResponse:
    """GET /chat/rooms/{chat_room_id}/membership: is the caller in this room?

    Auth: ``x-user-id`` header (injected by the gateway from the verified JWT), so
    a caller can only ask about itself.
    Path: ``chat_room_id`` numeric room id (422 otherwise).
    Response: 200 ``RoomMembershipResponse``; ``is_member`` is true only for an
    active participant (same rule as the WebSocket and message history).
    Errors: 500 on any failure.

    Why 200 + bool instead of 403: callers (document-service) must tell "not a
    member" apart from "chat-service failed" or an auth error from the
    gateway, which a 403 from here would blur together.
    """
    try:
        is_member = await chat_service.is_participant(chat_room_id, user_id, session)
        return RoomMembershipResponse(
            chat_room_id=chat_room_id, user_id=int(user_id), is_member=bool(is_member)
        )
    except Exception:
        logger.exception("Membership check failed room_id=%s", chat_room_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/rooms/{chat_room_id}/messages",
    status_code=status.HTTP_200_OK,
    response_model=ChatMessagePageResponse,
)
async def get_older_chat_messages(
    chat_room_id: int,
    before: int = Query(..., description="Id of the oldest message the client has"),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
) -> ChatMessagePageResponse:
    """GET /chat/rooms/{chat_room_id}/messages?before=&limit=: an older history page.

    Auth: ``x-user-id`` header (gateway); caller must be an active participant.
    Query: ``before`` (required) id of the oldest message already loaded;
    ``limit`` page size (default 50, 1..200).
    Response: 200 ``ChatMessagePageResponse`` (messages oldest first, ``has_more``).
    Errors: 400 ``before`` is not a message of this room; 403 not a participant;
    422 bad query; 500 otherwise.
    """
    try:
        await _require_participant(chat_room_id, user_id, session)
        messages, has_more = await chat_service.get_chat_messages_page(
            chat_room_id=chat_room_id,
            session=session,
            limit=limit,
            before_message_id=before,
        )
        return ChatMessagePageResponse(messages=messages, has_more=has_more)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        logger.exception("Failed to page chat messages room_id=%s", chat_room_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/rooms/{chat_room_id}/leave",
    status_code=status.HTTP_200_OK,
    response_model=LeaveChatRoomResponse,
)
async def leave_chat_room(
    chat_room_id: int,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
) -> LeaveChatRoomResponse:
    """POST /chat/rooms/{chat_room_id}/leave: the caller leaves the room.

    Auth: ``x-user-id`` header (gateway); a user can only remove themselves.
    Response: 200 ``LeaveChatRoomResponse``; ``room_archived`` is true when the
    caller was the last active member (room deactivated and hidden from every list;
    there is no delete because rooms have no owner yet).
    Errors: 403 caller is not an active participant; 500 otherwise.
    Note: sockets already open in the room are not closed here; the next message
    sent on one is refused (closed with 1008), and the client closes its own socket.
    """
    try:
        archived = await chat_service.leave_chat_room(
            chat_room_id=chat_room_id, user_id=int(user_id), session=session
        )
        return LeaveChatRoomResponse(chat_room_id=str(chat_room_id), room_archived=archived)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception:
        logger.exception("Failed to leave chat room room_id=%s", chat_room_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.patch(
    "/rooms/{chat_room_id}",
    status_code=status.HTTP_200_OK,
    response_model=ChatRoomUpdateResponse,
)
async def update_chat_room(
    chat_room_id: int,
    body: ChatRoomUpdateRequest,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
) -> ChatRoomUpdateResponse:
    """PATCH /chat/rooms/{chat_room_id}: rename a room and/or change its description.

    Auth: ``x-user-id`` header (gateway); any active participant may edit (rooms
    have no owner column yet).
    Body: ``ChatRoomUpdateRequest`` (``chat_room_name`` and/or ``description``,
    each <= 255 chars; omitted fields unchanged, an empty description clears it).
    Response: 200 ``ChatRoomUpdateResponse``.
    Errors: 400 blank name or empty body; 403 not a participant (also for a room
    that does not exist, since it has no participants); 422 too long; 500 otherwise.
    """
    try:
        await _require_participant(chat_room_id, user_id, session)
        room = await chat_service.update_chat_room(chat_room_id, body, session)
        # Field names only; the new values are user content and stay out of logs.
        logger.info(
            "Chat room updated room_id=%s fields=%s",
            chat_room_id,
            ",".join(sorted(body.model_dump(exclude_none=True))),
        )
        return ChatRoomUpdateResponse(
            chat_room_id=str(room.chat_room_id),
            chat_room_name=room.chat_room_name,
            description=room.description,
            updated_at=room.updated_at,
        )
    except HTTPException:
        raise
    except NoResultFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat room not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        logger.exception("Failed to update chat room room_id=%s", chat_room_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/messages/{chat_room_id}", response_model=ChatMessageHistoryResponse)
async def get_chat_messages(
    chat_room_id: str,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """GET /chat/messages/{chat_room_id}: newest history page, participants and preview.

    Auth: ``x-user-id`` header (injected by the gateway); the caller must be an
    active participant of the room.
    Path: ``chat_room_id`` numeric room id as a string.
    Query: ``limit`` page size (default 50, 1..200; 422 outside).
    Response: 200 ``ChatMessageHistoryResponse`` (newest ``limit`` messages oldest
    first, ``has_more``, participants, last_message, current_user_status). Older
    pages: ``GET /chat/rooms/{chat_room_id}/messages?before=``.
    Errors: 400 non-numeric room id; 403 caller is not a participant; 500 otherwise.
    """
    try:
        try:
            room_id = int(chat_room_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="chat_room_id must be numeric",
            )
        await _require_participant(room_id, user_id, session)

        # Why paged: rooms grow without bound and the UI rendered every message on open.
        messages, has_more = await chat_service.get_chat_messages_page(
            chat_room_id=room_id, session=session, limit=limit
        )
        participants: List[ChatParticipantResponse] = (
            await chat_service.get_chat_room_participants(
                chat_room_id=room_id, session=session
            )
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
        logger.info(
            "Chat history loaded room_id=%s messages=%d participants=%d",
            room_id,
            len(messages),
            len(participants),
        )
        return {
            "messages": messages,
            "has_more": has_more,
            "participants": participants,
            "last_message": last_message,
            "current_user_status": current_user_status,
        }
    except HTTPException:
        raise
    except NoResultFound as e:
        # Logged once here (WARNING, no traceback) where it becomes a 404.
        logger.warning("Chat room not found room_id=%s", room_id)
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
        logger.warning("Add user rejected room_id=%s reason=%s", room_id, e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to add user to chat room")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
