"""Chat business logic (service layer): rooms, participants, messages, presence.

Called by the REST router (request-scoped sessions) and by WebSocket handlers
(background sessions). Talks to the DB only through ``ChatRepository``.
"""

from datetime import datetime, timedelta
import os
from typing import Any, List
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from models.chat import chatMessage, chatRoom, chatParticipant
from repositories.chat_repository import ChatRepository

from schemas.chat import (
    ChatMessage,
    ChatMessageRequest,
    ChatParticipantResponse,
    ChatRoom,
    ChatMessageResponse,
    LastMessagePreviewResponse,
    CurrentUserChatRoomStatusResponse,
)
from services.websocket import WebSocketService
from db.database import get_background_session

# A participant counts as "active" only if they logged in within this many minutes.
ACTIVE_WINDOW_MINUTES = int(os.getenv("USER_ACTIVE_WINDOW_MINUTES", "30"))


def _is_recent_login(last_login: datetime | None) -> bool:
    """Return True if ``last_login`` (naive UTC, from the auth ``user`` table) is within
    the active window. A missing login counts as not recent."""
    if last_login is None:
        return False
    return last_login >= datetime.utcnow() - timedelta(minutes=ACTIVE_WINDOW_MINUTES)

class ChatService:
    """Chat use-cases. Errors are logged and re-raised for the router to map to HTTP."""

    def __init__(self, logger: Any):
        """Initialize the chat service.

        Args:
            logger: Logger used for all service messages.
        """
        self.logger = logger

    async def create_chat_room(self, chat_room: ChatRoom, user_id: int, session: AsyncSession):
        """Create a room and add the creating user as its first participant.

        Args:
            chat_room: DTO carrying the room name.
            user_id: Creator id (gateway ``x-user-id``).
            session: Request-scoped DB session.

        Returns:
            Tuple ``(chatRoom, chatParticipant)`` as persisted.

        Raises:
            Exception: Any DB error is logged and re-raised. Room and participant
                are written in one transaction, so a failure leaves nothing behind.
        """
        # Initialize the chat repository
        chat_repository: ChatRepository = ChatRepository(session=session)
        try:
            # Convert the ChatRoom to a chatRoom
            chat_room_model = chatRoom(chat_room_name=chat_room.chat_room_name)
            
            # created_at/updated_at/joined_at come from the model defaults.
            # The room id is generated client-side, so it is known before the insert.
            chat_participant_model = chatParticipant(
                chat_room_id=chat_room_model.chat_room_id, user_id=user_id
            )

            # One commit for both rows: no empty room if the participant insert fails.
            return await chat_repository.create_chat_room_with_participant(
                chat_room_model, chat_participant_model
            )
        except Exception as e:
            self.logger.debug("Failed to create chat room")
            raise
    
    async def get_chat_room_by_id(self, chat_room_id: int, session: AsyncSession):
        """Get a room by id as a ``ChatRoom`` DTO.

        Raises:
            sqlalchemy.exc.NoResultFound: If the room does not exist.
        """
        try:
            chat_repository: ChatRepository = ChatRepository(session=session)
            chat_room = await chat_repository.get_chat_room_by_id(chat_room_id)
            return ChatRoom(chat_room_id=chat_room.chat_room_id, chat_room_name=chat_room.chat_room_name, created_at=chat_room.created_at)
        except Exception as e:
            self.logger.debug("Failed to get chat room by ID")
            raise
    
    async def get_all_user_chat_rooms(self, user_id: int, session: AsyncSession)-> List[chatRoom]:
        """Get the active rooms the user is an active participant of."""
        try:
            chat_repository: ChatRepository = ChatRepository(session=session)
            rooms: List[chatRoom] = await chat_repository.get_all_user_chat_rooms(user_id=user_id)
            return rooms
        except Exception as e:
            self.logger.debug("Failed to get all user chat rooms")
            raise
    
    async def get_chat_messages_by_chat_room_id(self, chat_room_id: int, session: AsyncSession) -> List[ChatMessageResponse]:
        """Get a room's messages mapped to ``ChatMessageResponse`` (ids as strings)."""
        try:
            chat_repository: ChatRepository = ChatRepository(session=session)
            messages: List[chatMessage] = await chat_repository.get_chat_messages_by_chat_room_id(chat_room_id=chat_room_id)
            return [
                ChatMessageResponse(
                    id=str(message.chat_message_id),
                    chat_room_id=str(message.chat_room_id),
                    user_id=message.user_id,
                    role=message.role or "user",
                    message=message.message,
                    created_at=message.created_at,
                    updated_at=message.updated_at,
                )
                for message in messages
            ]
        except Exception as e:
            self.logger.debug("Failed to get chat messages by chat room ID")
            raise
    
    async def get_chat_room_participants(self, chat_room_id: int, session: AsyncSession) -> List[ChatParticipantResponse]:  
        """Get a room's participants enriched with auth-user profile data.

        ``is_active``/``status`` combine the membership flag with a recent-login check:
        ``active`` if both hold, else ``left`` (``left_at`` set) or ``inactive``.
        """
        try:
            chat_repository: ChatRepository = ChatRepository(session=session)
            participants: List[chatParticipant] = await chat_repository.get_chat_room_participants(chat_room_id=chat_room_id)
            profiles = await chat_repository.get_user_presence_by_ids(
                [int(participant.user_id) for participant in participants]
            )
            return [
                ChatParticipantResponse(
                    chat_participant_id=int(participant.chat_participant_id),
                    chat_room_id=int(participant.chat_room_id),
                    user_id=int(participant.user_id),
                    username=profiles.get(int(participant.user_id), {}).get("username"),
                    email=profiles.get(int(participant.user_id), {}).get("email"),
                    full_name=profiles.get(int(participant.user_id), {}).get("full_name"),
                    last_login=profiles.get(int(participant.user_id), {}).get("last_login"),
                    joined_at=participant.joined_at,
                    left_at=participant.left_at,
                    is_active=bool(participant.is_active)
                    and _is_recent_login(profiles.get(int(participant.user_id), {}).get("last_login")),
                    status=(
                        "active"
                        if bool(participant.is_active)
                        and _is_recent_login(profiles.get(int(participant.user_id), {}).get("last_login"))
                        else ("left" if participant.left_at else "inactive")
                    ),
                )
                for participant in participants
            ]
        except Exception as e:
            self.logger.debug("Failed to get chat room participants")
            raise

    async def get_current_user_chat_room_status(
        self,
        chat_room_id: int,
        user_id: int,
        session: AsyncSession,
    ) -> CurrentUserChatRoomStatusResponse:
        """Get the user's membership status in a room.

        Returns:
            Status with ``is_member=False`` when the user has no participant row;
            otherwise ``is_active`` follows the same rule as the participants list.
        """
        chat_repository: ChatRepository = ChatRepository(session=session)
        participant = await chat_repository.get_chat_room_participant_by_user_id(
            chat_room_id=chat_room_id,
            user_id=user_id,
        )
        if participant is None:
            return CurrentUserChatRoomStatusResponse(
                user_id=user_id,
                chat_room_id=chat_room_id,
                is_member=False,
                is_active=False,
                joined_at=None,
                left_at=None,
            )
        profiles = await chat_repository.get_user_presence_by_ids([int(user_id)])
        profile = profiles.get(int(user_id), {})
        is_active = bool(participant.is_active) and _is_recent_login(profile.get("last_login"))
        return CurrentUserChatRoomStatusResponse(
            user_id=user_id,
            chat_room_id=chat_room_id,
            is_member=True,
            is_active=is_active,
            joined_at=participant.joined_at,
            left_at=participant.left_at,
        )

    async def get_chat_room_last_message_preview(
        self,
        chat_room_id: int,
        session: AsyncSession,
    ) -> LastMessagePreviewResponse | None:
        """Get the last-message preview from the ``chatMessage`` table (source of truth).

        Returns:
            The preview, or None when the room has no messages or the lookup fails
            (errors are logged and swallowed so the history endpoint still works).
        """
        try:
            chat_repository: ChatRepository = ChatRepository(session=session)
            latest = await chat_repository.get_latest_chat_message_by_chat_room_id(
                chat_room_id=chat_room_id
            )
            if not latest:
                return None
            preview = LastMessagePreviewResponse(
                message=latest.message,
                created_at=latest.created_at,
            )
            return preview
        except Exception as e:
            self.logger.exception("Failed to get chat room last-message preview")
            return None
    
    async def create_chat_message(self, chat_message: ChatMessage, session: AsyncSession):
        """Persist a ``ChatMessage`` DTO using the caller's session.

        Returns:
            The created ``chatMessage`` row.
        """
        try:
            chat_repository: ChatRepository = ChatRepository(session=session)
            # ChatMessage schema uses `id`; fall back to generated integer ID when missing.
            message_id = int(chat_message.id) if chat_message.id is not None else int(uuid.uuid4().int) % (2**31)
            # Only pass timestamps the caller supplied: passing None would override
            # the model's ``default_factory`` and break the NOT NULL columns.
            timestamps = {
                key: value
                for key, value in (
                    ("created_at", chat_message.created_at),
                    ("updated_at", chat_message.updated_at),
                )
                if value is not None
            }
            chat_message_model = chatMessage(
                chat_message_id=message_id,
                chat_room_id=int(chat_message.chat_room_id),
                user_id=int(chat_message.user_id) if chat_message.user_id is not None else None,
                role=(chat_message.role or "user"),
                message=chat_message.message,
                **timestamps,
            )
            chat_message_created = await chat_repository.create_chat_message(chat_message_model)
            return chat_message_created
        except Exception as e:
            self.logger.debug("Failed to create chat message")
            raise
    
    async def is_participant(
        self, chat_room_id: int, user_id: int, session: AsyncSession
    ) -> bool:
        """Return True if the user is an active participant of the room.

        Used by routers to enforce room membership (HTTP 403 otherwise).
        """
        chat_repository: ChatRepository = ChatRepository(session=session)
        return await chat_repository.is_participant(
            chat_room_id=chat_room_id, user_id=user_id
        )

    async def process_message(
        self,
        message_request: ChatMessageRequest,
        user_id: int,
        session: AsyncSession,
    ) -> ChatMessageResponse:
        """Persist a message sent over plain HTTP and refresh the room preview.

        The caller's membership is checked by the router; this method only
        validates the request and writes. No WebSocket broadcast happens here.

        Args:
            message_request: Body with ``message`` and ``conversation_id`` (the
                chat room id as a string).
            user_id: Author id (gateway ``x-user-id``), never taken from the body.
            session: Request-scoped DB session.

        Returns:
            The saved message mapped to ``ChatMessageResponse``.

        Raises:
            ValueError: If ``conversation_id`` is missing or not a valid room id.
        """
        try:
            chat_room_id = int(message_request.conversation_id or "")
        except ValueError:
            raise ValueError("conversation_id must be a numeric chat room id")
        created = await self.create_chat_message(
            ChatMessage(
                user_id=user_id,
                role="user",
                message=message_request.message,
                chat_room_id=str(chat_room_id),
            ),
            session=session,
        )
        # Keep the room-list preview in sync (opens its own background session).
        await self.update_chat_room_last_message(
            chat_room_id=chat_room_id,
            message=message_request.message,
            user_id=user_id,
        )
        return ChatMessageResponse(
            id=str(created.chat_message_id),
            chat_room_id=str(created.chat_room_id),
            user_id=created.user_id,
            role=created.role or "user",
            message=created.message,
            created_at=created.created_at,
            updated_at=created.updated_at,
        )

    async def save_message_background(
        self,
        chat_room_id: str,
        content: str,
        user_id: int | None = None,
        role: str = "user",
    ) -> chatMessage:
        """Save a user or assistant message using its own background-safe session.

        Why: this runs inside ``asyncio.create_task`` from the WebSocket handler, where
        the request-scoped session may already be closed.

        Args:
            chat_room_id: Room id (string from the URL path).
            content: Message text.
            user_id: Author id, or None for assistant (LLM) messages.
            role: ``"user"`` or ``"assistant"``.

        Returns:
            The created ``chatMessage`` row.

        Raises:
            Exception: DB errors are logged and re-raised (nobody awaits the task,
                so a failure only shows up in the logs).
        """
        try:
            async with get_background_session() as db_session:
                chat_repository = ChatRepository(session=db_session)
                message_id = int(uuid.uuid4().int % (2**31))
                row = chatMessage(
                    chat_message_id=message_id,
                    chat_room_id=int(chat_room_id),
                    user_id=int(user_id) if user_id is not None else None,
                    role=(role or "user"),
                    message=content,
                    # created_at/updated_at: model defaults (naive local time).
                )
                created = await chat_repository.create_chat_message(row)
            # Ids and length only: message text is never logged.
            self.logger.info(
                "Message saved room_id=%s role=%s length=%d",
                chat_room_id,
                role,
                len(content or ""),
            )
            return created
        except Exception as e:
            self.logger.debug("Failed to save message room_id=%s", chat_room_id)
            raise
    
    async def add_user_to_chat_room(self, chat_room_id: int, user_id: int, session: AsyncSession):
        """Add a user to a room.

        Raises:
            ValueError: If the user already has a participant row in the room.
        """
        # Note: the ValueError below propagates unchanged; the router logs the one
        # WARNING (no traceback) and maps it to HTTP 400.
        try:
            chat_repository: ChatRepository = ChatRepository(session=session)
            # Check if the user is already in the chat room
            chat_participants = await chat_repository.get_chat_room_participants(chat_room_id=chat_room_id)
            if chat_participants:
                for chat_participant in chat_participants:
                    if chat_participant.user_id == user_id:
                        raise ValueError("User already in chat room")
            chat_participant_model = chatParticipant(chat_room_id=chat_room_id, user_id=user_id)
            chat_participant_created = await chat_repository.create_chat_participant(chat_participant_model)
            return chat_participant_created
        except ValueError:
            # Expected business rejection (already a member): not an error here.
            raise
        except Exception as e:
            self.logger.debug("Failed to add user to chat room")
            raise
    
    async def update_chat_room_last_message(self, chat_room_id: int, message: str, user_id: int) -> chatRoom:
        """
        Update the last-message fields for a room.

        Args:
            chat_room_id: Room to update.
            message: Preview text.
            user_id: Sender id (currently unused; the sender columns are not written).

        Returns:
            The updated room row.

        Important: this method opens its own DB session because it is typically
        called from background tasks (e.g. WebSocket handler). Passing a request-
        scoped session into a background task is error-prone and can result in
        immutable Row objects or closed-session errors.
        """
        try:
            async with get_background_session() as session:
                chat_repository = ChatRepository(session=session)
                return await chat_repository.update_chat_room_last_message(
                    chat_room_id=chat_room_id,
                    message=message,
                )
        except Exception as e:
            self.logger.debug("Failed to update chat room last message")
            raise
    
    
