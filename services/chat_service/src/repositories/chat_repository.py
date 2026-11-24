"""Chat repository (repository layer): async SQLModel/SQLAlchemy queries for rooms,
messages and participants.

Commits happen inside each write method; errors are logged and re-raised so the
service/router layers decide how to map them to HTTP errors.
"""

from typing import List, Optional, Tuple
from datetime import datetime
import uuid
from shared.logger import get_logger
from sqlalchemy import and_, bindparam, desc, func, or_, select, text, update
from sqlmodel.ext.asyncio.session import AsyncSession

from models.chat import chatMessage, chatRoom, chatParticipant

logger = get_logger("chat-service")

class AsyncBaseRepository:
    """Async repository base holding the ``AsyncSession`` passed in by the caller."""
    def __init__(self, session: AsyncSession):
        self.session = session

class ChatRepository(AsyncBaseRepository):
    """Data access for ``chatRoom``, ``chatMessage`` and ``chatParticipant``."""
    
    async def get_chat_room_by_id(self, chat_room_id: int) -> chatRoom:
        """Get a chat room by ID.

        Raises:
            sqlalchemy.exc.NoResultFound: If no room has this id.
        """
        try:
            result = await self.session.exec(select(chatRoom).where(chatRoom.chat_room_id == chat_room_id))
            return result.one()
        except Exception as e:
            logger.debug("Failed to get chat room by ID")
            raise
    
    async def get_all_user_chat_rooms(self, user_id: int) -> List[chatRoom]:
        """Get all active rooms in which the user is an active participant."""
        try:
            statement = (
                select(chatRoom)
                .select_from(chatParticipant)
                .join(
                    chatRoom,
                    chatParticipant.chat_room_id == chatRoom.chat_room_id,
                )
                .where(
                    chatParticipant.user_id == user_id,
                    chatParticipant.is_active == True,
                    chatRoom.is_active == True,
                )
            )
            # `execute(...).scalars()` guarantees we get chatRoom objects
            # instead of row wrappers from the joined query.
            result = await self.session.execute(statement)
            return result.scalars().all()
        except Exception as e:
            logger.debug("Failed to get all user chat rooms")
            raise
    
    async def create_chat_room(self, chat_room: chatRoom) -> chatRoom:
        """Insert a room, commit, and return it refreshed from the DB."""
        try:
            self.session.add(chat_room)
            await self.session.commit()
            await self.session.refresh(chat_room)
            return chat_room
        except Exception as e:
            logger.debug("Failed to create chat room")
            raise
    
    async def create_chat_room_with_participant(
        self, chat_room: chatRoom, chat_participant: chatParticipant
    ) -> tuple[chatRoom, chatParticipant]:
        """Insert a room and its first participant in a single commit.

        Why: two separate commits could leave an empty room if the second insert
        fails. The participant's ``chat_room_id`` is set here so the caller does
        not need the room to be flushed first.

        Args:
            chat_room: New room (id is generated client-side by the model).
            chat_participant: New participant; its ``chat_room_id`` is overwritten.

        Returns:
            The persisted ``(room, participant)`` pair, both refreshed.
        """
        try:
            chat_participant.chat_room_id = chat_room.chat_room_id
            self.session.add(chat_room)
            self.session.add(chat_participant)
            await self.session.commit()
            await self.session.refresh(chat_room)
            await self.session.refresh(chat_participant)
            return chat_room, chat_participant
        except Exception as e:
            # Release the failed transaction so the session stays usable.
            await self.session.rollback()
            logger.debug("Failed to create chat room with participant")
            raise

    async def create_chat_participant(self, chat_participant: chatParticipant) -> chatParticipant:
        """Insert a participant row, commit, and return it refreshed."""
        try:
            self.session.add(chat_participant)
            await self.session.commit()
            await self.session.refresh(chat_participant)
            return chat_participant
        except Exception as e:
            logger.debug("Failed to create chat participant")
            raise
    
    async def remove_user_from_chat_room(self, chat_participant: chatParticipant) -> chatParticipant:
        """Persist an already-modified participant (caller sets ``is_active``/``left_at``)."""
        try:
            self.session.add(chat_participant)
            await self.session.commit()
            return chat_participant
        except Exception as e:
            logger.debug("Failed to remove user from chat room")
            raise
    
    async def delete_chat_room(self, chat_room_id: int) -> None:
        """Mark the room's participants as left, then delete the room row.

        Note:
            ``AsyncSession.delete`` is a coroutine and is awaited below, while
            ``add`` is synchronous.

        Raises:
            sqlalchemy.exc.NoResultFound: If the room does not exist.
        """
        try:
            result = await self.session.exec(select(chatRoom).where(chatRoom.chat_room_id == chat_room_id))
            chat_room = result.one()
            result = await self.session.exec(select(chatParticipant).where(chatParticipant.chat_room_id == chat_room_id))
            chat_participant = result.all()
            for participant in chat_participant:
                participant.is_active = False
                participant.left_at = datetime.now()
                # AsyncSession.add is synchronous; awaiting it raises TypeError.
                self.session.add(participant)
            await self.session.delete(chat_room)
            await self.session.commit()
        except Exception as e:
            logger.debug("Failed to delete chat room")
            raise
    
    async def get_chat_messages_by_chat_room_id(self, chat_room_id: int) -> List[chatMessage]:
        """Get all messages of a room, oldest first (no paging).

        Ordered by ``created_at`` then ``chat_message_id`` so messages created in
        the same instant keep a stable order.
        """
        try:
            statement = (
                select(chatMessage)
                .where(chatMessage.chat_room_id == chat_room_id)
                .order_by(chatMessage.created_at.asc(), chatMessage.chat_message_id.asc())
            )
            result = await self.session.execute(statement)
            return result.scalars().all()
        except Exception as e:
            logger.debug("Failed to get chat messages by chat room ID")
            raise

    async def get_chat_message_in_room(
        self, chat_room_id: int, chat_message_id: int
    ) -> Optional[chatMessage]:
        """Return one message of a room, or None (also when it belongs to another room)."""
        try:
            statement = select(chatMessage).where(
                chatMessage.chat_room_id == chat_room_id,
                chatMessage.chat_message_id == chat_message_id,
            )
            result = await self.session.execute(statement)
            return result.scalars().first()
        except Exception:
            logger.debug("Failed to get chat message by id")
            raise

    async def get_chat_messages_page(
        self,
        chat_room_id: int,
        limit: int,
        before: Optional[chatMessage] = None,
    ) -> Tuple[List[chatMessage], bool]:
        """Return up to ``limit`` messages older than ``before`` (newest page if None).

        Ordering is ``(created_at, chat_message_id)``, oldest first in the result.
        Why a keyset cursor (the ``before`` message) and not OFFSET: new messages arrive
        while a user scrolls back, which would shift OFFSET pages and duplicate or skip
        rows. Message ids are random, so the cursor compares the
        ``(created_at, id)`` pair, not the id alone.

        Args:
            chat_room_id: Room to read.
            limit: Page size (caller validates the range).
            before: The oldest message the client already has, or None.

        Returns:
            ``(messages oldest first, has_more)``; ``has_more`` is True when older
            messages exist beyond this page.
        """
        try:
            statement = select(chatMessage).where(chatMessage.chat_room_id == chat_room_id)
            if before is not None:
                statement = statement.where(
                    or_(
                        chatMessage.created_at < before.created_at,
                        and_(
                            chatMessage.created_at == before.created_at,
                            chatMessage.chat_message_id < before.chat_message_id,
                        ),
                    )
                )
            # Newest first + one extra row to learn whether an older page exists.
            statement = statement.order_by(
                chatMessage.created_at.desc(), chatMessage.chat_message_id.desc()
            ).limit(limit + 1)
            result = await self.session.execute(statement)
            rows = list(result.scalars().all())
            has_more = len(rows) > limit
            page = rows[:limit]
            page.reverse()
            return page, has_more
        except Exception:
            logger.debug("Failed to get chat messages page")
            raise

    async def get_latest_chat_message_by_chat_room_id(self, chat_room_id: int) -> chatMessage | None:
        """Get the newest message of a room, or None if the room has no messages.

        Ordered by ``created_at`` then ``chat_message_id`` to break ties.
        """
        try:
            statement = (
                select(chatMessage)
                .where(chatMessage.chat_room_id == chat_room_id)
                .order_by(desc(chatMessage.created_at), desc(chatMessage.chat_message_id))
                .limit(1)
            )
            result = await self.session.execute(statement)
            latest = result.scalars().first()
            if latest is None:
                return None
            return latest
        except Exception as e:
            logger.debug("Failed to get latest chat message by chat room ID")
            raise
    
    async def create_chat_message(self, chat_message: chatMessage) -> chatMessage:
        """Insert a message, commit, and return it refreshed."""
        try:
            self.session.add(chat_message)
            await self.session.commit()
            await self.session.refresh(chat_message)
            return chat_message
        except Exception as e:
            logger.debug("Failed to create chat message")
            raise
    
    async def get_chat_room_participants(self, chat_room_id: int) -> List[chatParticipant]:
        """Get every participant row (active or not) of a room."""
        try:
            statement = (
                select(chatParticipant)
                .where(chatParticipant.chat_room_id == chat_room_id)
            )
            result = await self.session.execute(statement)
            return result.scalars().all()
        except Exception as e:
            logger.debug("Failed to get chat room participants")
            raise

    async def get_chat_room_participant_by_user_id(self, chat_room_id: int, user_id: int) -> chatParticipant | None:
        """Get the user's participant row in a room, or None if never a member."""
        try:
            statement = (
                select(chatParticipant)
                .where(
                    chatParticipant.chat_room_id == chat_room_id,
                    chatParticipant.user_id == user_id,
                )
            )
            result = await self.session.execute(statement)
            return result.scalars().first()
        except Exception as e:
            logger.debug("Failed to get chat room participant by user id")
            raise

    async def reactivate_participant(self, participant: chatParticipant) -> chatParticipant:
        """Make a former member (row with ``is_active=False``) active again and commit."""
        try:
            now = datetime.now()
            participant.is_active = True
            participant.left_at = None
            participant.joined_at = now
            participant.updated_at = now
            self.session.add(participant)
            await self.session.commit()
            await self.session.refresh(participant)
            return participant
        except Exception:
            await self.session.rollback()
            logger.debug("Failed to reactivate chat participant")
            raise

    async def leave_chat_room(self, chat_room_id: int, user_id: int) -> bool:
        """Mark the user's participation as left; archive the room if nobody is left.

        Both updates share one commit, so a partial failure cannot leave a room
        active with zero members. The room row is locked first (``FOR UPDATE``) so two
        members leaving at the same moment run one after the other; otherwise each
        could still see the other as active and neither would archive the room.

        Returns:
            True when this was the last active member and the room was archived
            (``chatroom.is_active = False``, hidden from every room list).

        Raises:
            ValueError: The user is not an active participant of the room.
        """
        try:
            now = datetime.now()
            # Serialize concurrent leaves of the same room (see docstring). No-op on
            # SQLite (tests); Postgres holds the lock until the commit below.
            await self.session.execute(
                select(chatRoom.chat_room_id)
                .where(chatRoom.chat_room_id == chat_room_id)
                .with_for_update()
            )
            result = await self.session.execute(
                update(chatParticipant)
                .where(
                    chatParticipant.chat_room_id == chat_room_id,
                    chatParticipant.user_id == user_id,
                    chatParticipant.is_active == True,  # noqa: E712 (SQL comparison)
                )
                .values(is_active=False, left_at=now, updated_at=now)
            )
            if result.rowcount == 0:
                await self.session.rollback()
                raise ValueError("You are not a participant of this chat room")
            remaining = await self.session.execute(
                select(func.count())
                .select_from(chatParticipant)
                .where(
                    chatParticipant.chat_room_id == chat_room_id,
                    chatParticipant.is_active == True,  # noqa: E712 (SQL comparison)
                )
            )
            archived = remaining.scalar_one() == 0
            if archived:
                await self.session.execute(
                    update(chatRoom)
                    .where(chatRoom.chat_room_id == chat_room_id)
                    .values(is_active=False, updated_at=now)
                )
            await self.session.commit()
            return archived
        except ValueError:
            raise
        except Exception:
            await self.session.rollback()
            logger.debug("Failed to leave chat room")
            raise

    async def update_chat_room_details(
        self, chat_room_id: int, values: dict
    ) -> chatRoom:
        """Apply ``values`` (``chat_room_name`` / ``description``) to a room and commit.

        Raises:
            sqlalchemy.exc.NoResultFound: If the room does not exist.
        """
        try:
            # execute().scalars(): this module imports SQLAlchemy's ``select``, for which
            # ``session.exec(...).one()`` yields a Row, not the chatRoom instance.
            result = await self.session.execute(
                select(chatRoom).where(chatRoom.chat_room_id == chat_room_id)
            )
            room = result.scalars().one()
            for key, value in values.items():
                setattr(room, key, value)
            room.updated_at = datetime.now()
            self.session.add(room)
            await self.session.commit()
            await self.session.refresh(room)
            return room
        except Exception:
            await self.session.rollback()
            logger.debug("Failed to update chat room")
            raise

    async def is_participant(self, chat_room_id: int, user_id: int) -> bool:
        """Return True if the user is an active participant of the room.

        Args:
            chat_room_id: Room to check.
            user_id: User to check.

        Returns:
            False when the user has no participant row or has left the room.
        """
        try:
            statement = (
                select(chatParticipant.chat_participant_id)
                .where(
                    chatParticipant.chat_room_id == chat_room_id,
                    chatParticipant.user_id == user_id,
                    chatParticipant.is_active == True,  # noqa: E712 (SQL comparison)
                )
                .limit(1)
            )
            result = await self.session.execute(statement)
            return result.first() is not None
        except Exception as e:
            logger.debug("Failed to check room membership")
            raise

    async def get_user_presence_by_ids(self, user_ids: List[int]) -> dict[int, dict]:
        """Fetch user profile and last-login data from the auth user table.

        Args:
            user_ids: User ids to look up (duplicates are ignored).

        Returns:
            Mapping ``user_id -> {user_id, username, email, full_name, last_login}``;
            ids with no matching user are omitted.
        """
        if not user_ids:
            return {}

        try:
            # NOTE(cross-service): reads the auth service's "user" table directly via
            # raw SQL (shared Supabase DB). A single IN query covers all ids.
            ids = sorted({int(value) for value in user_ids})
            # expanding=True renders ``IN (:a, :b, ...)`` for a list parameter.
            statement = text(
                'SELECT user_id, username, email, full_name, last_login '
                'FROM "user" WHERE user_id IN :user_ids'
            ).bindparams(bindparam("user_ids", expanding=True))
            result = await self.session.execute(statement, {"user_ids": ids})
            return {int(row["user_id"]): dict(row) for row in result.mappings().all()}
        except Exception as e:
            logger.debug("Failed to get user presence by ids")
            raise
    
    async def update_chat_room_last_message(
        self,
        chat_room_id: int,
        message: str,
    ) -> chatRoom:
        """
        Update the last message preview fields in a chat room (single UPDATE).

        Args:
            chat_room_id: Room to update.
            message: Preview text to store in ``last_message_text``.

        Returns:
            The room re-read from the DB after the commit.

        NOTE: DB schema defines `last_message_id` and `last_message_sender_id` as UUID.
        Current chat messages and users use integer IDs, so we intentionally do NOT
        write those UUID fields here (avoids asyncpg DataError).
        """
        try:
            stmt = (
                update(chatRoom)
                .where(chatRoom.chat_room_id == chat_room_id)
                .values(
                    last_message_text=message,
                    last_message_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )
            await self.session.execute(stmt)
            await self.session.commit()
            return await self.get_chat_room_by_id(chat_room_id)
        except Exception as e:
            logger.debug("Failed to update chat room last message")
            raise
