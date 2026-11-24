"""SQLModel tables owned by chat-service (model layer): rooms, messages, participants.

NOTE: IDs are generated client-side as ``uuid4().int % 2**31`` (random ints, not
sequences). Schema changes require an Alembic migration (see CLAUDE.md).
"""

from datetime import datetime
import uuid
from typing import Optional
from sqlmodel import Field
import sqlalchemy as sa
from shared.models.base import Base

class chatRoom(Base, table=True):
    """A chat room, with a denormalized last-message preview for the room list.

    NOTE: ``last_message_id`` / ``last_message_sender_id`` are UUID columns, but
    messages and users use integer ids, so the repository never writes them.
    """
    chat_room_id: int = Field(
        primary_key=True, default_factory=lambda: int(uuid.uuid4().int) % (2**31))
    chat_room_name: str = Field(index=True, max_length=255)
    last_message_id: Optional[uuid.UUID] = Field(
    default=None, index=True)
    last_message_text: Optional[str] = Field(max_length=200, default=None)
    last_message_at: Optional[datetime] = Field(default=None, index=True)
    last_message_sender_id: Optional[uuid.UUID] = Field(
    default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    description: Optional[str] = Field(default=None, max_length=255)
    is_active: bool = Field(default=True)

class chatMessage(Base, table=True):
    """One message in a room. ``role`` is ``user`` or ``assistant``; ``user_id`` is
    NULL for assistant (LLM) messages."""
    chat_message_id: int = Field(primary_key=True, default_factory=lambda: int(uuid.uuid4().int) % (2**31))
    chat_room_id: int = Field(index=True)
    user_id: Optional[int] = Field(default=None, index=True, nullable=True)
    role: str = Field(
        default="user",
        sa_column=sa.Column(sa.String(length=32), nullable=False, server_default="user", index=True),
    )
    message: str = Field(sa_column=sa.Column(sa.Text(), nullable=False))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class chatParticipant(Base, table=True):
    """Membership of a user in a room (``left_at`` set when the user leaves)."""
    chat_participant_id: int = Field(primary_key=True, default_factory=lambda: int(uuid.uuid4().int) % (2**31))
    chat_room_id: int = Field(index=True)
    user_id: int = Field(index=True)
    joined_at: datetime = Field(default_factory=datetime.now)
    # Type hint only: the column is already nullable with default None.
    left_at: Optional[datetime] = Field(default=None, nullable=True)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    