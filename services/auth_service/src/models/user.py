"""Auth service database models (model layer): ``User`` and ``RefreshToken`` tables."""

from datetime import datetime
import uuid
from typing import Optional
from sqlmodel import Field
from shared.models.base import Base


class User(Base, table=True):
    """User account table.

    ``user_id`` is a random 31-bit integer generated client-side (not a DB sequence).
    ``hashed_password`` stores the bcrypt hash of the SHA-256 pre-hashed password,
    never the plain password.
    """
    
    # TODO(auth): random 31-bit id can collide and is not a DB sequence; needs a migration to change.
    user_id: int = Field(primary_key=True, default_factory=lambda: int(uuid.uuid4().int) % (2**31))
    # TODO(auth): username is UNIQUE but nullable and defaults to full_name at registration.
    username: Optional[str] = Field(default=None, unique=True, index=True, max_length=100)
    email: str = Field(unique=True, index=True, max_length=100)
    hashed_password: str = Field(max_length=255)
    full_name: Optional[str] = Field(default=None, max_length=100)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = Field(default=None)
    
class RefreshToken(Base, table=True):
    """Opaque refresh token table, one row per issued token.

    ``token`` holds the random string sent to the client (looked up by exact value).
    Rotation marks the old row ``revoked`` and inserts a new one.
    """

    token_id: int = Field(primary_key=True, default_factory=lambda: int(uuid.uuid4().int) % (2**31))
    user_id: int = Field(index=True)
    token: str = Field(unique=True, max_length=255)
    expires_at: datetime
    revoked: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.now)
    last_used_at: Optional[datetime] = Field(default=None)
    