"""
Repository layer for the auth service.

Defines the base repository classes and ``UserRepository`` (async queries on the
``User`` table). Write methods commit and roll back on error; the service layer
decides how to map failures to API errors.
"""

from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from sqlalchemy import select, update, delete, or_
from datetime import datetime

from shared.logger import get_logger
from models.user import User

# Get configured logger
logger = get_logger("auth-service")

class BaseRepository:
    """Base repository class holding a synchronous session (unused by this service)."""
    def __init__(self, session: Session):
        self.session = session


class AsyncBaseRepository:
    """Base async repository: stores the request-scoped ``AsyncSession``."""
    def __init__(self, session: AsyncSession):
        self.session = session


class UserRepository(AsyncBaseRepository):
    """Repository for user-related database operations."""
         
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by exact username.

        Returns:
            The user, or None if not found.

        Raises:
            Exception: Any DB error (logged and re-raised).
        """
        try:
            result = await self.session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.debug("User lookup by username failed")
            raise
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by exact email.

        Returns:
            The user, or None if not found. NOTE: DB errors are also swallowed and
            returned as None, so callers cannot tell "missing" from "DB failure".
        """
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.debug("User lookup by email failed")
            return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by primary key.

        Returns:
            The user, or None if not found. NOTE: on a DB error the exception is
            logged and the method implicitly returns None.
        """
        try:
            result = await self.session.execute(
                select(User).where(User.user_id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.debug("User lookup by id failed user_id=%s", user_id)
    
    async def create_user(self, user_data: User) -> User:
        """Insert a new user and commit.

        Returns:
            The persisted user (refreshed from the DB).

        Raises:
            Exception: Any DB error (e.g. unique violation on email/username), after rollback.
        """
        try:
            self.session.add(user_data)
            await self.session.commit()
            await self.session.refresh(user_data)
            return user_data
        except Exception as e:
            await self.session.rollback()
            logger.debug("User create failed")
            raise
    
    async def update_user(self, user_id: int, update_data: Dict[str, Any]) -> bool:
        """Apply a partial update to a user and bump ``updated_at``.

        SECURITY: ``update_data`` keys go straight into ``UPDATE ... SET``; callers must
        pass only whitelisted columns (never raw request bodies).

        Returns:
            True if a row was updated.

        Raises:
            Exception: Any DB error, after rollback.
        """
        try:
            result = await self.session.execute(
                update(User)
                .where(User.user_id == user_id)
                .values(**update_data, updated_at=datetime.utcnow())
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.debug("User update failed user_id=%s", user_id)
            raise
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete a user row (hard delete). Returns True if a row was deleted."""
        try:
            result = await self.session.execute(
                delete(User).where(User.user_id == user_id)
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.debug("User delete failed user_id=%s", user_id)
            raise
    
    async def update_last_login(self, user_id: int) -> bool:
        """Set ``last_login`` to now (UTC). Returns True if a row was updated."""
        try:
            result = await self.session.execute(
                update(User)
                .where(User.user_id == user_id)
                .values(last_login=datetime.utcnow())
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.debug("Last-login update failed user_id=%s", user_id)
            raise
        
    async def get_matching_users(self, search_query: str) -> List[User]:
        """Search users by username, full name, email, or user_id.

        Case-insensitive substring match; a numeric query also matches ``user_id``
        exactly. No row limit is applied.

        Returns:
            All matching users.
        """
        try:
            filters = [
                User.username.ilike(f"%{search_query}%"),
                User.full_name.ilike(f"%{search_query}%"),
                User.email.ilike(f"%{search_query}%"),
            ]

            # If input is numeric, also match user_id exactly.
            if search_query.isdigit():
                filters.append(User.user_id == int(search_query))

            result = await self.session.execute(
                select(User).where(or_(*filters))
            )
            users = result.scalars().all()
            return users
        except Exception as e:
            logger.debug("Matching-users query failed")
            raise
