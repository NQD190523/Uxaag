"""
Authentication-related repositories grouped together.
Contains repositories for User, UserSession, UserToken, and PasswordReset models.
"""

from typing import Optional, Dict, Any
from sqlalchemy import select, update, delete, and_
from datetime import datetime
import logging

from app.repositories.base_repository import AsyncBaseRepository
from app.db.models import User, UserSession, UserToken, PasswordReset

logger = logging.getLogger(__name__)


class UserRepository(AsyncBaseRepository):
    """Repository for user-related database operations."""
    
    async def create_user(
        self, 
        username: str, 
        email: str, 
        hashed_password: str, 
        full_name: Optional[str] = None
    ) -> User:
        """Create a new user."""
        try:
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name
            )
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            return user
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create user: {str(e)}")
            raise
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        try:
            result = await self.session.execute(
                select(User).where(User.user_id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {str(e)}")
            raise
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            result = await self.session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {str(e)}")
            raise
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {str(e)}")
            raise
    
    async def update_user(self, user_id: int, update_data: Dict[str, Any]) -> bool:
        """Update user information."""
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
            logger.error(f"Failed to update user {user_id}: {str(e)}")
            raise
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete user."""
        try:
            result = await self.session.execute(
                delete(User).where(User.user_id == user_id)
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to delete user {user_id}: {str(e)}")
            raise
    
    async def update_last_login(self, user_id: int) -> bool:
        """Update user's last login time."""
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
            logger.error(f"Failed to update last login for user {user_id}: {str(e)}")
            raise


class UserSessionRepository(AsyncBaseRepository):
    """Repository for user session operations."""
    
    async def create_session(
        self, 
        user_id: str, 
        session_token: str, 
        refresh_token: str,
        expires_at: datetime,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserSession:
        """Create a new user session."""
        try:
            session = UserSession(
                user_id=user_id,
                session_token=session_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent
            )
            self.session.add(session)
            await self.session.commit()
            await self.session.refresh(session)
            return session
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create session: {str(e)}")
            raise
    
    async def get_session_by_token(self, session_token: str) -> Optional[UserSession]:
        """Get session by session token."""
        try:
            result = await self.session.execute(
                select(UserSession)
                .where(
                    and_(
                        UserSession.session_token == session_token,
                        UserSession.is_active == True,
                        UserSession.expires_at > datetime.utcnow()
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get session by token: {str(e)}")
            raise
    
    async def get_session_by_refresh_token(self, refresh_token: str) -> Optional[UserSession]:
        """Get session by refresh token."""
        try:
            result = await self.session.execute(
                select(UserSession)
                .where(
                    and_(
                        UserSession.refresh_token == refresh_token,
                        UserSession.is_active == True,
                        UserSession.expires_at > datetime.utcnow()
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get session by refresh token: {str(e)}")
            raise
    
    async def revoke_session(self, session_token: str) -> bool:
        """Revoke a session."""
        try:
            result = await self.session.execute(
                update(UserSession)
                .where(UserSession.session_token == session_token)
                .values(is_active=False)
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to revoke session: {str(e)}")
            raise
    
    async def revoke_all_user_sessions(self, user_id: str) -> bool:
        """Revoke all sessions for a user."""
        try:
            result = await self.session.execute(
                update(UserSession)
                .where(UserSession.user_id == user_id)
                .values(is_active=False)
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to revoke all sessions for user {user_id}: {str(e)}")
            raise
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            result = await self.session.execute(
                delete(UserSession).where(
                    UserSession.expires_at < datetime.utcnow()
                )
            )
            await self.session.commit()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            raise


class UserTokenRepository(AsyncBaseRepository):
    """Repository for user token operations."""
    
    async def create_token(
        self, 
        user_id: str, 
        token_type: str, 
        token_hash: str, 
        expires_at: datetime
    ) -> UserToken:
        """Create a new user token."""
        try:
            token = UserToken(
                user_id=user_id,
                token_type=token_type,
                token_hash=token_hash,
                expires_at=expires_at
            )
            self.session.add(token)
            await self.session.commit()
            await self.session.refresh(token)
            return token
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create token: {str(e)}")
            raise
    
    async def revoke_token(self, token_hash: str) -> bool:
        """Revoke a token."""
        try:
            result = await self.session.execute(
                update(UserToken)
                .where(UserToken.token_hash == token_hash)
                .values(is_revoked=True)
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to revoke token: {str(e)}")
            raise
    
    async def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens."""
        try:
            result = await self.session.execute(
                delete(UserToken).where(
                    UserToken.expires_at < datetime.utcnow()
                )
            )
            await self.session.commit()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to cleanup expired tokens: {str(e)}")
            raise


class PasswordResetRepository(AsyncBaseRepository):
    """Repository for password reset operations."""
    
    # Placeholder for future password reset operations
    # Add methods here as needed
