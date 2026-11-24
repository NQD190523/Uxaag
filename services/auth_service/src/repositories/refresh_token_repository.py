"""
Refresh token repository for persisting and looking up refresh tokens.

Layer: repository. Async queries on the ``RefreshToken`` table; each write method
commits and rolls back on error.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import select, update

from shared.logger import get_logger
from models.user import RefreshToken

from repositories.user_repository import AsyncBaseRepository

logger = get_logger("auth-service")


class RefreshTokenRepository(AsyncBaseRepository):
    """Repository for refresh token storage and lookup."""

    async def create(self, user_id: int, token: str, expires_at: datetime) -> RefreshToken:
        """Persist an opaque refresh token string for a user (exact value used for lookup on verify).

        Args:
            user_id: Owner of the token.
            token: Random opaque token string sent to the client.
            expires_at: Expiry as a naive UTC datetime.

        Returns:
            The created row.

        Raises:
            Exception: Any DB error, after rollback.
        """
        try:
            record = RefreshToken(
                user_id=user_id,
                token=token,
                expires_at=expires_at,
                revoked=False,
            )
            self.session.add(record)
            await self.session.commit()
            await self.session.refresh(record)
            logger.info(f"Refresh token created for user_id={user_id}")
            return record
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create refresh token: {str(e)}")
            raise

    async def get_by_token(self, token: str) -> Optional[RefreshToken]:
        """Find a refresh token by its exact value (opaque random string stored at issue).

        SECURITY: the token is stored and compared in plain text (not hashed), so a DB
        leak exposes live refresh tokens.
        TODO(security): store a hash of the token and compare hashes.

        Returns:
            The matching row, or None if absent. DB errors are logged and also yield None.
        """
        try:
            result = await self.session.execute(
                select(RefreshToken).where(RefreshToken.token == token)
            )
            logger.info("Refresh token lookup by value for user_id completed")
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get refresh token: {str(e)}")
            return None

    async def revoke_by_token(self, token: str) -> bool:
        """Mark a refresh token as revoked (used on rotation).

        Returns:
            True if a row was updated.

        Raises:
            Exception: Any DB error, after rollback.
        """
        try:
            result = await self.session.execute(
                update(RefreshToken)
                .where(RefreshToken.token == token)
                .values(revoked=True)
            )
            await self.session.commit()
            logger.info("Refresh token revoked successfully")
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to revoke refresh token: {str(e)}")
            raise

    async def revoke_all_for_user(self, user_id: int) -> int:
        """Revoke every still-active refresh token of a user.

        Used after a password change so other sessions cannot keep refreshing.

        Args:
            user_id: Owner whose tokens are revoked.

        Returns:
            Number of rows that were newly revoked.

        Raises:
            Exception: Any DB error, after rollback.
        """
        try:
            result = await self.session.execute(
                update(RefreshToken)
                .where(RefreshToken.user_id == user_id)
                .where(RefreshToken.revoked == False)  # noqa: E712 (SQL comparison)
                .values(revoked=True)
            )
            await self.session.commit()
            logger.info("Revoked %s refresh token(s) for user_id=%s", result.rowcount, user_id)
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            logger.error("Failed to revoke refresh tokens for user_id=%s: %s", user_id, e)
            raise

    async def update_last_used(self, token: str) -> bool:
        """Update last_used_at for a refresh token (audit trail).

        Returns:
            True if a row was updated.

        Raises:
            Exception: Any DB error, after rollback.
        """
        try:
            result = await self.session.execute(
                update(RefreshToken)
                .where(RefreshToken.token == token)
                .values(last_used_at=datetime.now())
            )
            await self.session.commit()
            logger.info("Refresh token last_used_at updated successfully")
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update last_used_at: {str(e)}")
            raise
