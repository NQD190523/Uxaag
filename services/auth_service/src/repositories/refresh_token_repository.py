"""
Refresh token repository for persisting and looking up refresh tokens.

Layer: repository. Async queries on the ``RefreshToken`` table; each write method
commits and rolls back on error.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, not_, or_, select, update

from shared.logger import get_logger
from core.security import SecurityUtil
from models.user import RefreshToken

from repositories.user_repository import AsyncBaseRepository

logger = get_logger("auth-service")


def _matches(token: str):
    """SQL condition matching the row of a RAW refresh token.

    Matches the hashed form (every row written since hashing) or, for legacy rows that
    still hold the raw token, the exact value.
    SECURITY: the exact-value branch skips ``sha256:`` rows. Without that, a stored
    digest read from a DB leak would itself work as a refresh token (it equals its
    own row), which is exactly what hashing is meant to prevent. Callers must
    therefore always pass the raw token, never ``record.token``.
    TODO(security): drop the exact-value branch once legacy rows have expired
    (refresh lifetime is 7 days, so after 2026-12-01).
    """
    return or_(
        RefreshToken.token == SecurityUtil.hash_refresh_token(token),
        and_(
            RefreshToken.token == token,
            not_(RefreshToken.token.startswith("sha256:")),
        ),
    )


class RefreshTokenRepository(AsyncBaseRepository):
    """Repository for refresh token storage and lookup.

    Callers always pass the raw token; hashing happens here so it cannot be forgotten.
    """

    async def create(self, user_id: int, token: str, expires_at: datetime) -> RefreshToken:
        """Persist a refresh token for a user. Only its SHA-256 digest is stored.

        Args:
            user_id: Owner of the token.
            token: Raw random token sent to the client (hashed before insert).
            expires_at: Expiry as a naive UTC datetime.

        Returns:
            The created row.

        Raises:
            Exception: Any DB error, after rollback.
        """
        try:
            record = RefreshToken(
                user_id=user_id,
                token=SecurityUtil.hash_refresh_token(token),
                expires_at=expires_at,
                revoked=False,
            )
            self.session.add(record)
            await self.session.commit()
            await self.session.refresh(record)
            logger.debug("Refresh token created user_id=%s", user_id)
            return record
        except Exception as e:
            await self.session.rollback()
            logger.debug("Refresh token create failed user_id=%s", user_id)
            raise

    async def get_by_token(self, token: str) -> Optional[RefreshToken]:
        """Find the row for a raw refresh token (hashed match, or a legacy plain row).

        Returns:
            The matching row, or None if absent. DB errors are logged and also yield None.
        """
        try:
            result = await self.session.execute(
                select(RefreshToken).where(_matches(token))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.debug("Refresh token lookup failed")
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
                .where(_matches(token))
                .values(revoked=True)
            )
            await self.session.commit()
            logger.debug("Refresh token revoked")
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.debug("Refresh token revoke failed")
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
            logger.info("Refresh tokens revoked count=%s user_id=%s", result.rowcount, user_id)
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            logger.debug("Refresh tokens revoke failed user_id=%s", user_id)
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
                .where(_matches(token))
                .values(last_used_at=datetime.now())
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.debug("Refresh token last_used_at update failed")
            raise
