"""Security helpers for the auth service (utility layer).

Password hashing (SHA-256 pre-hash + bcrypt), JWT access-token creation/verification
and opaque refresh-token generation. Used by ``AuthService`` and the auth router.
"""

import secrets
from datetime import datetime, timedelta
import hashlib
from typing import Any, Dict, Optional
import bcrypt
import jwt
from jwt import InvalidTokenError

from shared.logger import get_logger

logger = get_logger("auth-service")


class SecurityUtil:
    """Utility class for security-related functions like password hashing and token management."""
    
    def __init__(self, jwt_secret_key: str, jwt_algorithm: str = "HS256",
                 access_token_expire_minutes: int = 30,
                 refresh_token_expire_days: int = 7):
        """Store signing configuration.

        Args:
            jwt_secret_key: HMAC secret for signing JWTs. SECURITY: must equal the
                gateway's ``JWT_SECRET_KEY`` and must never be logged.
            jwt_algorithm: JWT signing algorithm (the gateway must accept the same one).
            access_token_expire_minutes: Access-token lifetime in minutes.
            refresh_token_expire_days: Refresh-token lifetime in days (used for DB expiry).
        """
        self.jwt_secret_key = jwt_secret_key
        self.jwt_algorithm = jwt_algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password for storage.

        SECURITY: the password is first reduced to a fixed 32-byte SHA-256 digest so
        bcrypt's 72-byte input limit cannot silently truncate long passwords, then
        hashed with bcrypt and a fresh random salt.

        Args:
            password: Plain-text password.

        Returns:
            The bcrypt hash as a UTF-8 string (safe to store in ``hashed_password``).
        """
        sha256_hash = hashlib.sha256(password.encode("utf-8")).digest()
        hashed = bcrypt.hashpw(sha256_hash, bcrypt.gensalt())
        return hashed.decode('utf-8')

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Check a plain password against a stored hash (same SHA-256 + bcrypt scheme).

        Args:
            plain_password: Password supplied by the user.
            hashed_password: Stored bcrypt hash.

        Returns:
            True on match; False on mismatch or any error (errors are logged).
        """
        try:
            sha256_hash = hashlib.sha256(plain_password.encode("utf-8")).digest()
            return bcrypt.checkpw(sha256_hash, hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error("Password verification error: %s", e)
            return False
        
    def _create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a signed JWT access token.

        SECURITY: this token is what the gateway verifies on every protected request.
        The ``type: access`` claim is added here and the gateway rejects any token
        whose ``type`` is not ``"access"`` (so refresh tokens cannot call APIs).

        Args:
            data: Claims to embed (``user_id``, ``email``, ``sub``).

        Returns:
            Encoded JWT whose ``exp`` is now + ``access_token_expire_minutes`` (UTC).
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        
    def _create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT refresh token (legacy, unused by AuthService).

        Prefer ``generate_random_refresh_token`` which is stored in the DB and revocable.
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, self.jwt_secret_key, algorithm=self.jwt_algorithm)

    @staticmethod
    def generate_random_refresh_token() -> str:
        """Generate a cryptographically random opaque refresh token.

        SECURITY: uses ``secrets`` (CSPRNG), 48 random bytes (~64 URL-safe chars). The
        value is stored in the DB and sent to the client as an HttpOnly cookie.
        """
        return secrets.token_urlsafe(48)

    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT's signature and expiry.

        SECURITY: only the configured algorithm is accepted (``algorithms=[...]``), which
        prevents algorithm-confusion attacks.

        Args:
            token: Encoded JWT from the ``Authorization: Bearer`` header.

        Returns:
            The decoded claims, or None if the token is invalid or expired.
        """
        try:
            payload = jwt.decode(token, self.jwt_secret_key, algorithms=[self.jwt_algorithm])
            return payload
        except InvalidTokenError:
            return None