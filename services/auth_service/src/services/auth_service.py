"""
Auth Service - Handles authentication and authorization business logic

Layer: service. Registration, login and refresh-token rotation. Uses
``UserRepository`` / ``RefreshTokenRepository`` for DB access and ``SecurityUtil``
for hashing and JWT creation. Raises ``ValueError`` for all failures; the router
maps them to HTTP codes.
"""


import logging
import re
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional

from repositories.user_repository import UserRepository
from repositories.refresh_token_repository import RefreshTokenRepository
from models.user import User
from .base_service import BaseService
from core.security import SecurityUtil
from schemas.auth_schema import TokenResponse, UserRegistration, UserLogin, UserProfile, AuthResponse


# Characters kept from the email local part when building a username handle.
_HANDLE_UNSAFE = re.compile(r"[^a-z0-9._-]+")
# Room for a "-1234" suffix inside the 100-char column, while keeping handles readable.
_HANDLE_MAX_BASE = 30


def _handle_base(email: str) -> str:
    """Return a lower-case handle candidate from the email's local part.

    ``"Nguyen.Duc+work@x.com"`` -> ``"nguyen.duc"``. The ``+tag`` is dropped (it is
    usually a mail filter, not a name); anything outside ``[a-z0-9._-]`` is removed.
    Falls back to ``"user"`` when nothing usable is left.
    """
    local = email.split("@", 1)[0].split("+", 1)[0].lower()
    base = _HANDLE_UNSAFE.sub("", local).strip("._-")[:_HANDLE_MAX_BASE]
    return base or "user"


class AuthService(BaseService):
    """
    Service for handling authentication business logic.
    Handles user registration, login, refresh-token rotation and profile lookup.
    """

    def __init__(self, logger: logging.Logger = None, security_util: Optional[SecurityUtil] = None):
        """
        Initialize the Auth Service.
        
        Args:
            logger: Logger instance
            security_util: SecurityUtil instance for token creation and password operations.
                          If None, will raise error when used (must be provided).
        """
        super().__init__(logger)
        self.security_util = security_util

    async def register_user(self, registration_data: UserRegistration, session: Any) -> AuthResponse:
        """
        Register a new user with email and password.
        
        Converts UserRegistration schema to User model, creates user in database,
        and returns AuthResponse with user profile.

        Also issues the first token pair: a JWT access token and an opaque refresh
        token that is persisted in the DB.
        
        Args:
            registration_data: UserRegistration schema containing email, password, full_name
            session: Async database session
            
        Returns:
            AuthResponse: Response with success status, message, and user profile
            
        Raises:
            ValueError: If email already exists, or on any other failure (unexpected
                errors are re-wrapped as ValueError, so the router returns 400 for them).
        """
        try:
            # Validate registration data
            if not registration_data.email or not registration_data.password:
                self.log_warning("Registration rejected reason=%s", "missing_fields")
                raise ValueError("Email and password are required")
            
            # Initialize UserRepository
            user_repository = UserRepository(session)
            
            # Check if email already exists
            existing_user = await user_repository.get_user_by_email(registration_data.email)
            if existing_user:
                self.log_warning("Registration rejected reason=%s", "email_exists")
                raise ValueError("Email is already registered")
            
            # SECURITY: SHA-256 pre-hash + bcrypt; the plain password is never stored or logged.
            hashed_password = SecurityUtil.hash_password(registration_data.password)
            
            # Ensure security_util is available
            if not self.security_util:
                raise ValueError("SecurityUtil must be initialized in AuthService")
            
            # Prepare user data for database
            user_data = User(
                email=registration_data.email,
                # Bug A7: username used to be full_name, which is UNIQUE, so two people
                # with the same name could not both sign up.
                username=await self._unique_username(registration_data.email, user_repository),
                hashed_password=hashed_password,
                full_name=registration_data.full_name,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Create user in database
            created_user = await user_repository.create_user(user_data)
            
            # Create access token (JWT) and random opaque refresh token
            token_data = {
                "user_id": str(created_user.user_id),
                "email": created_user.email,
                "sub": str(created_user.user_id)  # Standard JWT subject claim
            }
            access_token = self.security_util._create_access_token(token_data)
            refresh_token = SecurityUtil.generate_random_refresh_token()
            # Persist refresh token for revocation and audit
            expires_at = datetime.utcnow() + timedelta(days=self.security_util.refresh_token_expire_days)
            refresh_token_repo = RefreshTokenRepository(session)
            await refresh_token_repo.create(created_user.user_id, refresh_token, expires_at)
            token = TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=self.security_util.access_token_expire_minutes * 60
            )
            # Convert User model to UserProfile schema
            user_profile = UserProfile(
                user_id=str(created_user.user_id) if hasattr(created_user, 'user_id') else str(created_user.id),
                username=getattr(created_user, 'username', None),
                email=getattr(created_user, 'email', ''),
                full_name=getattr(created_user, 'full_name', None),
                is_active=getattr(created_user, 'is_active', True),
                created_at=getattr(created_user, 'created_at', datetime.utcnow()),
                last_login=getattr(created_user, 'last_login', None)
            )
            
            # Return success response with user profile
            return AuthResponse(
                success=True,
                message="User registered successfully",
                user=user_profile,
                token=token
            )
            
        except ValueError:
            # Already logged above with a specific reason; the router maps it to 400.
            raise
        except Exception as e:
            # Unexpected failures propagate as-is so the router returns a generic
            # 500 instead of a 400 carrying raw internal error text.
            # The router logs the single traceback (log once).
            self.log_debug("Registration failed")
            raise

    async def _unique_username(self, email: str, user_repository: UserRepository) -> str:
        """Pick a free username handle for a new user, derived from their email.

        Tries the plain handle (``duc``), then up to five ``duc-NNNN`` variants with a
        random 4-digit suffix, then a ``duc-<8 hex>`` one that practically cannot be
        taken.

        Why random suffixes, not ``duc2``, ``duc3``: counting up needs one query per
        existing duplicate and leaks how many accounts share the name.
        Note: the check and the insert are separate, so two signups racing for the same
        handle can still hit the UNIQUE constraint; the insert then fails like any other
        DB error and the user can retry.

        Args:
            email: The new user's email.
            user_repository: Repository bound to the request session.

        Returns:
            A username not currently in use.
        """
        base = _handle_base(email)
        candidates = [base] + [f"{base}-{secrets.randbelow(9000) + 1000}" for _ in range(5)]
        for candidate in candidates:
            if await user_repository.get_user_by_username(candidate) is None:
                return candidate
        return f"{base}-{secrets.token_hex(4)}"

    async def login_user(self, login_data: UserLogin, session: Any) -> AuthResponse:
        """
        Authenticate a user with email and password.

        Looks up user by email, verifies password, and returns AuthResponse
        with user profile and access/refresh tokens.

        SECURITY: unknown email and wrong password both raise the same
        "Invalid email or password" message to avoid user enumeration. The
        inactive-account check runs only after the password is verified.

        Args:
            login_data: UserLogin schema containing email and password
            session: Async database session

        Returns:
            AuthResponse: Response with success status, message, user profile, and tokens

        Raises:
            ValueError: If email/password invalid or user not found
        """
        if not self.security_util:
            raise ValueError("SecurityUtil must be initialized in AuthService")

        if not login_data.email or not login_data.password:
            self.log_warning("Login rejected reason=%s", "missing_fields")
            raise ValueError("Email and password are required")

        user_repository = UserRepository(session)
        db_user = await user_repository.get_user_by_email(login_data.email)

        if not db_user:
            self.log_warning("Login rejected reason=%s", "user_not_found")
            raise ValueError("Invalid email or password")

        # SECURITY: same generic error as for an unknown email (see docstring).
        is_valid = self.security_util.verify_password(login_data.password, db_user.hashed_password)
        if not is_valid:
            self.log_warning("Login rejected reason=%s", "invalid_password")
            raise ValueError("Invalid email or password")

        if not getattr(db_user, "is_active", True):
            self.log_warning("Login rejected reason=%s", "inactive_user")
            raise ValueError("Account is disabled")

        await user_repository.update_last_login(db_user.user_id)
        db_user.last_login = datetime.utcnow()

        token_data = {
            "user_id": str(db_user.user_id),
            "email": db_user.email,
            "sub": str(db_user.user_id),
        }
        # Create access token (JWT) and random opaque refresh token
        access_token = self.security_util._create_access_token(token_data)
        refresh_token = SecurityUtil.generate_random_refresh_token()
        expires_at = datetime.utcnow() + timedelta(days=self.security_util.refresh_token_expire_days)
        
        # Create refresh token record
        refresh_token_repo = RefreshTokenRepository(session)
        await refresh_token_repo.create(db_user.user_id, refresh_token, expires_at)
        
        # Create token response
        token = TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.security_util.access_token_expire_minutes * 60,
        )

        # Create user profile
        user_profile = UserProfile(
            user_id=str(db_user.user_id) if hasattr(db_user, "user_id") else str(getattr(db_user, "id", "")),
            username=getattr(db_user, "username", None),
            email=getattr(db_user, "email", ""),
            full_name=getattr(db_user, "full_name", None),
            is_active=getattr(db_user, "is_active", True),
            created_at=getattr(db_user, "created_at", datetime.utcnow()),
            last_login=getattr(db_user, "last_login", None),
        )

        return AuthResponse(
            success=True,
            message="Login successful",
            user=user_profile,
            token=token,
        )

    async def refresh_token(self, refresh_token: str, session: Any) -> AuthResponse:
        """
        Refresh an access token using an opaque refresh token (rotation).

        Verification is by DB lookup only (no JWT decode): the row is found by
        ``SecurityUtil.hash_refresh_token(token)``, or by the raw value for legacy rows
        written before hashing.
        On success a new access token and a new refresh token are issued and the
        presented refresh token is revoked.

        SECURITY: the new token is created before the old one is revoked, in separate
        commits and without a transaction, so concurrent requests with the same token
        can both succeed. Replaying a revoked token is rejected but does not revoke
        the rest of the user's tokens (no reuse detection).

        Args:
            refresh_token: Opaque token from the ``refresh_token`` cookie.
            session: Async database session.

        Returns:
            AuthResponse with ``user=None`` and the new token pair.

        Raises:
            ValueError: If the token is unknown, revoked or expired, or its user
                no longer exists or is disabled.
        """
        try:
            refresh_token_repo = RefreshTokenRepository(session)
            refresh_token_record = await refresh_token_repo.get_by_token(refresh_token)

            if not refresh_token_record:
                raise ValueError("Invalid refresh token")
            if refresh_token_record.revoked:
                raise ValueError("Refresh token revoked")
            if refresh_token_record.expires_at < datetime.utcnow():
                raise ValueError("Refresh token expired")
            
            # Re-issue the same claims as login (user_id, email, sub) so a refreshed token
            # is indistinguishable from a fresh one; the gateway injects x-user-email from
            # it. Loading the user also stops a disabled or deleted account from
            # refreshing its way past a ban.
            db_user = await UserRepository(session).get_user_by_id(
                int(refresh_token_record.user_id)
            )
            if db_user is None:
                raise ValueError("User not found")
            if not getattr(db_user, "is_active", True):
                raise ValueError("Account is disabled")
            token_data = {
                "user_id": str(db_user.user_id),
                "email": db_user.email,
                "sub": str(db_user.user_id),
            }
            access_token = self.security_util._create_access_token(token_data)
            new_refresh_token = SecurityUtil.generate_random_refresh_token()
            # New refresh token expiry (days), stored in DB
            refresh_expires_at = datetime.utcnow() + timedelta(
                days=self.security_util.refresh_token_expire_days
            )
            await refresh_token_repo.create(
                refresh_token_record.user_id, new_refresh_token, refresh_expires_at
            )
            await refresh_token_repo.update_last_used(refresh_token)
            # Raw token, not record.token: the repository only matches raw tokens.
            await refresh_token_repo.revoke_by_token(refresh_token)

            return AuthResponse(
                success=True,
                message="Token refreshed successfully",
                user=None,
                token=TokenResponse(
                    access_token=access_token,
                    refresh_token=new_refresh_token,
                    token_type="bearer",
                    expires_in=self.security_util.access_token_expire_minutes * 60,
                ),
            )
        except ValueError:
            # The router logs the rejection reason; no second line here.
            raise
    
    async def verify_refresh_token(self, session: Any, token: str) -> bool:
        """
        Verify an opaque refresh token by DB lookup only (no JWT decode).
        Checks: record exists, not revoked, not expired (UTC).

        Args:
            session: Async database session.
            token: Opaque refresh token string.

        Returns:
            True if the token is valid.

        Raises:
            ValueError: If the token is invalid, revoked or expired.
        """
        try:
            refresh_token_repo = RefreshTokenRepository(session)
            refresh_token_record = await refresh_token_repo.get_by_token(token)
            if not refresh_token_record:
                raise ValueError("Invalid refresh token")
            if refresh_token_record.revoked:
                raise ValueError("Refresh token revoked")
            if refresh_token_record.expires_at < datetime.utcnow():
                raise ValueError("Refresh token expired")
            return True
        except Exception as e:
            # Expected rejection (invalid/revoked/expired): WARNING, no traceback.
            self.log_warning("Refresh token rejected reason=%s", e)
            raise ValueError(f"Verify refresh token failed: {str(e)}")
    
    async def get_me(self, session: Any, user_id: int) -> UserProfile:
        """
        Get the current user's profile.

        Args:
            session: Async database session.
            user_id: Id taken from the verified JWT.

        Returns:
            The user's profile.

        Raises:
            ValueError: If the user does not exist. Unexpected errors propagate.
        """
        try:
            user_repository = UserRepository(session)
            user = await user_repository.get_user_by_id(user_id)
            if not user:
                raise ValueError("User not found")
            userProfile = UserProfile(
                user_id=str(user.user_id),
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login
            )
            return userProfile
        except ValueError:
            # The router logs the rejection reason; no second line here.
            raise
        except Exception as e:
            # Re-raise unexpected errors for the router's generic 500 handler.
            self.log_debug("Profile fetch failed")
            raise
