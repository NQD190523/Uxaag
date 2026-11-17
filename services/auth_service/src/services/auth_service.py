"""
Auth Service - Handles authentication and authorization business logic

Layer: service. Registration, login and refresh-token rotation. Uses
``UserRepository`` / ``RefreshTokenRepository`` for DB access and ``SecurityUtil``
for hashing and JWT creation. Raises ``ValueError`` for all failures; the router
maps them to HTTP codes.
"""


import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from repositories.user_repository import UserRepository
from repositories.refresh_token_repository import RefreshTokenRepository
from models.user import User
from .base_service import BaseService
from core.security import SecurityUtil
from schemas.auth_schema import TokenResponse, UserRegistration, UserLogin, UserProfile, AuthResponse


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
                self.log_error("Invalid registration data: missing email or password")
                raise ValueError("Email and password are required")
            
            # Initialize UserRepository
            user_repository = UserRepository(session)
            
            # Check if email already exists
            existing_user = await user_repository.get_user_by_email(registration_data.email)
            if existing_user:
                self.log_error("Registration failed: email already exists")
                raise ValueError("Email is already registered")
            
            # SECURITY: SHA-256 pre-hash + bcrypt; the plain password is never stored or logged.
            hashed_password = SecurityUtil.hash_password(registration_data.password)
            self.log_debug("Password hashed for new user")
            
            # Ensure security_util is available
            if not self.security_util:
                raise ValueError("SecurityUtil must be initialized in AuthService")
            
            # Prepare user data for database
            user_data = User(
                email=registration_data.email,
                # TODO(auth): username defaults to full_name, but username is UNIQUE, so two
                # users with the same full name collide on insert; generate a unique one.
                username=registration_data.full_name,
                hashed_password=hashed_password,
                full_name=registration_data.full_name,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Create user in database
            created_user = await user_repository.create_user(user_data)
            self.log_info("User registered successfully")
            
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
            
        except ValueError as ve:
            self.log_error(f"Registration validation error: {str(ve)}")
            raise
        except Exception as e:
            # Unexpected failures propagate as-is so the router returns a generic
            # 500 instead of a 400 carrying raw internal error text.
            self.log_error(f"Registration failed with unexpected error: {str(e)}")
            raise

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
            self.log_error("Invalid login data: missing email or password")
            raise ValueError("Email and password are required")

        user_repository = UserRepository(session)
        db_user = await user_repository.get_user_by_email(login_data.email)

        if not db_user:
            self.log_warning("Login failed: user not found")
            raise ValueError("Invalid email or password")

        # SECURITY: same generic error as for an unknown email (see docstring).
        is_valid = self.security_util.verify_password(login_data.password, db_user.hashed_password)
        if not is_valid:
            self.log_warning("Login failed: invalid password")
            raise ValueError("Invalid email or password")

        if not getattr(db_user, "is_active", True):
            self.log_warning("Login failed: inactive user")
            raise ValueError("Account is disabled")

        await user_repository.update_last_login(db_user.user_id)
        db_user.last_login = datetime.utcnow()
        self.log_info("User logged in successfully")

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

        Verification is by DB lookup only (token string must match stored value; no JWT decode).
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
            ValueError: If the token is unknown, revoked or expired.
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
            
            # Only user_id is re-issued here (no email/sub claims); the gateway accepts
            # user_id or sub as the identity.
            token_data = {"user_id": str(refresh_token_record.user_id)}
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
            await refresh_token_repo.revoke_by_token(refresh_token_record.token)

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
        except ValueError as ve:
            self.log_error(f"Refresh token failed: {str(ve)}")
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
            self.log_error(f"Verify refresh token failed: {str(e)}")
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
        except ValueError as ve:
            self.log_error(f"Get me failed: {str(ve)}")
            raise
        except Exception as e:
            # Re-raise unexpected errors for the router's generic 500 handler.
            self.log_error(f"Get me failed with unexpected error: {str(e)}")
            raise
