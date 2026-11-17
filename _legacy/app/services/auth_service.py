"""
Authentication Manager for the Authentication Service.
"""

import asyncio
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import bcrypt
import hashlib
from jose import JWTError, jwt
import secrets

from app.core.database_manager import get_database_manager
from app.repositories.base import UserRepository, UserSessionRepository, UserTokenRepository
from app.schemas.pydantic_schemas import UserRegistration, UserLogin, TokenResponse, UserProfile

logger = logging.getLogger(__name__)


class AuthManager:
    """Manager for authentication and authorization operations."""
    
    def __init__(self):
        """Initialize the authentication manager."""
        # Initialize database manager
        self.db_manager = get_database_manager("auth")
        
        # JWT settings (will be loaded from config)
        self.jwt_secret_key = None
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    async def initialize(self):
        """Initialize the authentication manager with database connection."""
        try:
            # Initialize database
            self.db_manager.initialize()
            await self.db_manager.create_tables()
            
            # Load JWT settings from config
            from app.core.config import config
            self.jwt_secret_key = config.get("jwt.secret_key", secrets.token_urlsafe(32))
            self.jwt_algorithm = config.get("jwt.algorithm", "HS256")
            self.access_token_expire_minutes = config.get("jwt.access_token_expire_minutes", 30)
            self.refresh_token_expire_days = config.get("jwt.refresh_token_expire_days", 7)
            
            logger.info("Authentication manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize authentication manager: {str(e)}")
            raise
    
    def _hash_password(self, password: str) -> str:
        """Hash a password safely for bcrypt by pre-hashing to ensure 72-byte limit.

        Bcrypt accepts up to 72 bytes. To support arbitrarily long and
        non-ASCII passwords without truncation, we first SHA-256 hash the
        UTF-8 encoded password, which produces exactly 32 bytes (256 bits).
        The raw 32-byte digest is then passed to bcrypt, staying well under
        the 72-byte limit.
        """
        # SHA-256 produces 32 bytes (well under bcrypt's 72-byte limit)
        sha256_hash = hashlib.sha256(password.encode("utf-8")).digest()
        # Hash the 32-byte digest with bcrypt
        hashed = bcrypt.hashpw(sha256_hash, bcrypt.gensalt())
        return hashed.decode('utf-8')
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password hashed with the pre-hash strategy above."""
        try:
            sha256_hash = hashlib.sha256(plain_password.encode("utf-8")).digest()
            return bcrypt.checkpw(sha256_hash, hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False
    
    def _create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, self.jwt_secret_key, algorithm=self.jwt_algorithm)
    
    def _create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, self.jwt_secret_key, algorithm=self.jwt_algorithm)
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.jwt_secret_key, algorithms=[self.jwt_algorithm])
            return payload
        except JWTError:
            return None
    
    async def register_user(self, registration_data: UserRegistration) -> Dict[str, Any]:
        """Register a new user."""
        try:
            async with self.db_manager.get_async_session() as session:
                user_repo = UserRepository(session)
                
                # Check if user already exists by email
                existing_email = await user_repo.get_user_by_email(registration_data.email)
                if existing_email:
                    return {
                        "success": False,
                        "message": "Email already exists",
                        "error": "EMAIL_EXISTS"
                    }
                
                # Hash password
                hashed_password = self._hash_password(registration_data.password)
                
                # Create user
                user = await user_repo.create_user(
                    username=None,  # Username will be set later in settings
                    email=registration_data.email,
                    hashed_password=hashed_password,
                    full_name=registration_data.full_name
                )
                
                return {
                    "success": True,
                    "message": "User registered successfully",
                    "user": {
                        "id": user.user_id,
                        "username": user.username,
                        "email": user.email,
                        "full_name": user.full_name,
                        "created_at": user.created_at.isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to register user: {str(e)}")
            return {
                "success": False,
                "message": "Registration failed",
                "error": str(e)
            }
    
    async def authenticate_user(self, login_data: UserLogin) -> Dict[str, Any]:
        """Authenticate a user with username and password."""
        try:
            async with self.db_manager.get_async_session() as session:
                user_repo = UserRepository(session)
                session_repo = UserSessionRepository(session)
                
                # Get user by email
                user = await user_repo.get_user_by_email(login_data.email)
                if not user:
                    return {
                        "success": False,
                        "message": "Invalid credentials",
                        "error": "INVALID_CREDENTIALS"
                    }
                
                # Check if user is active
                if not user.is_active:
                    return {
                        "success": False,
                        "message": "Account is deactivated",
                        "error": "ACCOUNT_DEACTIVATED"
                    }
                
                # Verify password
                if not self._verify_password(login_data.password, user.hashed_password):
                    return {
                        "success": False,
                        "message": "Invalid credentials",
                        "error": "INVALID_CREDENTIALS"
                    }
                
                # Update last login
                await user_repo.update_last_login(user.user_id)
                
                # Create tokens
                access_token = self._create_access_token({"sub": user.user_id, "email": user.email})
                refresh_token = self._create_refresh_token({"sub": user.user_id, "email": user.email})
                
                # Create session
                expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
                await session_repo.create_session(
                    user_id=user.user_id,
                    session_token=access_token,
                    refresh_token=refresh_token,
                    expires_at=expires_at
                )
                
                return {
                    "success": True,
                    "message": "Authentication successful",
                    "user": {
                        "id": user.user_id,
                        "username": user.username,
                        "email": user.email,
                        "full_name": user.full_name
                    },
                    "tokens": {
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "token_type": "bearer",
                        "expires_in": self.access_token_expire_minutes * 60
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to authenticate user: {str(e)}")
            return {
                "success": False,
                "message": "Authentication failed",
                "error": str(e)
            }
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user info."""
        try:
            payload = self._verify_token(token)
            if not payload or payload.get("type") != "access":
                return None
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            async with self.db_manager.get_async_session() as session:
                user_repo = UserRepository(session)
                session_repo = UserSessionRepository(session)
                
                # Check if session is still active
                user_session = await session_repo.get_session_by_token(token)
                if not user_session:
                    return None
                
                # Get user info
                user = await user_repo.get_user_by_id(user_id)
                if not user or not user.is_active:
                    return None
                
                return {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "is_active": user.is_active
                }
                
        except Exception as e:
            logger.exception("Failed to verify token")
            return None
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token."""
        try:
            async with self.db_manager.get_async_session() as session:
                user_repo = UserRepository(session)
                session_repo = UserSessionRepository(session)
                
                # Verify refresh token
                payload = self._verify_token(refresh_token)
                if not payload or payload.get("type") != "refresh":
                    return {
                        "success": False,
                        "message": "Invalid refresh token",
                        "error": "INVALID_REFRESH_TOKEN"
                    }
                
                # Check if session exists and is active
                user_session = await session_repo.get_session_by_refresh_token(refresh_token)
                if not user_session:
                    return {
                        "success": False,
                        "message": "Invalid refresh token",
                        "error": "INVALID_REFRESH_TOKEN"
                    }
                
                # Get user
                user = await user_repo.get_user_by_id(user_session.user_id)
                if not user or not user.is_active:
                    return {
                        "success": False,
                        "message": "User not found or inactive",
                        "error": "USER_NOT_FOUND"
                    }
                
                # Create new access token
                new_access_token = self._create_access_token({"sub": user.user_id, "username": user.username})
                
                # Update session with new access token
                await session_repo.revoke_session(user_session.session_token)
                expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
                await session_repo.create_session(
                    user_id=user.user_id,
                    session_token=new_access_token,
                    refresh_token=refresh_token,
                    expires_at=expires_at
                )
                
                return {
                    "success": True,
                    "message": "Token refreshed successfully",
                    "access_token": new_access_token,
                    "token_type": "bearer",
                    "expires_in": self.access_token_expire_minutes * 60
                }
                
        except Exception as e:
            logger.error(f"Failed to refresh token: {str(e)}")
            return {
                "success": False,
                "message": "Token refresh failed",
                "error": str(e)
            }
    
    async def logout(self, access_token: str) -> Dict[str, Any]:
        """Logout user by revoking session."""
        try:
            async with self.db_manager.get_async_session() as session:
                session_repo = UserSessionRepository(session)
                
                # Revoke session
                success = await session_repo.revoke_session(access_token)
                
                if success:
                    return {
                        "success": True,
                        "message": "Logged out successfully"
                    }
                else:
                    return {
                        "success": False,
                        "message": "Session not found",
                        "error": "SESSION_NOT_FOUND"
                    }
                
        except Exception as e:
            logger.error(f"Failed to logout: {str(e)}")
            return {
                "success": False,
                "message": "Logout failed",
                "error": str(e)
            }
