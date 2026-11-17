"""
Authentication-specific Pydantic schemas.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any
from datetime import datetime


class UserRegistration(BaseModel):
    """User registration request model."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request model."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserProfile(BaseModel):
    """User profile model."""
    user_id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class PasswordChange(BaseModel):
    """Password change request model."""
    old_password: str
    new_password: str = Field(..., min_length=8)


class PasswordReset(BaseModel):
    """Password reset request model."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""
    token: str
    new_password: str = Field(..., min_length=8)


class AuthResponse(BaseModel):
    """Authentication response model."""
    success: bool
    message: str
    user: Optional[UserProfile] = None
    tokens: Optional[TokenResponse] = None
