"""Pydantic request/response schemas for the auth service (schema layer).

Covers registration, login, token payloads, user profile and password change.
"""

# Pydantic Models
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


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
    """Token pair returned on register/login/refresh.

    ``expires_in`` is the access-token lifetime in seconds. ``refresh_token`` is
    also set as an HttpOnly cookie by the router; ``/auth/refresh`` blanks it in the
    JSON body.
    """
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: float

class UserProfile(BaseModel):
    """Public-facing user profile. ``user_id`` is serialised as a string."""
    user_id: str
    username: Optional[str] = None
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None

class UserProfileUpdate(BaseModel):
    """Fields the current user can edit from their profile page."""
    username: Optional[str] = Field(default=None, min_length=1, max_length=100)
    full_name: Optional[str] = Field(default=None, min_length=1, max_length=100)


class PasswordChangeRequest(BaseModel):
    """Password change request for the current user."""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8)


class AuthResponse(BaseModel):
    """Authentication response model."""
    success: bool
    message: str
    user: Optional[UserProfile] = None
    token: Optional[TokenResponse] = None


class UserPageResponse(BaseModel):
    """Frontend-friendly response model for user page profile data."""
    success: bool
    message: str
    user: UserProfile
    
class MatchingUsersResponse(BaseModel):
    """One user search hit (``POST /user/matching-users``); never includes the password hash."""
    user_id: int
    username: Optional[str] = None
    email: str
    full_name: Optional[str] = None
