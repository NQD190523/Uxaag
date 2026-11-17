"""
Shared database models for microservices.
"""

from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

# Create the base class for declarative models
Base = declarative_base()


class User(Base):
    """User model for authentication service."""
    __tablename__ = "users"
    
    user_id = Column(Integer(), primary_key=True, default=lambda: int(uuid.uuid4().int) % (2**31))
    username = Column(String(100), unique=True, nullable=True, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    tokens = relationship("UserToken", back_populates="user", cascade="all, delete-orphan")
    chat_rooms = relationship("User_Chat_Room", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")

    # Instance-level alias for compatibility (so code using user.id still works)
    @property
    def id(self):
        return self.user_id


class UserSession(Base):
    """User session model for tracking active sessions."""
    __tablename__ = "user_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer(), ForeignKey("users.user_id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), unique=True, nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_used = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class UserToken(Base):
    """User token model for JWT tokens."""
    __tablename__ = "user_tokens"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer(), ForeignKey("users.user_id"), nullable=False)
    token_type = Column(String(20), nullable=False)  # access, refresh, reset
    token_hash = Column(String(255), nullable=False, index=True)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="tokens")


class PasswordReset(Base):
    """Password reset model."""
    __tablename__ = "password_resets"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer(), ForeignKey("users.user_id"), nullable=False)
    reset_token = Column(String(255), unique=True, nullable=False, index=True)
    is_used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User")


# Service-specific models can be added here
class ServiceHealth(Base):
    """Service health tracking model."""
    __tablename__ = "service_health"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    service_name = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False)  # healthy, degraded, unhealthy
    last_check = Column(DateTime, default=datetime.utcnow)
    response_time = Column(Integer, nullable=True)  # milliseconds
    error_message = Column(Text, nullable=True)

class ChatMessage(Base):
    """Chat message model."""
    __tablename__ = "chat_messages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer(), ForeignKey("users.user_id"), nullable=False)
    message = Column(Text, nullable=False)
    chat_room_id = Column(String(36), ForeignKey("chat_rooms.chat_room_id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="messages")
    chat_room = relationship("ChatRoom", back_populates="messages")
    
    
class User_Chat_Room(Base):
    """User chat room model."""
    __tablename__ = "user_chat_rooms"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer(), ForeignKey("users.user_id"), nullable=False)
    chat_room_id = Column(String(36), ForeignKey("chat_rooms.chat_room_id"), nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="chat_rooms")
    chat_room = relationship("ChatRoom", back_populates="users")

class ChatRoom(Base):
    """Chat room model."""
    __tablename__ = "chat_rooms"
    
    chat_room_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)
    chat_room_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="chat_room", cascade="all, delete-orphan")
    users = relationship("User_Chat_Room", back_populates="chat_room", cascade="all, delete-orphan")