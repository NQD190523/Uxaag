"""
Repositories Package

This package contains repository classes organized by domain/feature:
- auth_repository.py - All authentication-related repositories (User, UserSession, UserToken, PasswordReset)
- chat_repository.py - All chat-related repositories (ChatMessage, ChatRoom, UserChatRoom)
- service_health_repository.py - Service health repository

For backward compatibility, existing imports from app.repositories.base still work.
For new code, you can import directly from domain repository files or use this __init__.py.
"""

# Base repository classes
from app.repositories.base_repository import BaseRepository, AsyncBaseRepository

# Domain-specific repositories
from app.repositories.auth_repository import (
    UserRepository,
    UserSessionRepository,
    UserTokenRepository,
    PasswordResetRepository,
)
from app.repositories.chat_repository import (
    ChatMessageRepository,
    ChatRoomRepository,
    UserChatRoomRepository,
)
from app.repositories.service_health_repository import ServiceHealthRepository

__all__ = [
    # Base classes
    "BaseRepository",
    "AsyncBaseRepository",
    # Auth repositories
    "UserRepository",
    "UserSessionRepository",
    "UserTokenRepository",
    "PasswordResetRepository",
    # Chat repositories
    "ChatMessageRepository",
    "ChatRoomRepository",
    "UserChatRoomRepository",
    # Other repositories
    "ServiceHealthRepository",
]
