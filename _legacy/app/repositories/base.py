"""
Repository pattern for database operations.

NOTE: This file is kept for backward compatibility.
All repositories have been organized by domain/feature:
- app.repositories.auth_repository - All authentication-related repositories
- app.repositories.chat_repository - All chat-related repositories

For new code, import directly from the domain repository files:
- from app.repositories.auth_repository import UserRepository, UserSessionRepository
- from app.repositories.chat_repository import ChatMessageRepository
etc.
"""

# Re-export from organized repository files for backward compatibility
from app.repositories.base_repository import BaseRepository, AsyncBaseRepository
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

__all__ = [
    "BaseRepository",
    "AsyncBaseRepository",
    "UserRepository",
    "UserSessionRepository",
    "UserTokenRepository",
    "PasswordResetRepository",
    "ChatMessageRepository",
    "ChatRoomRepository",
    "UserChatRoomRepository",
]