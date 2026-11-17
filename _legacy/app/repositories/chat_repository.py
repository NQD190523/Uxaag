"""
Chat-related repositories grouped together.
Contains repositories for ChatMessage, ChatRoom, and User_Chat_Room models.
"""

from typing import List, Optional
from sqlalchemy import select
import logging

from app.repositories.base_repository import AsyncBaseRepository
from app.db.models import ChatMessage, ChatRoom, User_Chat_Room

logger = logging.getLogger(__name__)


class ChatMessageRepository(AsyncBaseRepository):
    """Repository for chat message operations."""
    
    async def create_message(self, user_id: str, message: str) -> ChatMessage:
        """Create a new chat message."""
        try:
            message = ChatMessage(user_id=user_id, message=message)
            self.session.add(message)
            await self.session.commit()
            await self.session.refresh(message)
            return message
        except Exception as e:
            logger.error(f"Failed to create chat message: {str(e)}")
            raise
    
    async def get_messages_by_user_id(self, user_id: str) -> List[ChatMessage]:
        """Get chat messages by user ID."""
        try:
            result = await self.session.execute(
                select(ChatMessage).where(ChatMessage.user_id == user_id)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Failed to get chat messages by user ID {user_id}: {str(e)}")
            raise
    
    async def add_message_to_conversation(self, conversation_id: str, message: str, user_id: int) -> ChatMessage:
        """Add a message to a conversation."""
        try:
            message = ChatMessage(conversation_id=conversation_id, message=message, user_id=user_id)
            self.session.add(message)
            await self.session.commit()
            await self.session.refresh(message)
            return message
        except Exception as e:
            logger.error(f"Failed to add message to conversation: {str(e)}")
            raise


class ChatRoomRepository(AsyncBaseRepository):
    """Repository for chat room operations."""
    
    # Placeholder for future chat room operations
    # Add methods here as needed
    async def create_chat_room(self, chat_room_name: str) -> ChatRoom:
        """Create a new chat room."""
        try:
            chat_room = ChatRoom(chat_room_name=chat_room_name)
            self.session.add(chat_room)
            await self.session.commit()
            await self.session.refresh(chat_room)
            return chat_room
        except Exception as e:
            logger.error(f"Failed to create chat room: {str(e)}")
            return None
        


class UserChatRoomRepository(AsyncBaseRepository):
    """Repository for user chat room operations."""
    
    # Placeholder for future user chat room operations
    # Add methods here as needed
