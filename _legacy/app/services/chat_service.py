"""
Chat Manager for the Chat Service.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import config

from app.db.models import ChatRoom, User_Chat_Room
from app.repositories.base import ChatMessageRepository, ChatRoomRepository
from app.core.database_manager import get_database_manager

logger = logging.getLogger(__name__)


class ChatManager:
    """Manager for chat and conversation operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize the chat manager."""
        self.session = session
        self.chat_room_repository: Optional[ChatRoomRepository] = None
        self.chat_message_repository: Optional[ChatMessageRepository] = None
    async def initialize(self):
        """Initialize the chat manager with database connection."""
        try:
            # Initialize database
            await self.db_manager.create_tables()
            
            # Use ChatOpenAI with GitHub Models endpoint
            self.llm = ChatOpenAI(
                model="openai/gpt-4o",
                openai_api_key=config['github_token'],
                openai_api_base="https://models.github.ai/inference",
                temperature=0.7,
                max_tokens=4000,
                max_retries=2
            )
            
            # self.response_generator = ResponseGenerator(
            #     llm=llm,
            #     max_retries=3,
            #     temperature=0.7,
            #     max_tokens=1000
            # )
            
            logger.info("Chat manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat manager: {str(e)}")
            raise
        
    async def process_message(
        self,
        message: str,
        conversation_id: str,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message.
        
        Args:
            message: User's message
            conversation_id: Conversation ID
            context: Optional context
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Generate message ID
            message_id = str(uuid.uuid4())
            
            # Store user message
            user_message = {
                'id': message_id,
                'user_id': user_id,
                'content': message,
                'timestamp': datetime.utcnow().isoformat(),
                'context': context
            }
            chatRepository = None
            async with self.db_manager.get_async_session() as session:
                 chatRepository = ChatMessageRepository(session)
                
            await chatRepository.add_message_to_conversation(conversation_id, message, user_id)
            
            # Add to conversation
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
                self.conversation_metadata[conversation_id] = {
                    'created_at': datetime.utcnow().isoformat(),
                    'last_updated': datetime.utcnow().isoformat(),
                    'message_count': 0
                }
            
            self.conversations[conversation_id].append(user_message)
            self.conversation_metadata[conversation_id]['message_count'] += 1
            self.conversation_metadata[conversation_id]['last_updated'] = datetime.utcnow().isoformat()
            
            # Generate assistant response
            assistant_response = await self._generate_response(message, conversation_id, context)
            await ChatMessageRepository(session).add_message_to_conversation(conversation_id, assistant_response['message'], user_id)
            # Store assistant response
            assistant_message = {
                'id': str(uuid.uuid4()),
                'content': assistant_response['message'],
                'timestamp': datetime.utcnow().isoformat(),
                'context': assistant_response.get('context')
            }
            
            self.conversations[conversation_id].append(assistant_message)
            self.conversation_metadata[conversation_id]['message_count'] += 1
            self.conversation_metadata[conversation_id]['last_updated'] = datetime.utcnow().isoformat()
            
            logger.info(f"Processed message in conversation {conversation_id}")
            
            return {
                'message': assistant_response['message'],
                'message_id': assistant_message['id'],
                'context': assistant_response.get('context')
            }
            
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            raise
    
    async def _generate_response(
        self,
        message: str,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to the user's message.
        
        Args:
            message: User's message
            conversation_id: Conversation ID
            context: Optional context
            
        Returns:
            Dictionary with response message and context
        """
        try:
            # TODO: Integrate with RAG service for intelligent responses
            response =  self.llm.invoke(
                [SystemMessage(content="You are a helpful AI assistant. for a group chat, you will just need to answer the question but not based on any context or previous messages, you will just need to answer the question based on your knowledge and the question itself, and maybe if you need to search web for the answer."), 
                 HumanMessage(content=message)])
            response_message = response.content
            
            # Mock context
            response_context = {
                'conversation_id': conversation_id,
                'response_type': 'text',
                'confidence': 0.8,
                'sources': []  # Would be populated by RAG service
            }
            
            return {
                'message': response_message,
                'context': response_context
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            # Return a fallback response
            return {
                'message': "I apologize, but I'm having trouble processing your message right now. Please try again.",
                'context': {'error': str(e)}
            }
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of messages in the conversation
        """
        try:
            messages = await self.chat_message_repository.get_messages_by_conversation_id(conversation_id)
            if conversation_id not in self.conversations:
                return []
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
            raise
    
    async def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Conversation ID to delete
        """
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            
            if conversation_id in self.conversation_metadata:
                del self.conversation_metadata[conversation_id]
            
            logger.info(f"Deleted conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete conversation: {str(e)}")
            raise
    
    async def get_user_chat_rooms(self, user_id: int) -> List[Dict[str, Any]]:
        """
        List all user's chat rooms.
        
        Returns:
            List of user's chat rooms
        """
        try:
            chat_rooms = []
            user_chat_rooms = await User_Chat_Room.filter(user_id=user_id).all()
            for user_chat_room in user_chat_rooms:
                chat_rooms.append({
                    'chat_room_id': user_chat_room.chat_room_id,
                    'chat_room_name': user_chat_room.chat_room_name,
                    'created_at': user_chat_room.created_at
                })
            
            return chat_rooms
            
        except Exception as e:
            logger.error(f"Failed to list conversations: {str(e)}")
            raise
    
    async def create_chat_room(self, chat_room_name: str) -> None:
        """
        Create a chat room.
        
        Args:
            chat_room_name: Chat room name
        """
        try:
            chat_room = ChatRoom(chat_room_name=chat_room_name)
            async with self.db_manager.get_async_session() as session:
                chat_room_repository = ChatRoomRepository(session)
                await chat_room_repository.create_chat_room(chat_room)
        except Exception as e:
            logger.error(f"Failed to create chat room: {str(e)}")
            raise
