"""
Chat-specific Pydantic schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# Base Models
class ChatMessage(BaseModel):
    """Chat message model."""
    id: Optional[str] = None
    user_id: int
    message: str
    chat_room_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ChatRoom(BaseModel):
    """Chat room model."""
    chat_room_id: Optional[str] = None
    chat_room_name: str
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ConversationMetadata(BaseModel):
    """Conversation metadata model."""
    conversation_id: str
    created_at: datetime
    last_updated: datetime
    message_count: int


# Request Schemas
class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message."""
    message: str = Field(..., min_length=1, description="The message content")
    conversation_id: Optional[str] = Field(None, description="The ID of the conversation/chat room")
    user_id: int = Field(..., description="The ID of the user sending the message")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context for the message")


class ChatRoomCreateRequest(BaseModel):
    """Request model for creating a chat room."""
    chat_room_name: str = Field(..., min_length=1, max_length=100, description="Name of the chat room")


class ConversationHistoryRequest(BaseModel):
    """Request model for getting conversation history."""
    conversation_id: str = Field(..., description="The ID of the conversation")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Maximum number of messages to retrieve")
    offset: Optional[int] = Field(0, ge=0, description="Offset for pagination")


class ConversationDeleteRequest(BaseModel):
    """Request model for deleting a conversation."""
    conversation_id: str = Field(..., description="The ID of the conversation to delete")


class ChatRoomJoinRequest(BaseModel):
    """Request model for joining a chat room."""
    chat_room_id: str = Field(..., description="The ID of the chat room to join")
    user_id: int = Field(..., description="The ID of the user joining")


class ChatRoomLeaveRequest(BaseModel):
    """Request model for leaving a chat room."""
    chat_room_id: str = Field(..., description="The ID of the chat room to leave")
    user_id: int = Field(..., description="The ID of the user leaving")


# Response Schemas
class ChatMessageResponse(BaseModel):
    """Response model for a chat message."""
    id: str
    user_id: int
    message: str
    chat_room_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class ChatResponse(BaseModel):
    """Response model for chat message processing."""
    message: str
    message_id: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRoomResponse(BaseModel):
    """Response model for chat room information."""
    chat_room_id: str
    chat_room_name: str
    created_at: datetime
    message_count: Optional[int] = 0
    user_count: Optional[int] = 0


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    conversation_id: str
    messages: List[ChatMessageResponse]
    total_count: int
    limit: int
    offset: int


class ConversationListResponse(BaseModel):
    """Response model for listing conversations."""
    conversations: List[ConversationMetadata]
    total_count: int


class ChatRoomListResponse(BaseModel):
    """Response model for listing chat rooms."""
    chat_rooms: List[ChatRoomResponse]
    total_count: int


class ChatRoomDetailResponse(ChatRoomResponse):
    """Response model for detailed chat room information."""
    messages: Optional[List[ChatMessageResponse]] = None
    users: Optional[List[int]] = None  # List of user IDs in the room
