"""Pydantic request/response schemas for chat-service (schema layer).

In use by the routers/services: ``ChatMessage``, ``ChatRoom``,
``ChatRoomCreateRequest``, ``ChatMessageRequest`` and the ``*Response`` models.
The conversation/join/leave request models are not referenced anywhere in ``src/``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# Base Models
class ChatMessage(BaseModel):
    """Internal chat message DTO (ids are strings; converted to ints in the service)."""
    id: Optional[str] = None
    user_id: Optional[int] = None
    role: str = "user"
    message: str
    chat_room_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ChatRoom(BaseModel):
    """Internal chat room DTO passed from router to service."""
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
    """Body of ``POST /chat/messages``: ``conversation_id`` is the chat room id and
    ``user_id`` must equal the gateway ``x-user-id`` header."""
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
    user_id: Optional[int] = None
    role: str = "user"
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


class ChatParticipantResponse(BaseModel):
    """Response model for a user's membership in a chat room."""
    chat_participant_id: int
    chat_room_id: int
    user_id: int
    username: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    last_login: Optional[datetime] = None
    joined_at: datetime
    left_at: Optional[datetime] = None
    is_active: bool = True
    status: str = "active"


class LatestChatMessageResponse(BaseModel):
    """Compact response model for the latest message in a chat room.

    ``chat_message_id`` and ``user_id`` mirror ``chatRoom.last_message_id`` and
    ``chatRoom.last_message_sender_id``, which are UUID columns, so they are
    serialized as strings here.
    """
    chat_message_id: str
    chat_room_id: str
    user_id: Optional[str] = None
    message: str
    created_at: datetime


class UserChatRoomResponse(BaseModel):
    """
    Response model for a chat room the user belongs to.

    Combines data from:
    - `chatRoom`
    - `latest_message` (optional latest message)
    """
    chat_room_id: int
    chat_room_name: str
    description: Optional[str] = None
    room_created_at: datetime
    room_updated_at: Optional[datetime] = None
    room_is_active: bool = True
    latest_message: Optional[LatestChatMessageResponse] = None
    current_user_status: Optional["CurrentUserChatRoomStatusResponse"] = None


class UserChatRoomListResponse(BaseModel):
    """Response model for all chat rooms that a user belongs to."""
    user_id: int
    chat_rooms: List[UserChatRoomResponse]
    total_count: int


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


class LastMessagePreviewResponse(BaseModel):
    """Minimal last-message preview, built from the newest ``chatMessage`` row."""
    message: Optional[str] = None
    created_at: Optional[datetime] = None


class CurrentUserChatRoomStatusResponse(BaseModel):
    """Current user's membership status in a chat room."""
    user_id: int
    chat_room_id: int
    is_member: bool
    is_active: bool
    joined_at: Optional[datetime] = None
    left_at: Optional[datetime] = None


class ChatMessageHistoryResponse(BaseModel):
    """Message history payload for a room, including participants and last-message preview."""
    messages: List[ChatMessageResponse]
    participants: List[ChatParticipantResponse]
    last_message: Optional[LastMessagePreviewResponse] = None
    current_user_status: Optional[CurrentUserChatRoomStatusResponse] = None
