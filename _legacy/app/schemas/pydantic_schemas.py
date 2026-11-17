"""
Pydantic schemas for the RAG Chat Room Web application.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class BaseResponse(BaseModel):
    """Base response model for all endpoints."""
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error: str
    error_code: Optional[str] = None


class HealthResponse(BaseResponse):
    """Health check response model."""
    success: bool = True
    status: str  # healthy, degraded, unhealthy
    version: str
    components: Dict[str, str]


# Authentication Schemas
class UserRegistration(BaseModel):
    """User registration request model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request model."""
    username: str
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


class AuthResponse(BaseResponse):
    """Authentication response model."""
    user: Optional[UserProfile] = None
    tokens: Optional[TokenResponse] = None


# Document Service Schemas
class DocumentExtractRequest(BaseModel):
    """Request model for document extraction."""
    path: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentExtractResult(BaseModel):
    """Result model for document extraction."""
    content: str
    metadata: Dict[str, Any]
    confidence: float


class DocumentExtractResponse(BaseResponse):
    """Response model for document extraction."""
    result: DocumentExtractResult


class DocumentExtractAndStoreRequest(DocumentExtractRequest):
    """Request model for document extraction and storage."""
    title: Optional[str] = None
    category: Optional[str] = None


class DocumentExtractAndStoreResponse(DocumentExtractResponse):
    """Response model for document extraction and storage."""
    document_id: str
    stored_successfully: bool


# RAG Service Schemas
class RAGRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    context: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7


class RAGResponse(BaseResponse):
    """Response model for RAG queries."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float


# Vector Store Service Schemas
class VectorStoreRequest(BaseModel):
    """Request model for vector store operations."""
    documents: List[Dict[str, Any]]
    collection_name: Optional[str] = None


class VectorStoreResponse(BaseResponse):
    """Response model for vector store operations."""
    document_ids: List[str]
    collection_name: str


class VectorSearchRequest(BaseModel):
    """Request model for vector search."""
    query: str
    collection_name: Optional[str] = None
    limit: int = 10
    threshold: float = 0.7


class VectorSearchResponse(BaseResponse):
    """Response model for vector search."""
    results: List[Dict[str, Any]]
    query: str
    total_results: int


# Web Crawler Service Schemas
class CrawlRequest(BaseModel):
    """Request model for web crawling."""
    url: str
    max_depth: int = 1
    max_pages: int = 10
    follow_links: bool = True


class CrawlResponse(BaseResponse):
    """Response model for web crawling."""
    crawled_urls: List[str]
    total_pages: int
    errors: List[str] = Field(default_factory=list)


# Chat Service Schemas
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request model for chat."""
    message: str = Field(..., description="The message to send to the chatbot")
    conversation_id: Optional[str] = Field(None, description="The ID of the conversation")
    user_id: int = Field(..., description="The ID of the user sending the message")
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseResponse):
    """Response model for chat."""
    message: str
    conversation_id: str
    message_id: str
    context: Optional[Dict[str, Any]] = None
