"""
API v1 Endpoints
"""

from typing import List
from fastapi import APIRouter, HTTPException, WebSocket, status
from fastapi.responses import JSONResponse
from datetime import datetime

from app.schemas.auth_schemas import UserRegistration, UserLogin
from app.schemas.chat_schemas import ChatRoomCreateRequest, ChatRoomResponse
from app.schemas.pydantic_schemas import ChatRequest, HealthResponse, ErrorResponse, UserProfile
from app.core.logging import get_logger
from app.core.database_manager import get_database_manager
from app.services import auth_service, chat_service
from app.services.socket_connection_service import ConnectionManager
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["v1"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with database connectivity test."""
    components = {}
    overall_status = "healthy"
    
    try:
        # Test API health
        components["api"] = "healthy"
        
        # Test database health
        try:
            db_manager = get_database_manager("main")
            db_manager.initialize()
            db_health = await db_manager.health_check()
            
            if db_health["status"] == "healthy":
                components["database"] = "healthy"
            else:
                components["database"] = "unhealthy"
                overall_status = "degraded"
                
        except Exception as db_error:
            logger.exception("Database health check failed")
            components["database"] = f"unhealthy: {str(db_error)}"
            overall_status = "degraded"
        
        # Test auth service health
        try:
            auth_manager = auth_service.AuthManager()
            await auth_manager.initialize()
            components["auth_service"] = "healthy"
        except Exception as auth_error:
            logger.exception("Auth service health check failed")
            components["auth_service"] = f"unhealthy: {str(auth_error)}"
            overall_status = "degraded"
        
        # Test other services
        components["services"] = "healthy"
        
        return HealthResponse(
            success=True,
            status=overall_status,
            version="1.0.0",
            message=f"RAG Chat Room Web API is {overall_status}",
            components=components
        )
        
    except Exception as e:
        logger.exception("Health check failed")
        return HealthResponse(
            success=False,
            status="unhealthy",
            version="1.0.0",
            message="RAG Chat Room Web API is unhealthy",
            components={"error": str(e)}
        )

@router.get("/health/database")
async def database_health_check():
    """Database-specific health check endpoint."""
    try:
        db_manager = get_database_manager("main")
        db_manager.initialize()
        db_health = await db_manager.health_check()
        
        return JSONResponse(
            status_code=200 if db_health["status"] == "healthy" else 503,
            content=db_health
        )
        
    except Exception as e:
        logger.exception("Database health check failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "main",
                "database": "disconnected",
                "error": str(e)
            }
        )

# Auth endpoints
@router.post("/auth/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserRegistration):
    """
    Register a new user.
    
    - **email**: User's email address
    - **password**: User's password (min 8 characters)
    - **full_name**: User's full name (optional)
    """
    try:
        auth_manager = auth_service.AuthManager()
        await auth_manager.initialize()
        result = await auth_manager.register_user(user)
        
        if result["success"]:
            user_data = result["user"]
            # Convert response to match UserProfile schema
            # Handle datetime parsing safely
            created_at_str = user_data.get("created_at")
            if isinstance(created_at_str, str):
                try:
                    # Try ISO format first (handle both with and without timezone)
                    if created_at_str.endswith('Z'):
                        created_at_str = created_at_str.replace('Z', '+00:00')
                    created_at = datetime.fromisoformat(created_at_str)
                except (ValueError, AttributeError) as e:
                    # Fallback to current time if parsing fails
                    logger.warning(f"Failed to parse created_at: {created_at_str}, error: {e}")
                    created_at = datetime.utcnow()
            elif isinstance(created_at_str, datetime):
                created_at = created_at_str
            else:
                created_at = datetime.utcnow()
            
            return UserProfile(
                user_id=str(user_data["id"]),
                username=user_data.get("username") or "",
                email=user_data["email"],
                full_name=user_data.get("full_name"),
                is_active=True,
                created_at=created_at,
                last_login=None
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception(f"Registration failed with error: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/auth/login")
async def login_user(user: UserLogin):
    """
    Login a user.
    
    - **email**: User's email address
    - **password**: User's password
    """
    try:
        auth_manager = auth_service.AuthManager()
        await auth_manager.initialize()
        result = await auth_manager.authenticate_user(user)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result["message"]
            )
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


# TODO: Add more endpoints here
# Document endpoints
# RAG endpoints  
# Vector store endpoints
# Web crawler endpoints
# Chat endpoints
@router.post("/chat")
async def chat(request: ChatRequest, websocket: WebSocket):
    """
    Send a chat message.
    
    - **question**: User's question
    - **session_id**: Session ID (optional)
    - **use_rag**: Use RAG (optional)
    """
    try:
        chat_manager = chat_service.ChatManager()
        await chat_manager.initialize()
        connection_manager = ConnectionManager()
        
        await connection_manager.connect(websocket, request.conversation_id)
        while True:
            data = await websocket.receive_text()
            if data == "close":
                await connection_manager.disconnect(websocket, request.conversation_id)
                break
            else:
                await chat_manager.process_message(data, request.conversation_id, request.user_id, request.context)
                await connection_manager.broadcast(data, request.conversation_id)
                
        result = await chat_manager.process_message(request.message, request.conversation_id, request.user_id, request.context)
        return result
    except Exception as e:
        logger.exception(f"Chat failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )

@router.post("/chat/create-room", response_model=ChatRoomResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_room(ChatRoomCreateRequest: ChatRoomCreateRequest):
    """
    Create a chat room.
    
    - **chat_room_name**: Name of the chat room
    """
    try:
        chat_manager = chat_service.ChatManager()
        await chat_manager.initialize()
        result = await chat_manager.create_chat_room(ChatRoomCreateRequest.chat_room_name)
        return result
    except Exception as e:
        logger.exception(f"Failed to create chat room: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat room: {str(e)}"
        )

@router.get("/chat/get-user-chat-rooms", response_model=List[ChatRoomResponse], status_code=status.HTTP_200_OK)
async def get_user_chat_rooms(user_id: int):
    """
    Get all user's chat rooms.
    
    - **user_id**: User ID
    """
    try:
        chat_manager = chat_service.ChatManager()
        await chat_manager.initialize()
        result = await chat_manager.get_user_chat_rooms(user_id)
        return result
    except Exception as e:
        logger.exception(f"Failed to get user's chat rooms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user's chat rooms: {str(e)}"
        )
