"""
RAG Chat Room Web
Main FastAPI application entry point.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add app to Python path
app_path = Path(__file__).parent / "app"
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.logging import setup_logging, enable_console_logging_for_all
from app.core.config import config
from app.api.v1.endpoints import router as api_router

# Setup logging: enable console for all loggers, then service-specific
enable_console_logging_for_all(config.get("logging.level", "INFO"))
logger = setup_logging("rag-chat-room-web", log_level=config.get("logging.level", "INFO"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("🚀 Starting RAG Chat Room Web application...")
    try:
        # Initialize application components
        # TODO: Add initialization logic here
        logger.info("✅ Application components initialized successfully")
        logger.info("🌐 RAG Chat Room Web application is ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application components: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down RAG Chat Room Web application...")

# Create FastAPI application
app = FastAPI(
    title="RAG Chat Room Web",
    description="Collaborative chat rooms with Retrieval-Augmented Generation capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Chat Room Web",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    
    port = config.get_main_port()
    host = config.get_main_host()
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
