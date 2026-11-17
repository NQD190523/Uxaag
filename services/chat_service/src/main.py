"""Chat Service entrypoint (app/config layer): FastAPI app on port 8005.

Mounts the REST router and the WebSocket router under ``/api/v1``, configures
CORS, and runs a Redis pub/sub listener for the lifetime of the process.
Exposes ``GET /health``.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from shared.logger import get_logger

# Load environment variables
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.logger import setup_logging, enable_console_logging_for_all
from core.config import config
from api.endpoints import router
from api.websocket import router as websocket_router
from db.redis_client import check_redis_connection, run_room_pubsub_listener

# Initialize logging
setup_logging(
    service_name="chat-service",
    log_level="INFO",
)
logger = get_logger("chat-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: start/stop the Redis pub/sub listener.

    Why: each worker process subscribes to the global chat-list channel and
    pushes updates to its own WebSocket clients, so list updates reach users
    connected to any worker.
    """
    await check_redis_connection()
    stop_event = asyncio.Event()
    # Background task; stop_event lets shutdown end the listener gracefully.
    listener_task = asyncio.create_task(run_room_pubsub_listener(stop_event))
    try:
        yield
    finally:
        stop_event.set()
        await listener_task


# Create FastAPI app
app = FastAPI(
    title="Chat Service",
    description="Chat and conversation management service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allowed origins come from the comma-separated ``CORS_ORIGINS`` env var
# (default: the local Next.js dev server). A wildcard origin is not used because
# credentials are allowed. Methods/headers still honour the YAML ``cors`` section.
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
cors_config = config.get("cors", {})
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_config.get("allow_credentials", True),
    allow_methods=cors_config.get("allow_methods", ["*"]),
    allow_headers=cors_config.get("allow_headers", ["*"]),
)

# Include routers
app.include_router(router, prefix="/api/v1")
app.include_router(websocket_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """GET /health: liveness probe (no auth, no body).

    Returns:
        ``{"status": "healthy", "service": "chat-service", "version": "1.0.0"}``
    """
    return {
        "status": "healthy",
        "service": "chat-service",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        log_level="info",
    )
