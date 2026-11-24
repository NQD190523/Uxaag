"""Chat Service entrypoint (app/config layer): FastAPI app on port 8005.

Mounts the REST router and the WebSocket router under ``/api/v1``, configures
CORS, and runs a Redis pub/sub listener for the lifetime of the process.
Exposes ``GET /health``.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared.logger import get_logger, setup_logging
from shared.request_logging import RequestLoggingMiddleware

# Load environment variables (LOG_LEVEL etc. must be in os.environ before setup).
load_dotenv()

# Configure logging before any local import so module-level log calls and import-time
# errors in the modules below already use the shared handlers.
setup_logging("chat-service")
logger = get_logger("chat-service")

from core.config import config
from api.endpoints import router
from api.websocket import router as websocket_router
from db.redis_client import check_redis_connection, run_room_pubsub_listener


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: start/stop the Redis pub/sub listener.

    Why: each worker process subscribes to the global chat-list channel and
    pushes updates to its own WebSocket clients, so list updates reach users
    connected to any worker.
    """
    logger.info("Chat service starting")
    await check_redis_connection()
    stop_event = asyncio.Event()
    # Background task; stop_event lets shutdown end the listener gracefully.
    listener_task = asyncio.create_task(run_room_pubsub_listener(stop_event))
    try:
        yield
    finally:
        stop_event.set()
        await listener_task
        logger.info("Chat service stopped")


# Create FastAPI app
app = FastAPI(
    title="Chat Service",
    description="Chat and conversation management service",
    version="1.0.0",
    lifespan=lifespan,
)

# One log line per HTTP request (method, path, status, duration) with a request id.
# Added before CORS so CORS ends up outermost and its preflight replies still work.
app.add_middleware(RequestLoggingMiddleware)

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
        # log_config=None: keep the handlers installed by setup_logging; uvicorn
        # would otherwise replace them with its own.
        log_config=None,
    )
