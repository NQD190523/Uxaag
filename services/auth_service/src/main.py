"""
Auth Service - Microservice for user authentication and authorization
Port: 8006

Layer: application entry point. Creates the FastAPI app, mounts the ``/auth`` and
``/user`` routers under ``/api/v1`` and exposes ``/health``. Clients normally reach
it through the API gateway (``/auth/*`` public, ``/user/*`` protected).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables FIRST, before any imports that depend on them
# Look for .env in the uxaag/ root: src/main.py -> src -> auth_service -> services -> uxaag.
# (config.py separately reads auth_service/.env through pydantic-settings.)
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.logger import setup_logging, get_logger
from api.endpoints import router
from middleware.middleware import LoggingMiddleware
from api.user import router as user_router

# Initialize logging
setup_logging(
    service_name="auth-service",
    log_level="INFO",
)
logger = get_logger("auth-service")

# Create FastAPI app
app = FastAPI(
    title="Auth Service",
    description="User authentication and authorization service",
    version="1.0.0",
)

# Add CORS middleware
# SECURITY: origins come from the CORS_ORIGINS env var (comma-separated, default is
# the local Next.js dev server). A wildcard must not be combined with
# allow_credentials=True, so list explicit origins per environment.
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(router, prefix="/api/v1")
app.include_router(user_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """GET /health: liveness probe.

    Auth: none (not routed through the gateway's JWT check).

    Returns:
        JSON ``{"status": "healthy", "service": "auth-service", "version": ...}``
        with HTTP 200.
    """
    logger.info("Health check called")
    return {
        "status": "healthy",
        "service": "auth-service",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8006,
        log_level="info",
    )
