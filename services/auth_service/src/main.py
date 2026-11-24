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
# Load the service's own .env (src/main.py -> src -> auth_service), the same file
# config.py reads via pydantic-settings. Why not uxaag/.env: os.environ beats
# pydantic's env_file, so a shared root file silently overrode auth's settings
# (e.g. a stale DATABASE_URL); each service is deployed with its own env.
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.logger import setup_logging
from shared.request_logging import RequestLoggingMiddleware

# Logging must be configured BEFORE local imports so their module-level loggers and
# import-time messages use the shared handlers. Level comes from LOG_LEVEL.
setup_logging("auth-service")

from api.endpoints import router  # noqa: E402
from api.user import router as user_router  # noqa: E402

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

# Added after CORS so it is the outermost middleware: it assigns/propagates
# X-Request-ID and logs method, path, status and duration of every request.
app.add_middleware(RequestLoggingMiddleware)

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
        # keep the shared handlers instead of uvicorn's own logging config
        log_config=None,
    )
