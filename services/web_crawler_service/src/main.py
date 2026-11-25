"""Web crawler service entrypoint (config layer): FastAPI app on port 8004.

Mounts the crawl router under ``/api/v1`` (``POST /api/v1/crawl/rooms/{id}/pages``),
enables CORS for ``CORS_ORIGINS`` and exposes ``GET /health``. The service has no
database: it fetches public web pages, converts them to Markdown and saves them into a
room's document workspace through the gateway (document-service).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from shared.logger import get_logger, setup_logging
from shared.request_logging import RequestLoggingMiddleware

# Load the service-local .env (if present) BEFORE importing the routers: core/config.py
# reads the crawl limits and API_GATEWAY_URL at import time.
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

# Configure logging after dotenv (LOG_LEVEL/LOG_DIR) and before any local imports.
setup_logging("web-crawler-service")
logger = get_logger("web-crawler-service")

from api.endpoints import router  # noqa: E402  (must run after load_dotenv and setup_logging)


def get_cors_origins() -> list[str]:
    """Return the allowed CORS origins from ``CORS_ORIGINS`` (comma-separated).

    Returns:
        Non-empty, stripped origins; defaults to ``["http://localhost:3000"]``.
    """
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Web Crawler Service",
    description="Imports web pages into room documents as Markdown",
    version="1.0.0",
)

# One log line per HTTP request (id, user, status, duration).
app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """GET /health - liveness probe.

    Auth: none. Returns: ``{"status", "service", "version"}`` (always 200).
    """
    return {
        "status": "healthy",
        "service": "web-crawler-service",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    # log_config=None keeps uvicorn from replacing the handlers set by setup_logging.
    uvicorn.run(app, host="0.0.0.0", port=8004, log_config=None)
