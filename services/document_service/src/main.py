"""Document service entrypoint (config layer): builds the FastAPI app on port 8001.

Mounts the documents router under ``/api/v1``, enables CORS for ``CORS_ORIGINS``, installs
logging-aware exception handlers and exposes ``GET /health``. The service stores
per-room Markdown/PDF files on local disk; it does not use the database.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from shared.logger import get_logger, setup_logging

# Load the service-local .env (if present) BEFORE importing the routers: the
# converter and Markdown store read env vars (e.g. DOCUMENT_STORAGE_ROOT,
# MAX_PDF_UPLOAD_BYTES) at import time.
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

from api.endpoints import router  # noqa: E402  (must run after load_dotenv)


def configure_logging() -> None:
    """Configure the shared ``document-service`` logger (level from ``LOG_LEVEL``)."""
    setup_logging("document-service", os.getenv("LOG_LEVEL", "INFO"))


configure_logging()
logger = get_logger("document-service")


def get_cors_origins() -> list[str]:
    """Return the allowed CORS origins from ``CORS_ORIGINS`` (comma-separated).

    Returns:
        Non-empty, stripped origins; defaults to ``["http://localhost:3000"]``.
    """
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Document Service",
    description="Document extraction and conversion service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.exception_handler(HTTPException)
async def http_exception_logger(request: Request, exc: HTTPException):
    """Log every ``HTTPException`` at WARNING level, then return it as JSON.

    Returns:
        JSON response ``{"detail": ...}`` with the original status code.
    """
    logger.warning(
        "HTTP error %s on %s %s: %s",
        exc.status_code,
        request.method,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_logger(request: Request, exc: Exception):
    """Log any unhandled exception with traceback and return a generic 500.

    The detail is intentionally generic so internals are not leaked to clients.
    """
    logger.exception(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


@app.get("/health")
async def health_check():
    """GET /health - liveness probe.

    Auth: none. Returns: ``{"status", "service", "version"}`` (always 200).
    """
    return {
        "status": "healthy",
        "service": "document-service",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting document service on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
