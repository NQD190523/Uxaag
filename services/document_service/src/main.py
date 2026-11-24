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
from shared.request_logging import RequestLoggingMiddleware

# Load the service-local .env (if present) BEFORE importing the routers: the
# converter and Markdown store read env vars (e.g. DOCUMENT_STORAGE_ROOT,
# MAX_PDF_UPLOAD_BYTES) at import time.
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

# Logging is configured after dotenv (so LOG_LEVEL/LOG_DIR from .env apply) and before the
# local imports below, whose module-level loggers and singletons start logging at import.
setup_logging("document-service")
logger = get_logger("document-service")

from api.endpoints import router  # noqa: E402  (must run after load_dotenv and setup_logging)


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

# One INFO/WARNING/ERROR line per HTTP request (id, user, status, duration).
app.add_middleware(RequestLoggingMiddleware)

app.include_router(router, prefix="/api/v1")


@app.exception_handler(HTTPException)
async def http_exception_logger(request: Request, exc: HTTPException):
    """Return an ``HTTPException`` as JSON, logging it at DEBUG only.

    Why DEBUG: ``RequestLoggingMiddleware`` already logs every 4xx at WARNING with
    method, path and status, and the raising layer logs the specific reason
    (e.g. ``markdown_store`` path rejects). A third WARNING would duplicate both.
    The detail is a fixed message from our code, never user content.

    Returns:
        JSON response ``{"detail": ...}`` with the original status code.
    """
    logger.debug(
        "Request rejected: status=%s method=%s path=%s detail=%s",
        exc.status_code,
        request.method,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_logger(request: Request, exc: Exception):
    """Return a generic 500 for any unhandled exception without logging it again.

    Why no log: this handler runs outside the request middleware (no request id),
    and ``RequestLoggingMiddleware`` already logs the unhandled error once with its
    traceback. The detail is intentionally generic so internals are not leaked.
    """
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

    logger.info("Starting document service host=0.0.0.0 port=8001")
    # log_config=None keeps uvicorn from replacing the handlers set by setup_logging.
    uvicorn.run(app, host="0.0.0.0", port=8001, log_config=None)
