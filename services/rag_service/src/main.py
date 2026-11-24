"""RAG service entrypoint (config layer): builds the FastAPI app on port 8002.

Mounts two routers: the RAG stub at ``/api/v1/rag`` (no retrieval yet) and the LLM router
at ``/api/v1/rag/llm`` (the working LangGraph + ChatOllama chat endpoint used by
chat-service for ``@model`` messages). The ``/api/v1/rag/llm`` prefix makes the gateway
path ``/rag/llm/chat`` resolve to ``/api/v1/rag/llm/chat``. Exposes ``GET /health``.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.logger import get_logger, setup_logging
from shared.request_logging import RequestLoggingMiddleware

# Keep env loading at process start so downstream modules can read API keys.
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

# Configure logging after dotenv (LOG_LEVEL/LOG_DIR) and before the local imports below.
setup_logging("rag-service")
logger = get_logger("rag-service")

from api.endpoints import router as rag_router  # noqa: E402
from api.llm import router as llm_router  # noqa: E402

app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation service",
    version="1.0.0",
)

# Origins come from CORS_ORIGINS (comma-separated); credentials are allowed, so no wildcard.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One log line per HTTP request (id, user, status, duration).
app.add_middleware(RequestLoggingMiddleware)

# Keep API routes under /api/v1 to match chat/auth service style.
app.include_router(rag_router, prefix="/api/v1")
# llm_router already has prefix /llm; mounting under /api/v1/rag gives /api/v1/rag/llm/chat,
# which is where the gateway entry `rag` (url .../api/v1/rag) sends /rag/llm/chat (bug G3).
app.include_router(llm_router, prefix="/api/v1/rag")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "rag-service",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        # Keep the handlers installed by setup_logging; uvicorn must not replace them.
        log_config=None,
    )
