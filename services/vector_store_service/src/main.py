"""Vector store service entrypoint (config layer): FastAPI app on port 8003.

NOTE: stub / not functional. Only ``GET /health`` and a placeholder
``POST /api/v1/vectors/store`` exist; both return static JSON. No database, embeddings or
vector index are involved, and the fuller router in ``api/endpoints.py`` is not
mounted. There is no gateway entry for this service and it is not in docker compose.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Loads a .env from the current working directory (not from the service folder).
load_dotenv()


def get_cors_origins() -> list[str]:
    """Return the allowed CORS origins from ``CORS_ORIGINS`` (comma-separated).

    Returns:
        Non-empty, stripped origins; defaults to ``["http://localhost:3000"]``.
    """
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Vector Store Service",
    description="Vector database operations service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """GET /health - liveness probe.

    Auth: none. Returns: ``{"status", "service", "version"}`` (always 200).
    """
    return {
        "status": "healthy",
        "service": "vector-store-service",
        "version": "1.0.0"
    }


@app.post("/api/v1/vectors/store")
async def store_vectors():
    """POST /api/v1/vectors/store - placeholder that stores nothing.

    NOTE: stub. Auth: none. Body: none. Returns a static ``{"success": True, "message": ...}``.
    """
    return {
        "success": True,
        "message": "Vector Store service is operational"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
