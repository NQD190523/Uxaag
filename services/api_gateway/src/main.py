"""
API Gateway - Routes requests to appropriate microservices
Port: 8000

Layer: application entry point. Creates the FastAPI app, exposes ``/health`` and
mounts the catch-all proxy router (``api/router.py``). The gateway is the auth
boundary: it verifies the JWT for ``protected`` services and injects ``x-user-id``
before forwarding (see ``api/proxy.py``).
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router as gateway_router
import uvicorn
from core.config import SERVICES

app = FastAPI(
    title="API Gateway",
    description="Central API Gateway routing to microservices",
    version="1.0.0"
)
@app.get("/health")
async def health():
    """GET /health: gateway liveness plus the list of configured service names.

    Auth: none. Note it does not probe the upstream services.

    Returns:
        ``{"status": "ok", "services": [<keys of services.yaml>]}`` with HTTP 200.
    """
    return {"status": "ok", "services": list(SERVICES.keys())}

# Catch-all proxy route; /health above is registered first so it wins.
app.include_router(gateway_router)

# SECURITY: origins come from CORS_ORIGINS (comma-separated); credentials are allowed,
# so a wildcard must not be used here.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials= True,
    allow_methods= ["*"],
    allow_headers= ["*"],
)



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
