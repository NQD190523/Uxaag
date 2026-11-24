"""
API Gateway - Routes requests to appropriate microservices
Port: 8000

Layer: application entry point. Creates the FastAPI app, exposes ``/health`` and
mounts the catch-all proxy router (``api/router.py``). The gateway is the auth
boundary: it verifies the JWT for ``protected`` services and injects ``x-user-id``
before forwarding (see ``api/proxy.py``).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from shared.logger import setup_logging
from shared.request_logging import RequestLoggingMiddleware

# Load the gateway .env first so LOG_LEVEL / LOG_DIR are visible to setup_logging
# (core.config loads the same file, but it is imported after logging is configured).
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Logging must be configured BEFORE local imports so their module-level loggers and
# import-time messages use the shared handlers (CODING_CONVENTIONS 4a).
setup_logging("api-gateway")

from api.router import router as gateway_router  # noqa: E402
from core.config import SERVICES  # noqa: E402

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

# Added last so it is the outermost middleware: it assigns the request id and times
# every request, including ones answered by CORS or by the proxy's HTTP errors.
# SECURITY: the gateway is the auth boundary, so a client-sent x-user-id is never
# trusted for logging; the user is bound by proxy.py after the JWT is verified.
app.add_middleware(RequestLoggingMiddleware, trust_user_header=False)



if __name__ == "__main__":
    # log_config=None: keep the shared handlers instead of uvicorn's own logging config.
    uvicorn.run(app, host="127.0.0.1", port=8000, log_config=None)
