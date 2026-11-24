"""Gateway route table (router layer).

Defines a catch-all route that picks the upstream service from the first path
segment and hands the request to ``proxy_request``. Authentication is done in
the proxy, not here.
"""

from fastapi import APIRouter, HTTPException, Request
from api.proxy import proxy_request
from core.security import require_auth
from core.config import SERVICES
from shared.logger import get_logger

logger = get_logger("api-gateway")

router = APIRouter()


@router.api_route("/{service:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def gateway_router(request: Request, service: str):
    """
    Generic gateway route.

    - First path segment after leading slash is treated as the service name.
    - Remaining path is forwarded to the configured upstream service.

    Example:
      /auth/register  -> service_name = "auth"

    Auth: decided per service by the ``protected`` flag in ``services.yaml``
    (enforced in ``proxy_request``). Methods: GET, POST, PUT, DELETE, PATCH.

    NOTE: the segment is matched against the keys of ``SERVICES``, not against each
    entry's ``prefix``, so the key and the prefix must agree.

    Args:
        request: Incoming request.
        service: The full path after the leading slash (FastAPI ``path`` parameter).

    Returns:
        The proxied upstream response.

    Raises:
        HTTPException: 404 if the first segment is not a configured service.
    """
    # Normalize and split path
    path_str = service.lstrip("/")
    parts = path_str.split("/", 1)
    service_name = parts[0] if parts else ""

    logger.debug("Routing request: service=%s", service_name)

    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail="Service not found or not configured.")

    return await proxy_request(request, service_name)