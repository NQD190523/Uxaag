"""Reverse-proxy logic of the API gateway (router/proxy layer).

Authenticates the caller (for protected services), rewrites the path according to
``services.yaml``, forwards the request with ``httpx`` and returns the upstream
response. Cross-service: calls whichever upstream service ``SERVICES`` maps to.
"""

from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import time

import httpx
from core.config import SERVICES
from core.security import require_auth
from shared.log_context import REQUEST_ID_HEADER, bind_user, get_request_id
from shared.logger import get_logger

logger = get_logger("api-gateway")

# Request headers never forwarded upstream: hop-by-hop headers (httpx recomputes them)
# and the identity headers, which only the gateway may set from the verified JWT.
_REQUEST_DROP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "x-user-id",
        "x-user-email",
        # Re-added once from the gateway's own (validated) request id, never duplicated.
        "x-request-id",
    }
)

# Upstream response headers not copied back to the client. content-length /
# transfer-encoding are recomputed by Starlette; content-encoding is dropped because
# ``aiter_bytes()`` yields already-decoded bytes; content-type is set via ``media_type``.
_RESPONSE_DROP_HEADERS = frozenset(
    {
        "content-length",
        "transfer-encoding",
        "connection",
        "content-encoding",
        "content-type",
        # The gateway's RequestLoggingMiddleware adds X-Request-ID to every response.
        "x-request-id",
    }
)


async def proxy_request(request: Request, service: str) -> Response:
    """
    Proxy the incoming request to the appropriate microservice based on the service name.

    Steps: look up the service, verify the JWT if ``protected`` (default True), strip the
    public prefix, forward headers (dropping any client-sent ``x-user-id`` / ``x-user-email`` and
    injecting them from the verified JWT claims), stream back.

    Args:
        request: Incoming request (method, path, query, headers, cookies, body).
        service: Key of the service in ``services.yaml`` (e.g. ``"chat"``).

    Returns:
        A ``StreamingResponse`` mirroring the upstream status code and body.

    Raises:
        HTTPException: 404 unknown service, 401 from ``require_auth``, 504 upstream
            timeout, 502 upstream connection error.
    """
    # NOTE: `service` is rebound from the name (str) to its config dict.
    service = SERVICES.get(service)
    if not service:
        # FastAPI's HTTPException expects `status_code` and optional `detail`
        raise HTTPException(status_code=404, detail="Service not found")

    # SECURITY: authentication happens here. A missing `protected` key defaults to True
    # (secure by default); `auth` is the only service configured as public.
    payload = None
    if service.get("protected", True):
        payload = await require_auth(request)
        # Tag every later log line of this request with the caller's id.
        bind_user(payload.get("user_id") or payload.get("sub"))
    
    path = request.url.path
    if service.get("strip_prefix", True):
        # e.g. /chat/rooms with prefix /chat -> /rooms, appended to the service base url.
        prefix = service["prefix"]
        path = path.removeprefix(prefix)
    
    url = f"{service['url']}{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    
    # Forward client headers except hop-by-hop ones; httpx recomputes host/content-length.
    # SECURITY: Authorization and Cookie are passed through to the upstream as-is.
    # SECURITY: x-user-id / x-user-email are dropped here (case-insensitive) so a client
    # can never spoof its identity, even on public services where nothing re-adds them.
    headers = {
        key: value for key, value in request.headers.items()
        if key.lower() not in _REQUEST_DROP_HEADERS
    }

    # SECURITY: identity headers are injected ONLY from the verified JWT claims;
    # downstream services trust them as the caller's identity.
    if payload:
        user_id = payload.get("user_id") or payload.get("sub")
        email = payload.get("email")
        if user_id:
            headers["x-user-id"] = str(user_id)
        if email:
            headers["x-user-email"] = str(email)

    # Forward the request id so downstream services log under the same id.
    headers[REQUEST_ID_HEADER] = get_request_id()

    timeout = service.get("timeout", 30)
    prefix_name = service.get("prefix")
    started = time.perf_counter()
    # Log method + upstream path only (no headers, cookies or query string: they can carry tokens).
    logger.debug("Proxying request: method=%s url=%s", request.method, url.split("?", 1)[0])
    async with httpx.AsyncClient(timeout=timeout) as client:
        try :
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=request.stream(),
                # The query string is already part of `url`; do not pass `params` too.
                # The Cookie header is forwarded above, so `cookies=` is not needed either.
                follow_redirects=False,
            )

            proxy_response = StreamingResponse(
                resp.aiter_bytes(),
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )

            # multi_items() keeps repeated headers (e.g. several Set-Cookie) separate;
            # append() adds each one instead of overwriting the previous value.
            for name, value in resp.headers.multi_items():
                if name.lower() in _RESPONSE_DROP_HEADERS:
                    continue
                proxy_response.headers.append(name, value)
            return proxy_response

        except httpx.TimeoutException:
            logger.warning(
                "Upstream timeout: 504 service=%s timeout_s=%s duration_ms=%d",
                prefix_name, timeout, (time.perf_counter() - started) * 1000,
            )
            raise HTTPException(status_code=504, detail="Upstream service timeout")
        except httpx.RequestError as e:
            # SECURITY: keep upstream hostnames/errors out of the client response.
            # Log only the route prefix: ``service`` is the whole config dict here.
            # Only the exception class is logged (its text may contain the upstream URL).
            logger.error(
                "Upstream request failed: 502 service=%s error=%s duration_ms=%d",
                prefix_name, type(e).__name__, (time.perf_counter() - started) * 1000,
            )
            raise HTTPException(status_code=502, detail="Bad Gateway")
