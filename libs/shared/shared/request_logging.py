"""ASGI middleware that writes one log line per HTTP request (shared library).

Add it in every service's ``main.py``::

    app.add_middleware(RequestLoggingMiddleware)

For each request it:

1. Takes the ``X-Request-ID`` header (set by the gateway) or creates a new id, and the
   gateway-injected ``x-user-id``, and binds both to the logging context.
2. Times the request and logs ``METHOD /path -> status (N ms)`` when it finishes:
   INFO for 2xx/3xx, WARNING for 4xx, ERROR for 5xx. An unhandled exception is
   logged at ERROR with its traceback and then re-raised.
3. Echoes ``X-Request-ID`` in the response so callers can quote it in bug reports.

It is a pure ASGI middleware (not ``BaseHTTPMiddleware``) so it adds almost no
overhead and works with streaming responses. WebSocket and lifespan events pass
through untouched; WebSocket handlers log their own connect/disconnect lines.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Iterable

from shared.log_context import (
    REQUEST_ID_HEADER,
    bind_context,
    get_request_id,
    reset_context,
)

logger = logging.getLogger("request")

# Health checks run every few seconds; logging them would only add noise.
DEFAULT_SKIP_PATHS = ("/health", "/api/v1/health", "/documents/health", "/favicon.ico")


# SECURITY: an incoming request id is reused only if it is short and plain, so a client
# cannot inject fake fields (" | user=1 |") or huge values into log lines.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestLoggingMiddleware:
    """Log each HTTP request with id, user, status and duration."""

    def __init__(
        self,
        app,
        skip_paths: Iterable[str] = DEFAULT_SKIP_PATHS,
        trust_user_header: bool = True,
    ) -> None:
        """Wrap ``app``.

        Args:
            app: The ASGI app to wrap.
            skip_paths: Requests whose path ends with one of these are not logged.
            trust_user_header: Read the caller's user id from ``x-user-id``. True for
                internal services (the gateway sets that header); the gateway itself
                passes False, because there the header comes from the client.
        """
        self.app = app
        self.skip_paths = tuple(skip_paths)
        self.trust_user_header = trust_user_header

    async def __call__(self, scope, receive, send) -> None:
        """Handle one ASGI call (see module docstring)."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        incoming_id = headers.get(REQUEST_ID_HEADER.lower(), "")
        tokens = bind_context(
            incoming_id if _SAFE_REQUEST_ID.match(incoming_id) else None,
            headers.get("x-user-id") if self.trust_user_header else None,
        )
        rid_header = (REQUEST_ID_HEADER.encode("latin-1"), get_request_id().encode("latin-1"))
        path = scope.get("path", "")
        method = scope.get("method", "")
        skip = path.endswith(self.skip_paths)
        start = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message) -> None:
            # Capture the status and echo the request id so callers can quote it.
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                message["headers"] = list(message.get("headers", [])) + [rid_header]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            ms = (time.perf_counter() - start) * 1000
            logger.exception("%s %s -> unhandled error (%.0f ms)", method, path, ms)
            raise
        else:
            if not skip:
                ms = (time.perf_counter() - start) * 1000
                status = status_holder["status"]
                level = (
                    logging.ERROR if status >= 500
                    else logging.WARNING if status >= 400
                    else logging.INFO
                )
                logger.log(level, "%s %s -> %s (%.0f ms)", method, path, status, ms)
        finally:
            reset_context(tokens)


__all__ = ["RequestLoggingMiddleware", "DEFAULT_SKIP_PATHS"]
