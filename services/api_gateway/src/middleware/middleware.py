"""Correlation-id logging middleware function for the gateway (middleware layer).

NOTE: this function is not registered anywhere in ``main.py`` (it would need
``app.middleware("http")``), so it is currently unused.
"""

from contextvars import ContextVar
import logging
import uuid
from fastapi import Request


# ContextVar allows the correlation_id to be accessed anywhere in this request
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="n/a")

logger = logging.getLogger("middleware")

async def logging_middleware(request: Request, call_next):
    """Tag a request with an ``X-Correlation-ID``, log it, and echo the id back.

    Args:
        request: Incoming HTTP request.
        call_next: Next ASGI handler.

    Returns:
        The downstream response with the ``X-Correlation-ID`` header set.
    """
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    token = correlation_id_ctx.set(correlation_id)

    try:
        # Log the incoming request
        logger.info(
            "Incoming request: %s %s",
            request.method,
            request.url.path,
            extra={"correlation_id": correlation_id},
        )
        response = await call_next(request)

        # Add ID to response headers so the client/caller sees it
        response.headers["X-Correlation-ID"] = correlation_id
        return response
    finally:
        # Always reset the ContextVar, even if the downstream handler raised.
        correlation_id_ctx.reset(token)
