"""Request logging middleware for the auth service (middleware layer).

Propagates an ``X-Correlation-ID`` (from the caller or generated) through a
``ContextVar`` and logs every incoming request.
"""

from contextvars import ContextVar
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from shared.logger import get_logger

# ContextVar allows the correlation_id to be accessed anywhere in this request
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="n/a")

SERVICE_NAME = "auth-service"
logger = get_logger(SERVICE_NAME)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log incoming requests with a correlation ID."""

    async def dispatch(self, request: Request, call_next):
        """Tag the request with a correlation id, log it, and echo the id back.

        Args:
            request: Incoming HTTP request.
            call_next: Next ASGI handler in the chain.

        Returns:
            The downstream response with an ``X-Correlation-ID`` header added.
        """
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        token = correlation_id_ctx.set(correlation_id)
        
        # Log the incoming request
        logger.info(f"Incoming request: {request.method} {request.url.path}", 
                extra={"correlation_id": correlation_id, "service_name": SERVICE_NAME})
        response = await call_next(request)
        
        # Add ID to response headers so the client/caller sees it
        response.headers["X-Correlation-ID"] = correlation_id
        
        # Clean up ContextVar (note: not in a finally block, so it is skipped if call_next raises)
        correlation_id_ctx.reset(token)
        return response