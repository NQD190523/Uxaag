"""Per-request logging context shared by every service (shared library).

Holds the current request id and user id in ``ContextVar``s so any log line written
while handling a request (or a WebSocket message) can include them without passing
them through every function. ``RequestLoggingMiddleware`` sets them for HTTP
requests; WebSocket handlers and background jobs call ``bind_context`` themselves.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
import uuid

# Header used to carry the request id between services (gateway -> service -> service).
REQUEST_ID_HEADER = "X-Request-ID"

# "-" is printed when no request is active (startup, background loops).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
user_id_var: ContextVar[str] = ContextVar("user_id", default="-")


def new_request_id() -> str:
    """Return a short random request id (12 hex chars: unique enough, cheap to print)."""
    return uuid.uuid4().hex[:12]


def get_request_id() -> str:
    """Return the current request id, or ``"-"`` outside a request."""
    return request_id_var.get()


def bind_context(
    request_id: str | None = None, user_id: str | int | None = None
) -> tuple[Token, Token]:
    """Set the request/user id for log lines written from now on in this context.

    Args:
        request_id: Id to use; a new one is generated when ``None`` or empty.
        user_id: Caller's user id, if known.

    Returns:
        Tokens to pass to ``reset_context`` when the unit of work is finished.
    """
    rid_token = request_id_var.set(request_id or new_request_id())
    uid_token = user_id_var.set(str(user_id) if user_id not in (None, "") else "-")
    return rid_token, uid_token


def bind_user(user_id: str | int | None) -> None:
    """Attach the user id to the current context once it is known (e.g. after auth)."""
    if user_id not in (None, ""):
        user_id_var.set(str(user_id))


def reset_context(tokens: tuple[Token, Token]) -> None:
    """Restore the context saved by ``bind_context``."""
    rid_token, uid_token = tokens
    request_id_var.reset(rid_token)
    user_id_var.reset(uid_token)
