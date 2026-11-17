"""JWT checks for WebSocket connections (service layer).

The WebSocket does not pass through the gateway, so chat-service validates the
access token itself using the same secret/algorithm as the auth service and gateway
(``JWT_SECRET_KEY``, ``JWT_ALGORITHM`` env vars).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import jwt
from fastapi import WebSocket
from jwt import InvalidTokenError

# Re-export ``InvalidTokenError`` so routers need not import ``jwt`` directly.
__all__ = [
    "JWT_SECRET_KEY",
    "JWT_ALGORITHM",
    "InvalidTokenError",
    "extract_bearer_token",
    "decode_user_access_token",
]

# Read once at import time; if unset, the WS router rejects connections (close 1011).
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


def extract_bearer_token(
    websocket: WebSocket, query_token: Optional[str] = None
) -> Optional[str]:
    """Extract the raw JWT from a WebSocket handshake.

    Prefers the ``?token=`` query param (browsers cannot set headers on WebSockets),
    then falls back to an ``Authorization: Bearer`` header.

    Returns:
        The token string, or None if neither is present.
    """
    if query_token:
        return query_token
    auth_header = websocket.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return None


def decode_user_access_token(raw_token: str) -> tuple[dict[str, Any], Any]:
    """Validate a JWT and return ``(payload, user_id)``.

    ``user_id`` is the ``sub`` claim, falling back to ``user_id``.

    Tokens with a ``type`` claim other than ``"access"`` (e.g. refresh tokens) are
    rejected, matching the gateway. A missing ``type`` claim is tolerated.

    Raises:
        InvalidTokenError: If the signature/expiry is invalid, the ``type`` claim is
            present but not ``"access"``, or no user id claim exists.
        RuntimeError: If ``JWT_SECRET_KEY`` is not configured.
    """
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT secret is not configured")
    payload = jwt.decode(raw_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    # The auth service issues type=access|refresh; only access tokens may open sockets.
    token_type = payload.get("type")
    if token_type is not None and token_type != "access":
        raise InvalidTokenError("Token type is not 'access'")
    user_id = payload.get("sub") or payload.get("user_id")
    if not user_id:
        raise InvalidTokenError("Missing user identity in token")
    return payload, user_id
