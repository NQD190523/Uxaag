"""Chat membership client (clients layer): ask chat-service whether a user is in a room.

Cross-service HTTP call: document-service -> API gateway -> chat-service
``GET /api/v1/chat/rooms/{chat_room_id}/membership``.

Why through the gateway with the caller's JWT (not straight to chat-service with an
``x-user-id`` header): chat-service trusts ``x-user-id`` blindly, so the only safe
source of that header is the gateway, which sets it from the verified token. This is
the same pattern chat-service uses for its ``@model`` call to rag-service.
"""

from __future__ import annotations

import os
import time

import httpx
from shared.log_context import REQUEST_ID_HEADER, get_request_id
from shared.logger import get_logger

# Gateway base URL. Local runs: http://127.0.0.1:8000; Docker: http://api-gateway:8000.
_DEFAULT_GATEWAY = "http://127.0.0.1:8000"
logger = get_logger("document-service")


class MembershipCheckError(Exception):
    """chat-service could not answer (network error, timeout or 5xx)."""


class MembershipAuthError(Exception):
    """The gateway rejected the forwarded token (401), e.g. it expired mid-request."""


async def fetch_room_membership(*, chat_room_id: int, authorization: str) -> bool:
    """Return True if the token's user is an active participant of ``chat_room_id``.

    The gateway URL comes from ``API_GATEWAY_URL`` and the timeout from
    ``MEMBERSHIP_CHECK_TIMEOUT`` (seconds, default 5).

    Args:
        chat_room_id: Room to check.
        authorization: The caller's full ``Authorization`` header (``Bearer <jwt>``),
            forwarded unchanged. Never logged.

    Returns:
        ``is_member`` from chat-service.

    Raises:
        MembershipAuthError: The gateway returned 401.
        MembershipCheckError: Connection error, timeout, any other non-200 status or
            a malformed body. Callers must fail closed on this.
    """
    base = (os.getenv("API_GATEWAY_URL") or _DEFAULT_GATEWAY).rstrip("/")
    url = f"{base}/chat/rooms/{int(chat_room_id)}/membership"
    try:
        timeout = float(os.getenv("MEMBERSHIP_CHECK_TIMEOUT", "5"))
    except ValueError:
        # A typo in env must not turn every room request into a 500.
        timeout = 5.0
    headers = {
        "Authorization": authorization,
        "Accept": "application/json",
        # Same request id across document -> gateway -> chat for log tracing.
        REQUEST_ID_HEADER: get_request_id(),
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError as e:
        logger.warning(
            "Membership check failed room_id=%s duration_ms=%.0f error=%s",
            chat_room_id,
            (time.perf_counter() - start) * 1000,
            type(e).__name__,
        )
        raise MembershipCheckError(type(e).__name__) from e

    if response.status_code == 401:
        raise MembershipAuthError("gateway rejected token")
    if response.status_code != 200:
        # Status only: the body could echo request data.
        logger.warning(
            "Membership check bad status room_id=%s status=%s",
            chat_room_id,
            response.status_code,
        )
        raise MembershipCheckError(f"status {response.status_code}")

    try:
        return bool(response.json()["is_member"])
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Membership check malformed body room_id=%s", chat_room_id)
        raise MembershipCheckError("malformed body") from e
