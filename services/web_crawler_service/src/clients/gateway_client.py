"""Gateway client (clients layer): calls from the crawler to other services.

* ``check_room_membership``: gateway -> chat-service
  ``GET /chat/rooms/{id}/membership``, made before any page is fetched;
* ``save_markdown_file``: gateway -> document-service
  ``POST /documents/chat-room/{id}/markdown-files``.

Why through the gateway with the caller's own ``Authorization`` header: chat- and
document-service trust ``x-user-id`` without checking it, and only the gateway may set it
(from the verified JWT). Forwarding the user's token also means document-service runs its
own membership check on every save, so the crawler cannot write where the user may not.
"""

from __future__ import annotations

import secrets

import httpx
from shared.log_context import REQUEST_ID_HEADER, get_request_id
from shared.logger import get_logger

from core import config

logger = get_logger("web-crawler-service")

# Name attempts before falling back to a random suffix: page.md, page-2.md ... page-5.md.
_NAME_ATTEMPTS = 5


class GatewayAuthError(Exception):
    """The gateway rejected the forwarded token (401)."""


class RoomAccessDenied(Exception):
    """The caller is not a member of the room (403 / ``is_member`` false)."""


class GatewayUnavailable(Exception):
    """The gateway or the service behind it failed (network error, timeout, 5xx)."""


class DocumentSaveError(Exception):
    """document-service refused the file. The message is safe to show to the user."""


def _headers(authorization: str) -> dict[str, str]:
    """Headers for a gateway call: the user's token plus the request id for log tracing."""
    return {
        "Authorization": authorization,
        "Accept": "application/json",
        REQUEST_ID_HEADER: get_request_id(),
    }


async def check_room_membership(
    *, chat_room_id: int, authorization: str, transport: httpx.AsyncBaseTransport | None = None
) -> None:
    """Return normally only if the token's user is an active member of the room.

    Args:
        chat_room_id: Target room.
        authorization: Caller's ``Bearer`` header, forwarded unchanged. Never logged.
        transport: Test hook.

    Raises:
        GatewayAuthError: 401 from the gateway.
        RoomAccessDenied: Not a member.
        GatewayUnavailable: Any other failure. Callers fail closed.
    """
    url = f"{config.API_GATEWAY_URL}/chat/rooms/{int(chat_room_id)}/membership"
    try:
        async with httpx.AsyncClient(
            timeout=config.GATEWAY_TIMEOUT_SECONDS, transport=transport
        ) as client:
            response = await client.get(url, headers=_headers(authorization))
    except httpx.HTTPError as e:
        logger.warning("Membership check failed room_id=%s error=%s", chat_room_id, type(e).__name__)
        raise GatewayUnavailable(type(e).__name__) from e
    if response.status_code == 401:
        raise GatewayAuthError()
    if response.status_code != 200:
        logger.warning("Membership check bad status room_id=%s status=%s", chat_room_id, response.status_code)
        raise GatewayUnavailable(f"status {response.status_code}")
    try:
        is_member = bool(response.json()["is_member"])
    except (ValueError, KeyError, TypeError) as e:
        raise GatewayUnavailable("malformed body") from e
    if not is_member:
        raise RoomAccessDenied()


async def save_markdown_file(
    *,
    chat_room_id: int,
    folder: str,
    stem: str,
    content: str,
    authorization: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Create ``<folder>/<stem>.md`` in the room, picking a free name if it is taken.

    document-service never overwrites (409 when the path exists), so the name is retried
    as ``stem-2.md`` ... ``stem-5.md`` and then ``stem-<random>.md``.

    Args:
        chat_room_id: Target room.
        folder: Folder relative to the room root (created by document-service if missing).
        stem: File name without ``.md``; already slugified.
        content: Markdown text.
        authorization: Caller's ``Bearer`` header. Never logged.
        transport: Test hook.

    Returns:
        The saved file's path relative to the room root.

    Raises:
        GatewayAuthError: 401.
        RoomAccessDenied: 403.
        GatewayUnavailable: Network error, timeout or 5xx.
        DocumentSaveError: Any other refusal (400, or every name taken).
    """
    url = f"{config.API_GATEWAY_URL}/documents/chat-room/{int(chat_room_id)}/markdown-files"
    names = [f"{stem}.md"] + [f"{stem}-{n}.md" for n in range(2, _NAME_ATTEMPTS + 1)]
    names.append(f"{stem}-{secrets.token_hex(3)}.md")
    async with httpx.AsyncClient(
        timeout=config.GATEWAY_TIMEOUT_SECONDS, transport=transport
    ) as client:
        for filename in names:
            body = {"folder": folder, "filename": filename, "content": content}
            try:
                response = await client.post(url, json=body, headers=_headers(authorization))
            except httpx.HTTPError as e:
                logger.warning("Save failed room_id=%s error=%s", chat_room_id, type(e).__name__)
                raise GatewayUnavailable(type(e).__name__) from e
            if response.status_code == 200:
                try:
                    return str(response.json()["relative_path"])
                except (ValueError, KeyError, TypeError):
                    # Saved, but the answer is odd: report the path we asked for.
                    return f"{folder}/{filename}" if folder else filename
            if response.status_code == 409:
                continue
            if response.status_code == 401:
                raise GatewayAuthError()
            if response.status_code == 403:
                raise RoomAccessDenied()
            if response.status_code >= 500:
                logger.warning("Save bad status room_id=%s status=%s", chat_room_id, response.status_code)
                raise GatewayUnavailable(f"status {response.status_code}")
            raise DocumentSaveError(f"The document service refused the file (HTTP {response.status_code})")
    raise DocumentSaveError("No free file name for this page")
