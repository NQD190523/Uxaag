"""FastAPI dependencies for document-service routers (router layer).

``require_room_member`` guards every ``/documents/chat-room/{chat_room_id}/...`` endpoint:
the caller must be an active participant of the room (bug D1).
"""

from fastapi import Header, HTTPException, status
from shared.logger import get_logger

from clients.chat_membership_client import MembershipAuthError, MembershipCheckError
from services.room_access import is_room_member

logger = get_logger("document-service")


async def require_room_member(
    chat_room_id: int,
    x_user_id: str = Header(..., alias="x-user-id"),
    authorization: str | None = Header(default=None),
) -> str:
    """Allow the request only if the caller is a member of ``chat_room_id``.

    ``chat_room_id`` is read from the route's path parameter of the same name.

    SECURITY: fails closed. If chat-service cannot answer, the request is refused
    (503) rather than allowed.

    Args:
        chat_room_id: Room from the URL path.
        x_user_id: Caller id injected by the gateway (missing header gives 422).
        authorization: Caller's ``Bearer`` token, passed through by the gateway and
            forwarded to chat-service for the membership lookup.

    Returns:
        The caller's user id, so handlers can keep using ``x_user_id``.

    Raises:
        HTTPException: 401 no/rejected token, 403 not a member, 503 chat-service
            unavailable.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    try:
        allowed = await is_room_member(
            user_id=x_user_id, chat_room_id=chat_room_id, authorization=authorization
        )
    except MembershipAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token rejected",
        )
    except MembershipCheckError:
        # The client already logged the cause (status or error type).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Room membership could not be verified, try again later",
        )
    if not allowed:
        logger.warning("Document access denied room_id=%s", chat_room_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a participant of this chat room",
        )
    return x_user_id
