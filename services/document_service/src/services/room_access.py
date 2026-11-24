"""Room access check (service layer): may this user touch this room's documents?

Fixes bug D1: document-service used to accept any authenticated user for any room.
Membership is owned by chat-service, so it is asked over HTTP
(``clients.chat_membership_client``); this module adds a short positive-only cache.

Why cache: the document UI fires many calls per room (tree, read, autosave), and each
would otherwise cost a gateway + chat + DB round trip.
Why positive-only: a user who is not (yet) a member must see access the moment they are
added, and nobody gains access from a stale "no".
Trade-off: a user removed from a room keeps access for up to the TTL
(``MEMBERSHIP_CACHE_SECONDS``, default 30; ``0`` disables the cache).
"""

from __future__ import annotations

import os
import time

from clients.chat_membership_client import fetch_room_membership

# (user_id, chat_room_id) -> monotonic expiry time. In-process only, so each worker
# keeps its own copy; that is fine for a short TTL.
_member_cache: dict[tuple[str, int], float] = {}
# Hard cap so a long-running process cannot grow the dict without bound.
_MAX_CACHE_ENTRIES = 10_000


def _cache_ttl() -> float:
    """Return the cache TTL in seconds from ``MEMBERSHIP_CACHE_SECONDS`` (default 30)."""
    try:
        return max(0.0, float(os.getenv("MEMBERSHIP_CACHE_SECONDS", "30")))
    except ValueError:
        return 30.0


def clear_membership_cache() -> None:
    """Forget every cached result (used by tests)."""
    _member_cache.clear()


async def is_room_member(*, user_id: str, chat_room_id: int, authorization: str) -> bool:
    """Return True if ``user_id`` is an active participant of ``chat_room_id``.

    Args:
        user_id: Caller id from the gateway's ``x-user-id`` header (cache key only;
            chat-service derives the identity from the token itself).
        chat_room_id: Room being accessed.
        authorization: Caller's ``Authorization`` header, forwarded to the gateway.

    Returns:
        Membership answer, from cache when a fresh positive entry exists.

    Raises:
        MembershipAuthError / MembershipCheckError: Propagated from the client;
            the router maps them to 401 / 503.
    """
    key = (str(user_id), int(chat_room_id))
    now = time.monotonic()
    expiry = _member_cache.get(key)
    if expiry is not None and expiry > now:
        return True

    is_member = await fetch_room_membership(
        chat_room_id=chat_room_id, authorization=authorization
    )
    ttl = _cache_ttl()
    if is_member and ttl > 0:
        if len(_member_cache) >= _MAX_CACHE_ENTRIES:
            # Simple reset instead of LRU: entries live only seconds anyway.
            _member_cache.clear()
        _member_cache[key] = now + ttl
    else:
        _member_cache.pop(key, None)
    return is_member
