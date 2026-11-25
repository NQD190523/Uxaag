"""Redis integration for chat-service (infrastructure layer).

Design:
- Full chat messages are broadcast only to the specific room in-memory.
- Lightweight room-list updates are published to one global Redis channel.
- Each worker subscribes to that global channel and delivers each update to the
  local sockets of that room's members only (``services/room_list_fanout.py``);
  the update carries message text, so it must not reach non-members.
"""

from __future__ import annotations

import asyncio
from typing import Final

import redis.asyncio as redis

from core.config import config
from services.room_list_fanout import fan_out_room_list_update
from shared.logger import get_logger

logger = get_logger("chat-service")

CHAT_LIST_UPDATES_CHANNEL: Final[str] = "chat_list_updates"

_redis_url = config.get_redis_url()

redis_client: redis.Redis = redis.from_url(
    _redis_url,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True,
)

# Second client so pub/sub does not block or conflict with publish on the same connection.
redis_subscriber: redis.Redis = redis.from_url(
    _redis_url,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True,
)


async def get_redis():
    """Return the shared publisher/command Redis client (e.g. as a dependency)."""
    return redis_client


async def check_redis_connection() -> bool:
    """Ping Redis and log explicit connectivity status.

    Returns:
        True if PING succeeded, False otherwise. Never raises: the service keeps
        running without Redis (cross-worker fan-out is then unavailable).
    """
    try:
        pong = await redis_client.ping()
        if pong:
            logger.info("Redis connected")
            return True
        logger.warning("Redis ping returned empty response")
        return False
    except Exception:
        logger.exception("Redis connection failed")
        return False


async def run_room_pubsub_listener(stop_event: asyncio.Event) -> None:
    """Listen to the global chat-list channel and fan out to local WebSockets.

    Runs until ``stop_event`` is set (started from the app lifespan). Each
    received string is forwarded as-is (it is a ``last_message_updated``
    payload built by ``ws_payloads.build_last_message_payload``).

    Args:
        stop_event: Set on shutdown to end the listener and clean up.
    """
    pubsub = redis_subscriber.pubsub()
    try:
        await pubsub.subscribe(CHAT_LIST_UPDATES_CHANNEL)
    except Exception:
        logger.exception(
            "Redis pub/sub unavailable, only same-worker WebSocket fan-out works"
        )
        # Degraded mode: keep this task alive until shutdown so the lifespan
        # ``await listener_task`` returns cleanly.
        await stop_event.wait()
        try:
            await pubsub.close()
        except Exception:
            pass
        return

    logger.info("Redis pub/sub listener started channel=%s", CHAT_LIST_UPDATES_CHANNEL)

    # ``pubsub.listen()`` blocks forever, so it runs in its own task; the outer
    # coroutine waits on ``stop_event`` and cancels it on shutdown.
    async def consume() -> None:
        # Outer loop = reconnect: if the connection drops, listen() raises; we log a
        # WARNING, wait, resubscribe, and carry on until shutdown.
        while not stop_event.is_set():
            try:
                async for msg in pubsub.listen():
                    if stop_event.is_set():
                        return
                    # Skip subscribe/unsubscribe confirmations; only real messages matter.
                    if msg.get("type") != "message":
                        continue
                    data = msg.get("data")
                    if not isinstance(data, str):
                        continue
                    try:
                        # Only members of the updated room get it (in any room they
                        # have open), so their chat list refreshes without leaking text.
                        await fan_out_room_list_update(data)
                    except Exception:
                        # One bad message must not stop the listener; per-message
                        # success is not logged (high volume).
                        logger.exception("Redis fan-out broadcast failed")
                # listen() ended without raising (e.g. subscription closed). Without
                # a pause the outer loop would spin hot, so back off before retrying.
                if not stop_event.is_set():
                    logger.warning("Redis pub/sub listener ended, retrying in 2s")
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "Redis pub/sub connection lost, reconnecting in 2s error=%s",
                    type(e).__name__,
                )
                await asyncio.sleep(2)
                try:
                    await pubsub.subscribe(CHAT_LIST_UPDATES_CHANNEL)
                    logger.info("Redis pub/sub reconnected")
                except Exception as sub_err:
                    logger.warning(
                        "Redis pub/sub resubscribe failed error=%s", type(sub_err).__name__
                    )

    task = asyncio.create_task(consume())
    try:
        await stop_event.wait()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        try:
            await pubsub.unsubscribe(CHAT_LIST_UPDATES_CHANNEL)
            await pubsub.close()
        except Exception as e:
            logger.warning("Redis pub/sub cleanup failed error=%s", type(e).__name__)
        logger.info("Redis pub/sub listener stopped")
