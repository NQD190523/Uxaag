"""Service layer helper: fire-and-forget asyncio tasks that are not garbage collected.

The event loop keeps only weak references to tasks, so an unreferenced
``asyncio.create_task`` result can be collected mid-run. ``spawn_background_task``
keeps a strong reference until the task finishes and logs any exception.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine

from shared.logger import get_logger

logger = get_logger("chat-service")

# Strong references to running background tasks; removed by the done-callback.
_background_tasks: set[asyncio.Task] = set()


def _on_task_done(task: asyncio.Task) -> None:
    """Drop the finished task from the registry and log its exception, if any."""
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Background task %s failed: %s", task.get_name(), exc, exc_info=exc
        )


def spawn_background_task(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task:
    """Schedule ``coro`` as a background task and keep it referenced until done.

    Args:
        coro: Coroutine to run concurrently with the caller.
        name: Short task name used in failure logs.

    Returns:
        The created task (callers normally ignore it).
    """
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)
    return task
