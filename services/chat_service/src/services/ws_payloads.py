"""WebSocket payload helpers (service layer, pure functions, no I/O).

Outbound frames are JSON strings distinguished by ``action``: ``chat_message``,
``llm_reply``, ``llm_error``, ``last_message_updated``. The frontend handlers in
``useChatWebSocket`` must stay in sync with the shapes built here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime


def llm_followup_enabled() -> bool:
    """Return whether LLM follow-ups are enabled.

    Set ``CHAT_LLM_ENABLED=false`` to disable all gateway LLM calls (even when
    ``is_model_prompt`` is true). Defaults to enabled.
    """
    return os.getenv("CHAT_LLM_ENABLED", "true").lower() in ("1", "true", "yes")


def parse_is_model_prompt(raw: object) -> bool:
    """Return True when the client marks the message as an AI prompt.

    Args:
        raw: The ``is_model_prompt`` value from the inbound JSON (bool, string such as
            ``"true"``/``"1"``/``"yes"``, or None).
    """
    if raw is True:
        return True
    if raw is False or raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes")
    return bool(raw)


def coalesce_inbound_message_text(message: dict) -> str:
    """Return the stripped text of an inbound ``send_message`` frame.

    Uses ``content``, falling back to ``message`` (some clients send the text under
    either key). Returns an empty string if neither is set.
    """
    text = (message.get("content") or message.get("message") or "").strip()
    return text


def default_llm_system_prompt() -> str | None:
    """Return the system prompt from ``CHAT_LLM_SYSTEM_PROMPT``, or None if unset/blank
    (None lets rag-service use its own default)."""
    raw = (os.getenv("CHAT_LLM_SYSTEM_PROMPT") or "").strip()
    return raw or None


def build_chat_message_payload(
    *,
    chat_room_id: str,
    user_id: str,
    content: str,
    created_at: datetime,
    updated_at: datetime,
    is_model_prompt: bool = False,
    role: str = "user",
) -> str:
    """Build the ``chat_message`` frame broadcast to a room when a user sends a message.

    Payload: ``action="chat_message"``, ``chat_room_id``, ``user_id`` (str), ``role``,
    ``content``, ``is_model_prompt`` (bool), ``created_at`` and ``updated_at``
    (ISO 8601).

    Returns:
        The JSON string to send over the socket.
    """
    return json.dumps(
        {
            "action": "chat_message",
            "chat_room_id": chat_room_id,
            "user_id": str(user_id),
            "role": role,
            "content": content,
            "is_model_prompt": is_model_prompt,
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
        },
        ensure_ascii=False,
    )


def build_last_message_payload(
    *,
    chat_room_id: str,
    user_id: str,
    message: str,
    updated_at: datetime,
) -> str:
    """Build the ``last_message_updated`` frame (published on Redis for chat lists).

    Payload: ``action="last_message_updated"``, ``chat_room_id``, ``user_id`` (str),
    ``last_message_text``, ``last_message_at`` (ISO 8601, taken from ``updated_at``).

    Returns:
        The JSON string to publish/send.
    """
    return json.dumps(
        {
            "action": "last_message_updated",
            "chat_room_id": chat_room_id,
            "user_id": str(user_id),
            "last_message_text": message,
            "last_message_at": updated_at.isoformat(),
        },
        ensure_ascii=False,
    )


def build_llm_reply_payload(
    *,
    chat_room_id: str,
    content: str,
    created_at: datetime,
) -> str:
    """Build the ``llm_reply`` frame carrying the assistant's answer.

    Payload: ``action="llm_reply"``, ``chat_room_id``, ``user_id="assistant"``,
    ``role="assistant"``, ``content``, ``created_at`` (ISO 8601).

    Returns:
        The JSON string to send over the socket.
    """
    return json.dumps(
        {
            "action": "llm_reply",
            "chat_room_id": chat_room_id,
            "user_id": "assistant",
            "role": "assistant",
            "content": content,
            "created_at": created_at.isoformat(),
        },
        ensure_ascii=False,
    )


def build_llm_error_payload(*, chat_room_id: str, detail: str) -> str:
    """Build the ``llm_error`` frame sent when the assistant could not answer.

    Payload: ``action="llm_error"``, ``chat_room_id``, ``detail`` (user-safe text).

    Returns:
        The JSON string to send over the socket.
    """
    return json.dumps(
        {
            "action": "llm_error",
            "chat_room_id": chat_room_id,
            "detail": detail,
        },
        ensure_ascii=False,
    )
