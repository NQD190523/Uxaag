"""Gateway client (service layer): call the RAG microservice LLM chat API.

This is a cross-service HTTP call: chat-service -> API gateway -> rag-service.

The gateway validates JWT (`Authorization: Bearer ...`) and forwards to rag-service
`POST /api/v1/rag/llm/chat`. Use the same access token the WebSocket client used.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from shared.log_context import REQUEST_ID_HEADER, get_request_id
from shared.logger import get_logger

# Base URL of the API gateway (not rag-service directly), e.g. http://localhost:8000
# In Docker Compose, set e.g. API_GATEWAY_URL=http://api-gateway:8000
_DEFAULT_GATEWAY = "http://127.0.0.1:8000"
logger = get_logger("chat-service")


async def post_llm_chat_via_api_gateway(
    *,
    user_message: str,
    access_token: str,
    system_prompt: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """POST ``/rag/llm/chat`` on the API gateway (-> rag-service ``/api/v1/rag/llm/chat``).

    The body matches rag-service ``LLMChatRequest``. The gateway base URL comes from
    ``API_GATEWAY_URL`` (default ``http://127.0.0.1:8000``); the timeout from
    ``LLM_GATEWAY_TIMEOUT`` (default 120 s, LLM calls are slow).

    Args:
        user_message: Prompt text for the model.
        access_token: The user's JWT, forwarded as ``Authorization: Bearer`` so the
            gateway can authenticate the request.
        system_prompt: Optional system prompt override.
        timeout_seconds: Per-call timeout override in seconds.

    Returns:
        Parsed JSON response (expects keys ``success`` and ``reply``).

    Raises:
        httpx.HTTPStatusError: On a 4xx/5xx response from the gateway.
        httpx.HTTPError: On connection errors or timeouts.
    """
    base = (os.getenv("API_GATEWAY_URL") or _DEFAULT_GATEWAY).rstrip("/")
    url = f"{base}/rag/llm/chat"
    timeout = timeout_seconds or float(os.getenv("LLM_GATEWAY_TIMEOUT", "120"))
    logger.debug(
        "LLM request started timeout=%s msg_len=%d has_system_prompt=%s",
        timeout,
        len(user_message or ""),
        system_prompt is not None and bool(system_prompt.strip()),
    )

    headers = {
        "Authorization": f"Bearer {access_token.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Forward the current request id so one @model flow can be traced across
        # chat -> gateway -> rag in the logs.
        REQUEST_ID_HEADER: get_request_id(),
    }
    # Only send system_prompt when set, so rag-service applies its own default.
    payload: dict[str, Any] = {"message": user_message}
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as e:
            # Connection error / timeout: no status code to report. The caller
            # logs the traceback, so keep this one short.
            logger.warning(
                "LLM request failed duration_ms=%.0f error=%s",
                (time.perf_counter() - start) * 1000,
                type(e).__name__,
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        if response.status_code >= 400:
            # Never log the body (may echo request data). Only a short ``detail``
            # from a JSON error is useful; otherwise just the body length.
            detail = None
            try:
                parsed = response.json()
                if isinstance(parsed, dict) and isinstance(parsed.get("detail"), str):
                    if len(parsed["detail"]) <= 200:
                        detail = parsed["detail"]
            except ValueError:
                pass
            logger.warning(
                "LLM request finished status=%s duration_ms=%.0f detail=%s body_len=%d",
                response.status_code,
                duration_ms,
                detail,
                len(response.content),
            )
        else:
            logger.info(
                "LLM request finished status=%s duration_ms=%.0f",
                response.status_code,
                duration_ms,
            )
        response.raise_for_status()
        return response.json()
