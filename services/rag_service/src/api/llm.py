"""LLM chat router (router layer): ``POST /api/v1/rag/llm/chat``.

Thin HTTP wrapper around ``llm_graph.run_llm_graph`` (LangGraph + ChatOllama, Gemma by
default). chat-service calls it through the gateway for ``@model`` messages.

NOTE: main.py mounts this router under ``/api/v1/rag`` so the gateway path
``/rag/llm/chat`` resolves to ``/api/v1/rag/llm/chat`` (bug G3 fixed).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from shared.logger import get_logger

from services.llm_graph.openai_compatible_graph import run_llm_graph

router = APIRouter(prefix="/llm", tags=["llm"])
logger = get_logger("rag-service")


class LLMChatRequest(BaseModel):
    """Request body for ``POST /llm/chat``: user message plus an optional system prompt."""

    message: str = Field(..., min_length=1, description="User message to send to the LLM")
    system_prompt: str | None = Field(
        None,
        description="Optional system prompt (role, tone, constraints)",
    )


class LLMChatResponse(BaseModel):
    """Response body for ``POST /llm/chat``: ``success`` flag and the model's reply text."""

    success: bool
    reply: str


@router.post("/chat", response_model=LLMChatResponse)
async def llm_chat(body: LLMChatRequest):
    """POST /llm/chat (full path ``/api/v1/rag/llm/chat``) - run the LLM graph and return its reply.

    Auth: none checked here (no ``x-user-id`` needed); the gateway protects the prefix.
    Body: ``LLMChatRequest`` (``message`` min length 1, optional ``system_prompt``).
    Returns: ``LLMChatResponse``. Provider settings come from env: ``GEMMA_BASE_URL`` /
    ``OLLAMA_BASE_URL``, ``GEMMA_MODEL``, ``GEMMA_API_KEY``, ``LLM_TEMPERATURE``,
    ``LLM_NUM_PREDICT`` (see ``services/llm_graph/config.py``).
    Errors: 400 (``ValueError`` from the graph), 502 (provider call failed).
    """
    try:
        logger.info(
            "rag llm endpoint hit | message_len=%s | has_system_prompt=%s",
            len(body.message or ""),
            bool((body.system_prompt or "").strip()),
        )
        reply = await run_llm_graph(
            body.message,
            system_prompt=body.system_prompt,
        )
        return LLMChatResponse(success=True, reply=reply)
    except ValueError as e:
        logger.warning("rag llm bad request | %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("rag llm endpoint failed")
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {e!s}",
        ) from e
