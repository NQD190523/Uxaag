"""LangGraph workflow that calls the LLM (service layer).

Builds a one-node graph (START -> call_model -> END) around LangChain's ``ChatOllama``.

NOTE: despite the module name, the client is Ollama-compatible (Gemma via Ollama), not the
OpenAI REST shape. The name is kept to avoid breaking imports.
"""

from __future__ import annotations

import logging
from typing import Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from services.llm_graph.config import (
    LLMConnectionSettings,
    describe_llm_settings,
    load_llm_settings,
)

logger = logging.getLogger(__name__)


# --- State shape: one dictionary that every graph step reads and updates -----------------
class LLMGraphState(TypedDict):
    """State dictionary passed between graph nodes.

    Only ``user_message`` is required to start a run; ``assistant_reply`` is written by the
    model node. LangGraph merges each node's returned dict into this state.
    """

    # What the end user asked; required for the only node we use today.
    user_message: str

    # Optional system rules (tone, safety, “you are a helper for …”).
    system_prompt: NotRequired[str]

    # Filled by the model node after the HTTP call returns.
    assistant_reply: NotRequired[str]


def _build_chat_model(settings: LLMConnectionSettings) -> ChatOllama:
    """Create the ``ChatOllama`` client that performs the HTTP calls to the provider.

    Adds an ``Authorization: Bearer`` header only when an API key is configured (needed for
    cloud/proxied Ollama endpoints, not for a local server).
    """

    # This object knows how to serialize messages and parse the JSON response from the API.
    logger.info("rag llm init | %s", describe_llm_settings(settings))
    # For cloud/proxied endpoints, send Bearer token when configured.
    client_kwargs: dict[str, Any] = {}
    if (settings.api_key or "").strip():
        client_kwargs["headers"] = {
            "Authorization": f"Bearer {settings.api_key.strip()}",
        }

    return ChatOllama(
        model=settings.model,
        temperature=settings.temperature,
        base_url=settings.base_url,  # e.g. http://localhost:11434
        num_predict=settings.num_predict,
        client_kwargs=client_kwargs or None,
    )


def _messages_from_state(state: LLMGraphState) -> list[BaseMessage]:
    """
    Turn our flat state into LangChain message objects the model understands.

    Order matters: system first (if any), then the human turn.
    """

    messages: list[BaseMessage] = []

    # Only attach a system message when the caller provided non-empty instructions.
    system_text = (state.get("system_prompt") or "").strip()
    if system_text:
        messages.append(SystemMessage(content=system_text))

    # The user’s question or instruction in plain language.
    messages.append(HumanMessage(content=state["user_message"]))
    return messages


def _extract_text(message: BaseMessage) -> str:
    """
    Normalize the model output to a single string for our API and logs.

    Most chat models return an AIMessage with `.content` as str or structured parts.
    """

    if isinstance(message, AIMessage):
        content: Any = message.content
        if isinstance(content, str):
            return content
        # Some providers return a list of content parts; join text pieces safely.
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts) if parts else str(content)
    return str(getattr(message, "content", message))


def _make_model_node(llm: ChatOllama):
    """Build the async node function LangGraph runs as the ``call_model`` step.

    A closure over ``llm`` avoids re-reading settings on every request.
    """

    async def call_remote_model(state: LLMGraphState) -> dict[str, str]:
        """
        Single graph step: send messages to the provider and write the reply into state.

        LangGraph merges the returned dict into the previous state before finishing or routing.
        """

        # Convert our simple fields into provider-ready message objects.
        lc_messages = _messages_from_state(state)

        # `ainvoke` runs the HTTP request in async mode so FastAPI stays non-blocking.
        logger.info(
            "rag llm call | has_system_prompt=%s | user_message_len=%s",
            bool((state.get("system_prompt") or "").strip()),
            len(state.get("user_message") or ""),
        )
        try:
            ai_message = await llm.ainvoke(lc_messages)
        except Exception:
            logger.exception("rag llm provider call failed")
            raise

        # Store only the assistant text in state for the next node or the final result.
        return {"assistant_reply": _extract_text(ai_message)}

    return call_remote_model


def build_llm_graph(settings: LLMConnectionSettings | None = None):
    """Build and compile the LLM graph: a linear flow with one node, then END.

    More nodes can be added later (e.g. retrieve documents before calling the model).

    Args:
        settings: Explicit settings; defaults to ``load_llm_settings()``.

    Returns:
        A compiled LangGraph runnable (supports ``ainvoke``).
    """

    # Resolve configuration once per graph build (usually once per process).
    resolved = settings or load_llm_settings()

    # Shared client instance reused across invocations for connection pooling benefits.
    llm = _build_chat_model(resolved)

    # LangGraph needs a typed state schema so it knows how to merge partial updates.
    builder = StateGraph(LLMGraphState)

    # Register our only step under a short name we can reference when drawing or logging.
    builder.add_node("call_model", _make_model_node(llm))

    # Connect the special START marker to our first node (same idea as set_entry_point).
    builder.add_edge(START, "call_model")

    # After the model answers, there is nothing left to do in this minimal graph.
    builder.add_edge("call_model", END)

    # `compile` returns a runnable object with `invoke` / `ainvoke`.
    return builder.compile()


# Module-level singleton so repeated HTTP calls reuse the same compiled graph and client.
_compiled_graph = None


def get_compiled_llm_graph(settings: LLMConnectionSettings | None = None):
    """Return the process-wide compiled graph, building it on first use.

    Args:
        settings: Pass only in tests or to override env; a graph built from explicit settings
            is returned as is and is not cached.

    Returns:
        A compiled LangGraph runnable.

    NOTE: the cache is not guarded by a lock; concurrent first calls may build it twice (harmless).
    """

    global _compiled_graph
    if settings is not None:
        return build_llm_graph(settings)
    if _compiled_graph is None:
        _compiled_graph = build_llm_graph()
    return _compiled_graph


async def run_llm_graph(
    user_message: str,
    *,
    system_prompt: str | None = None,
    settings: LLMConnectionSettings | None = None,
) -> str:
    """Run the graph once and return the assistant text.

    Args:
        user_message: The user's prompt.
        system_prompt: Optional system instructions placed before the user turn.
        settings: Optional explicit settings (bypasses the cached graph).

    Returns:
        The model reply, stripped; empty string if the model returned nothing.

    Raises:
        Exception: Provider/network errors are logged and re-raised; routers map them to 502.
    """

    effective = settings or load_llm_settings()
    logger.info("rag llm run | %s", describe_llm_settings(effective))

    # Reuse one compiled graph per process unless the caller passed explicit settings.
    graph = get_compiled_llm_graph(settings)

    # Initial state must always include the user’s text; system prompt is optional.
    initial: LLMGraphState = {"user_message": user_message}
    if system_prompt:
        initial["system_prompt"] = system_prompt

    # Run the whole graph asynchronously and read the final merged state.
    try:
        final_state = await graph.ainvoke(initial)
    except Exception:
        logger.exception("rag llm graph execution failed")
        raise

    # The node wrote the model output under this key.
    return (final_state.get("assistant_reply") or "").strip()
