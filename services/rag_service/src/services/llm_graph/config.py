"""LLM connection settings (config layer): reads env vars into a frozen dataclass.

Tells LangChain's ``ChatOllama`` which Ollama-compatible endpoint (local or cloud) and
model to use. Secrets are only ever logged in masked form.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConnectionSettings:
    """Immutable snapshot of the LLM config so it cannot change mid-request.

    Attributes:
        base_url: Ollama-compatible API root, without trailing slash.
        model: Model id served by that endpoint (e.g. ``gemma3:27b``).
        temperature: Sampling temperature (0 is most deterministic).
        num_predict: Maximum number of tokens to generate.
        api_key: Optional Bearer token for cloud/proxied endpoints.
    """

    base_url: str
    model: str
    temperature: float
    num_predict: int
    api_key: str | None


def _mask_secret(secret: str | None) -> str:
    """Return a log-safe representation of a secret (first/last 4 chars and length, or a placeholder)."""
    raw = (secret or "").strip()
    if not raw:
        return "<missing>"
    if len(raw) <= 8:
        return f"<present:{len(raw)}>"
    return f"{raw[:4]}...{raw[-4:]} (len={len(raw)})"


def load_llm_settings() -> LLMConnectionSettings:
    """Build settings from the process environment (often filled from a ``.env`` file).

    Env keys: ``OLLAMA_BASE_URL`` / ``GEMMA_BASE_URL``, ``GEMMA_MODEL`` / ``OLLAMA_MODEL``,
    ``GEMMA_API_KEY``, ``LLM_TEMPERATURE`` (default 0.7), ``LLM_NUM_PREDICT`` (default 2048).
    Unparseable numeric values silently fall back to the defaults.

    Returns:
        A new ``LLMConnectionSettings``.
    """

    # ChatOllama talks to an Ollama-compatible API (local or remote).
    # Default local endpoint is Ollama's standard API.
    base_url = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("GEMMA_BASE_URL")
        or "http://localhost:11434"
    )

    # Model id served by Ollama, e.g. gemma3:27b, gemma3:12b, gemma3:4b.
    model = os.getenv("GEMMA_MODEL") or os.getenv("OLLAMA_MODEL") or "gemma3:27b"

    # Optional token for cloud/proxied Ollama-compatible endpoints. First non-empty of
    # OLLAMA_API_KEY, GEMMA_API_KEY, LLM_API_KEY wins (bug R1: it used to be overwritten).
    api_key = (
        os.getenv("OLLAMA_API_KEY")
        or os.getenv("GEMMA_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    
    # How random answers are; 0 is more deterministic, higher is more creative.
    raw_temp = os.getenv("LLM_TEMPERATURE", "0.7")
    try:
        temperature = float(raw_temp)
    except ValueError:
        temperature = 0.7

    # Maximum generation tokens.
    raw_num_predict = os.getenv("LLM_NUM_PREDICT", "2048")
    try:
        num_predict = int(raw_num_predict)
    except ValueError:
        num_predict = 2048

    return LLMConnectionSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        api_key=(api_key.strip() if isinstance(api_key, str) else api_key),
    )


def describe_llm_settings(settings: LLMConnectionSettings) -> dict[str, str]:
    """Return a log-safe dict describing the settings (the API key is masked).

    Returns:
        String-valued dict with base_url, model, temperature, num_predict and masked api_key.
    """
    return {
        "base_url": settings.base_url,
        "model": settings.model,
        "temperature": str(settings.temperature),
        "num_predict": str(settings.num_predict),
        "api_key": _mask_secret(settings.api_key),
    }
