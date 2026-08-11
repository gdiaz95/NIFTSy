from __future__ import annotations

from typing import Any

from niftsy.exceptions import NiftsyError
from niftsy.llm.base import LLMBackend


def resolve_provider(model: str, provider: str = "auto") -> str:
    if provider and provider != "auto":
        return provider

    model_lower = model.lower()
    if "gemini" in model_lower:
        return "gemini"
    if model_lower.startswith("gpt") or "openai" in model_lower:
        return "openai"
    return "local"


def build_llm_backend(model: str, provider: str = "auto", **kwargs: Any) -> LLMBackend:
    """Build the LLM backend for the resolved provider.

    All three provider SDKs (google-genai, openai, vllm+torch) are base
    dependencies -- provider selection is a runtime/config choice, not an
    install-time one, so no lazy-import guard is needed here.
    """
    resolved = resolve_provider(model, provider)

    if resolved == "gemini":
        from niftsy.llm.gemini import GeminiBackend
        return GeminiBackend(model=model, **kwargs)

    if resolved == "openai":
        from niftsy.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(model=model, **kwargs)

    if resolved == "local":
        from niftsy.llm.local_vllm import LocalVLLMBackend
        return LocalVLLMBackend(model=model, **kwargs)

    raise NiftsyError(f"Unsupported provider '{resolved}'.")
