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
    resolved = resolve_provider(model, provider)

    if resolved == "gemini":
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:
            raise NiftsyError("Gemini support requires: pip install niftsy[gemini]") from exc
        from niftsy.llm.gemini import GeminiBackend
        return GeminiBackend(model=model, **kwargs)

    if resolved == "openai":
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as exc:
            raise NiftsyError("OpenAI support requires: pip install niftsy[openai]") from exc
        from niftsy.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(model=model, **kwargs)

    if resolved == "local":
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as exc:
            raise NiftsyError("Local vLLM support requires: pip install niftsy[local]") from exc
        from niftsy.llm.local_vllm import LocalVLLMBackend
        return LocalVLLMBackend(model=model, **kwargs)

    raise NiftsyError(f"Unsupported provider '{resolved}'.")
