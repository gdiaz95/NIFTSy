from __future__ import annotations

import threading
from typing import Protocol

from niftsy.exceptions import NiftsyError

__all__ = ["LLMBackend", "UsageTracker", "NiftsyError"]


class LLMBackend(Protocol):
    provider: str

    def generate_batch(self, prompts: list[str], config: dict | None = None) -> list[str]:
        ...


class UsageTracker:
    """Thread-safe accumulator for LLM token/request usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thinking_tokens": 0,
            "requests": 0,
        }

    def record(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_tokens: int = 0,
        requests: int = 1,
    ) -> None:
        with self._lock:
            self._totals["prompt_tokens"] += max(0, int(prompt_tokens or 0))
            self._totals["completion_tokens"] += max(0, int(completion_tokens or 0))
            self._totals["thinking_tokens"] += max(0, int(thinking_tokens or 0))
            self._totals["requests"] += max(0, int(requests or 0))

    def summary(self) -> dict:
        with self._lock:
            totals = dict(self._totals)
        totals["total_tokens"] = (
            totals["prompt_tokens"] + totals["completion_tokens"] + totals["thinking_tokens"]
        )
        return totals
