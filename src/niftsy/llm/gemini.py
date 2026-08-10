from __future__ import annotations

import logging
import os
import time

from google import genai

from niftsy.exceptions import NiftsyError
from niftsy.llm.base import UsageTracker
from niftsy.llm.http_retry import retry_api_request, stringify_response_value

LOGGER = logging.getLogger(__name__)


def _empty_response_retry_attempts(config):
    return max(1, int(config.get("empty_response_retry_attempts", 3)))


def _empty_response_retry_sleep(config):
    return max(0.0, float(config.get("empty_response_retry_sleep_sec", 1.0)))


class GeminiBackend:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model_name = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise NiftsyError(
                "Gemini provider selected but GEMINI_API_KEY/GOOGLE_API_KEY is missing. "
                "Add it to your .env file."
            )
        self._client = genai.Client(api_key=self.api_key)
        self.usage = UsageTracker()

    def _format_diagnostics(self, response) -> str:
        details = []
        candidates = getattr(response, "candidates", None) or []
        details.append(f"candidate_count={len(candidates)}")

        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = stringify_response_value(
                getattr(prompt_feedback, "block_reason", None)
            )
            if block_reason:
                details.append(f"prompt_block_reason={block_reason}")

        for idx, candidate in enumerate(candidates[:2]):
            finish_reason = stringify_response_value(
                getattr(candidate, "finish_reason", None)
            )
            if finish_reason:
                details.append(f"candidate_{idx}_finish_reason={finish_reason}")

            safety_ratings = getattr(candidate, "safety_ratings", None) or []
            if safety_ratings:
                condensed = []
                for rating in safety_ratings[:3]:
                    category = stringify_response_value(getattr(rating, "category", None))
                    probability = stringify_response_value(getattr(rating, "probability", None))
                    blocked = getattr(rating, "blocked", None)
                    condensed.append(
                        f"{category or 'unknown'}:{probability or 'unknown'}"
                        + (f":blocked={blocked}" if blocked is not None else "")
                    )
                details.append(f"candidate_{idx}_safety_ratings=[{', '.join(condensed)}]")

        return "; ".join(details)

    def generate_batch(self, prompts: list[str], config: dict | None = None) -> list[str]:
        config = config or {}
        return [self._generate_one(prompt, config) for prompt in prompts]

    def _generate_one(self, prompt: str, config: dict) -> str:
        model_name = self.model_name
        if model_name.startswith("models/"):
            model_name = model_name[len("models/"):]

        generation_config = {
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.95),
            "max_output_tokens": config.get("max_tokens", 512),
        }
        thinking_budget = config.get("thinking_budget")
        if thinking_budget is not None:
            generation_config["thinking_config"] = {"thinking_budget": int(thinking_budget)}

        def _request():
            return self._client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config,
            )

        empty_response_attempts = _empty_response_retry_attempts(config)
        empty_response_wait = _empty_response_retry_sleep(config)
        diagnostics = "candidate_count=0"

        for attempt in range(1, empty_response_attempts + 1):
            response = retry_api_request(
                provider_name="Gemini",
                make_request=_request,
                config=config,
                model_message="Couldn't connect to Gemini API. Check model name and API key.",
                network_message="Couldn't connect to Gemini API. Check network access and API key.",
            )

            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                self.usage.record(
                    prompt_tokens=getattr(usage, "prompt_token_count", 0),
                    completion_tokens=getattr(usage, "candidates_token_count", 0),
                    thinking_tokens=getattr(usage, "thoughts_token_count", 0),
                    requests=1,
                )
            else:
                self.usage.record(requests=1)

            text = getattr(response, "text", None)
            if text:
                return text.strip()

            diagnostics = self._format_diagnostics(response)
            if attempt < empty_response_attempts:
                LOGGER.warning(
                    "Gemini returned an empty response body. Retrying in %ss (attempt %s/%s)... %s",
                    empty_response_wait, attempt, empty_response_attempts, diagnostics,
                )
                time.sleep(empty_response_wait)
                continue

        raise NiftsyError(
            f"Gemini returned an empty response after retries. Details: {diagnostics}"
        )
