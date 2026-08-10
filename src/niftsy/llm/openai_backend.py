from __future__ import annotations

import logging
import os
import time

from openai import OpenAI

from niftsy.exceptions import NiftsyError
from niftsy.llm.base import UsageTracker
from niftsy.llm.http_retry import retry_api_request, stringify_response_value

LOGGER = logging.getLogger(__name__)


def _empty_response_retry_attempts(config):
    return max(1, int(config.get("empty_response_retry_attempts", 3)))


def _empty_response_retry_sleep(config):
    return max(0.0, float(config.get("empty_response_retry_sleep_sec", 1.0)))


def _normalize_message_content(content):
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    text_parts = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                text_parts.append(item.strip())
            continue

        if isinstance(item, dict):
            raw_text = item.get("text")
            if isinstance(raw_text, str) and raw_text.strip():
                text_parts.append(raw_text.strip())
                continue
            if isinstance(raw_text, dict):
                text_value = raw_text.get("value")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())
                    continue
            if item.get("type") in {"text", "output_text"} and item.get("value"):
                text_parts.append(str(item["value"]).strip())
            continue

        raw_text = getattr(item, "text", None)
        if isinstance(raw_text, str) and raw_text.strip():
            text_parts.append(raw_text.strip())
            continue

        if hasattr(raw_text, "value") and getattr(raw_text, "value", None):
            text_parts.append(str(raw_text.value).strip())
            continue

        if hasattr(item, "value") and getattr(item, "value", None):
            text_parts.append(str(item.value).strip())

    return "\n".join(part for part in text_parts if part).strip()


def _extract_message_text(message):
    if message is None:
        return ""
    if isinstance(message, dict):
        return _normalize_message_content(message.get("content"))
    return _normalize_message_content(getattr(message, "content", None))


class OpenAIBackend:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model_name = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise NiftsyError(
                "OpenAI provider selected but OPENAI_API_KEY is missing. "
                "Add it to your .env file."
            )
        self._client = OpenAI(api_key=self.api_key)
        self.usage = UsageTracker()

    def _format_diagnostics(self, response) -> str:
        details = []
        choices = getattr(response, "choices", None) or []
        details.append(f"choice_count={len(choices)}")

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if prompt_tokens is not None:
                details.append(f"prompt_tokens={prompt_tokens}")
            if completion_tokens is not None:
                details.append(f"completion_tokens={completion_tokens}")

        for idx, choice in enumerate(choices[:2]):
            finish_reason = stringify_response_value(getattr(choice, "finish_reason", None))
            if finish_reason:
                details.append(f"choice_{idx}_finish_reason={finish_reason}")

            message = getattr(choice, "message", None)
            refusal = getattr(message, "refusal", None) if message is not None else None
            if refusal:
                details.append(f"choice_{idx}_refusal=present")

            text = _extract_message_text(message)
            details.append(f"choice_{idx}_has_text={bool(text)}")

        return "; ".join(details)

    def _build_payload(self, prompt: str, config: dict, model_name: str) -> dict:
        model_name_lower = model_name.lower()
        is_newer_model = any(
            newer_model in model_name_lower for newer_model in ["gpt-5", "gpt-o1"]
        )
        uses_max_completion_tokens = is_newer_model or "gpt-4o-mini" in model_name_lower

        temp_value = config.get("temperature", 0.7)
        if is_newer_model:
            temp_value = 1.0

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp_value,
            ("max_completion_tokens" if uses_max_completion_tokens else "max_tokens"): config.get(
                "max_tokens", 512
            ),
        }

        if not is_newer_model:
            payload["top_p"] = config.get("top_p", 0.95)

        return payload

    def generate_batch(self, prompts: list[str], config: dict | None = None) -> list[str]:
        config = config or {}
        return [self._generate_one(prompt, config) for prompt in prompts]

    def _generate_one(self, prompt: str, config: dict) -> str:
        model_name = self.model_name
        if model_name.lower().startswith("openai/"):
            model_name = model_name.split("/", 1)[1]

        payload = self._build_payload(prompt, config, model_name)

        def _request():
            return self._client.chat.completions.create(**payload)

        empty_response_attempts = _empty_response_retry_attempts(config)
        empty_response_wait = _empty_response_retry_sleep(config)
        diagnostics = "choice_count=0"

        for attempt in range(1, empty_response_attempts + 1):
            response = retry_api_request(
                provider_name="OpenAI",
                make_request=_request,
                config=config,
                model_message="Couldn't connect to OpenAI API. Check model name and API key.",
                network_message="Couldn't connect to OpenAI API. Check network access and API key.",
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                self.usage.record(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    requests=1,
                )
            else:
                self.usage.record(requests=1)

            choices = getattr(response, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                text = _extract_message_text(message)
                if text:
                    return text

            diagnostics = self._format_diagnostics(response)
            if attempt < empty_response_attempts:
                LOGGER.warning(
                    "OpenAI returned an empty response body. Retrying in %ss (attempt %s/%s)... %s",
                    empty_response_wait, attempt, empty_response_attempts, diagnostics,
                )
                time.sleep(empty_response_wait)
                continue

        raise NiftsyError(
            f"OpenAI returned an empty response after retries. Details: {diagnostics}"
        )
