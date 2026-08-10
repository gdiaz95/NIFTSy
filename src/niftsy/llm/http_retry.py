from __future__ import annotations

import logging
import time
import urllib.error

from niftsy.exceptions import NiftsyError

LOGGER = logging.getLogger(__name__)


def stringify_response_value(value):
    if value is None:
        return None
    if hasattr(value, "name"):
        return str(getattr(value, "name"))
    if hasattr(value, "value"):
        raw_value = getattr(value, "value")
        if raw_value is not None:
            return str(raw_value)
    return str(value)


def extract_status_code(exc):
    """Best-effort status extraction for SDK errors (Gemini/OpenAI clients)."""
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def is_retryable_http_error(status_code):
    return status_code in {408, 429, 500, 502, 503, 504}


def safe_read_http_error(exc):
    try:
        body = exc.read().decode("utf-8")
        return body[:500]
    except Exception:
        return str(exc)


def _request_retry_attempts(config):
    return max(1, int(config.get("request_retry_attempts", 5)))


def _request_retry_base_sleep(config):
    return max(0.0, float(config.get("request_retry_base_sleep_sec", 2.0)))


def retry_api_request(
    provider_name,
    make_request,
    config,
    model_message,
    network_message,
):
    max_attempts = _request_retry_attempts(config)
    base_wait_seconds = _request_retry_base_sleep(config)

    for attempt in range(1, max_attempts + 1):
        try:
            return make_request()
        except urllib.error.HTTPError as exc:
            details = safe_read_http_error(exc)
            if is_retryable_http_error(exc.code) and attempt < max_attempts:
                wait_seconds = base_wait_seconds * attempt
                LOGGER.warning(
                    "%s request failed with HTTP %s. Retrying in %ss (attempt %s/%s)...",
                    provider_name, exc.code, wait_seconds, attempt, max_attempts,
                )
                time.sleep(wait_seconds)
                continue
            raise NiftsyError(f"{model_message} Details: {details}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_attempts:
                wait_seconds = base_wait_seconds * attempt
                LOGGER.warning(
                    "%s request failed due to a network error. Retrying in %ss (attempt %s/%s)...",
                    provider_name, wait_seconds, attempt, max_attempts,
                )
                time.sleep(wait_seconds)
                continue
            raise NiftsyError(network_message) from exc
        except Exception as exc:
            status_code = extract_status_code(exc)
            if status_code is not None:
                details = str(exc)
                if is_retryable_http_error(status_code) and attempt < max_attempts:
                    wait_seconds = base_wait_seconds * attempt
                    LOGGER.warning(
                        "%s request failed with HTTP %s. Retrying in %ss (attempt %s/%s)...",
                        provider_name, status_code, wait_seconds, attempt, max_attempts,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise NiftsyError(f"{model_message} Details: {details}") from exc

            if attempt < max_attempts:
                wait_seconds = base_wait_seconds * attempt
                LOGGER.warning(
                    "%s request raised an unexpected error. Retrying in %ss (attempt %s/%s)... %s: %s",
                    provider_name, wait_seconds, attempt, max_attempts, type(exc).__name__, exc,
                )
                time.sleep(wait_seconds)
                continue
            raise NiftsyError(
                f"{model_message} Details: {type(exc).__name__}: {exc}"
            ) from exc
