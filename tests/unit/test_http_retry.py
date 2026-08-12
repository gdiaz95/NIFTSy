import pytest

from niftsy.exceptions import NiftsyError
from niftsy.llm.http_retry import is_permanent_error, retry_api_request


def test_is_permanent_error_detects_known_markers():
    assert is_permanent_error("insufficient_quota")
    assert is_permanent_error("Error: credit_balance_exhausted, please add funds")
    assert not is_permanent_error("temporary server hiccup")


class _FakeQuotaError(Exception):
    status_code = 429

    def __str__(self):
        return "insufficient_quota: you have no credits remaining"


def test_retry_api_request_does_not_retry_permanent_errors():
    calls = []

    def make_request():
        calls.append(1)
        raise _FakeQuotaError()

    with pytest.raises(NiftsyError, match="insufficient_quota"):
        retry_api_request(
            provider_name="TestProvider",
            make_request=make_request,
            config={"request_retry_attempts": 5, "request_retry_base_sleep_sec": 0.0},
            model_message="model error",
            network_message="network error",
        )
    assert len(calls) == 1  # no retries for a permanent error


class _FakeTransientError(Exception):
    """A genuine rate-limit hiccup: same HTTP status as the quota case (429),
    but no permanent-error marker in the message -- this one should recover
    on retry, unlike _FakeQuotaError above."""

    status_code = 429

    def __str__(self):
        return "rate limited, please slow down and try again shortly"


def test_retry_api_request_retries_transient_errors_and_succeeds():
    calls = []

    def make_request():
        calls.append(1)
        if len(calls) < 2:
            raise _FakeTransientError()
        return "success on retry"

    result = retry_api_request(
        provider_name="TestProvider",
        make_request=make_request,
        config={"request_retry_attempts": 5, "request_retry_base_sleep_sec": 0.0},
        model_message="model error",
        network_message="network error",
    )
    assert result == "success on retry"
    assert len(calls) == 2  # failed once, retried once, succeeded


def test_retry_api_request_gives_up_after_max_attempts_on_persistent_transient_error():
    calls = []

    def make_request():
        calls.append(1)
        raise _FakeTransientError()  # never succeeds

    with pytest.raises(NiftsyError, match="model error"):
        retry_api_request(
            provider_name="TestProvider",
            make_request=make_request,
            config={"request_retry_attempts": 3, "request_retry_base_sleep_sec": 0.0},
            model_message="model error",
            network_message="network error",
        )
    assert len(calls) == 3  # respects request_retry_attempts, then gives up
