import numpy as np
import pandas as pd
import pytest

from niftsy.config import LLMConfig, PromptConfig
from niftsy.text.generation import generate_free_text_column


class _MetaOutputBackend:
    """A backend whose generate_batch always returns text that
    clean_generated_text reduces to an empty string (a truncate-at-phrase
    match at index 0), used to verify such rows are marked failed instead
    of silently written as an empty-string success."""

    def __init__(self):
        self.calls = 0
        self.provider = "local"

    def generate_batch(self, prompts, config=None):
        self.calls += 1
        return ["word count" for _ in prompts]


class _AlwaysFailingBackend:
    """A backend whose generate_batch always raises the same error, used to
    exercise both the 'stop immediately' and 'isolate via per-row fallback'
    paths depending on the error message."""

    def __init__(self, message):
        self.calls = 0
        self.provider = "local"
        self._message = message

    def generate_batch(self, prompts, config=None):
        self.calls += 1
        raise RuntimeError(self._message)


def _small_df():
    return pd.DataFrame({
        "age": [22, 35, 41, 29, 60],
        "bio": ["a", "b", "c", "d", "e"],
    })


def test_account_wide_error_stops_immediately_without_per_row_fallback():
    df = _small_df()
    backend = _AlwaysFailingBackend("unsupported_parameter: top_p is not supported for this model")
    nn_idx = np.zeros((5, 1), dtype=int)

    with pytest.raises(RuntimeError, match="unsupported_parameter"):
        generate_free_text_column(
            backend,
            original_df=df,
            synthetic_df=df,
            nn_idx=nn_idx,
            llm_cfg=LLMConfig(batch_size=5),
            prompt_cfg=PromptConfig(),
            text_column="bio",
            k_neighbors=1,
            show_progress=False,
        )
    # One call for the whole batch, no per-row fallback retries (would be 5
    # more calls -- one per row -- if the systemic check didn't fire).
    assert backend.calls == 1


def test_local_oom_falls_back_to_per_row_isolation_instead_of_crashing():
    # Unlike an account-wide API error, a local GPU OOM might only affect the
    # full batch size -- the per-row fallback (a much smaller retry unit)
    # deserves a chance to succeed instead of aborting the whole run.
    df = _small_df()
    backend = _AlwaysFailingBackend("CUDA error: out of memory")
    nn_idx = np.zeros((5, 1), dtype=int)

    generated, failed_row_indices = generate_free_text_column(
        backend,
        original_df=df,
        synthetic_df=df,
        nn_idx=nn_idx,
        llm_cfg=LLMConfig(batch_size=5, failed_row_retry_passes=0),
        prompt_cfg=PromptConfig(),
        text_column="bio",
        k_neighbors=1,
        show_progress=False,
    )
    # 1 whole-batch attempt + 5 individual per-row attempts (all still fail,
    # since this fake backend always raises) -- but crucially, the function
    # does NOT crash the whole run; it isolates the failures per row.
    assert backend.calls == 1 + 5
    assert sorted(failed_row_indices) == [0, 1, 2, 3, 4]


def test_meta_output_response_is_marked_failed_not_silently_blank():
    df = _small_df()
    backend = _MetaOutputBackend()
    nn_idx = np.zeros((5, 1), dtype=int)

    generated, failed_row_indices = generate_free_text_column(
        backend,
        original_df=df,
        synthetic_df=df,
        nn_idx=nn_idx,
        llm_cfg=LLMConfig(batch_size=5, failed_row_retry_passes=0),
        prompt_cfg=PromptConfig(),
        text_column="bio",
        k_neighbors=1,
        show_progress=False,
    )
    # Every row's response cleans down to "" -- these must be reported as
    # failures, not silently written as empty-string "successes".
    assert sorted(failed_row_indices) == [0, 1, 2, 3, 4]
