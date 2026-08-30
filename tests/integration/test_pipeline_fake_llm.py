import pandas as pd

import niftsy.pipeline as pipeline_module
from niftsy import (
    GenerationConfig,
    LLMConfig,
    SyntheticDataGenerator,
    generate_synthetic_dataset,
)
from tests.integration.conftest import FakeLLMBackend


def _real_df():
    return pd.DataFrame({
        "age": [22, 35, 41, 29, 60, 33, 45, 27, 31, 50],
        "income": [30000, 52000, 61000, 40000, 90000, 48000, 70000, 39000, 45000, 82000],
        "bio": ["works in tech", "teacher", "nurse", "student", "retired",
                "engineer", "artist", "chef", "driver", "manager"],
    })


def _real_df_two_text_columns():
    df = _real_df()
    df["notes"] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    return df


def test_end_to_end_with_fake_backend():
    df = _real_df()
    result = generate_synthetic_dataset(
        df, text_columns=["bio"], model="fake-model", provider="auto",
        n_rows=5, llm=FakeLLMBackend(), seed=42,
    )
    assert len(result.dataframe) == 5
    assert set(result.dataframe.columns) == set(df.columns)
    assert result.dataframe["bio"].str.len().gt(0).all()
    assert result.failed_row_indices == []


def test_k0_ablation_runs_without_neighbors():
    df = _real_df()
    result = generate_synthetic_dataset(
        df, text_columns=["bio"], model="fake-model", n_rows=3,
        llm=FakeLLMBackend(), k_neighbors=0,
    )
    assert len(result.dataframe) == 3


def test_dry_run_makes_no_backend_calls():
    df = _real_df()
    calls = []
    class CountingBackend(FakeLLMBackend):
        def generate_batch(self, prompts, config=None):
            calls.append(len(prompts))
            return super().generate_batch(prompts, config)
    generate_synthetic_dataset(
        df, text_columns=["bio"], model="fake-model", n_rows=5,
        llm=CountingBackend(), dry_run=True,
    )
    assert calls == []  # dry_run must not touch the backend


def test_unknown_text_column_raises_immediately():
    import pytest

    from niftsy.exceptions import NiftsyError
    df = _real_df()
    with pytest.raises(NiftsyError, match="typo_column"):
        generate_synthetic_dataset(df, text_columns=["typo_column"], model="fake-model", n_rows=3, llm=FakeLLMBackend())


def test_generate_synthetic_dataset_does_not_clobber_configured_provider():
    # Regression test: generate_synthetic_dataset() used to unconditionally
    # overwrite config.llm.model/provider with its own hardcoded defaults
    # whenever the caller didn't ALSO pass model=/provider= kwargs -- silently
    # resetting a caller-configured provider (e.g. "local") back to "auto".
    df = _real_df()
    config = GenerationConfig(llm=LLMConfig(model="my-local-model", provider="local"))
    generate_synthetic_dataset(
        df, text_columns=["bio"], n_rows=3, llm=FakeLLMBackend(), config=config,
    )
    assert config.llm.model == "my-local-model"
    assert config.llm.provider == "local"


def test_run_log_includes_config_snapshot_hash_and_duration():
    df = _real_df()
    config = GenerationConfig(llm=LLMConfig(model="fake-model", provider="auto"))
    result = generate_synthetic_dataset(
        df, text_columns=["bio"], n_rows=3, llm=FakeLLMBackend(), config=config,
    )
    assert result.run_log["dry_run"] is False
    assert result.run_log["config"]["llm"]["model"] == "fake-model"
    assert isinstance(result.run_log["config_hash"], str) and len(result.run_log["config_hash"]) == 12
    assert result.run_log["duration_seconds"] >= 0
    assert "h " in result.run_log["duration_human"]

    # Same config -> same fingerprint; a real behavior change -> different one.
    same_config_result = generate_synthetic_dataset(
        df, text_columns=["bio"], n_rows=3, llm=FakeLLMBackend(), config=GenerationConfig(llm=LLMConfig(model="fake-model", provider="auto")),
    )
    assert same_config_result.run_log["config_hash"] == result.run_log["config_hash"]

    different_config_result = generate_synthetic_dataset(
        df, text_columns=["bio"], n_rows=3, llm=FakeLLMBackend(), k_neighbors=1,
        config=GenerationConfig(llm=LLMConfig(model="fake-model", provider="auto")),
    )
    assert different_config_result.run_log["config_hash"] != result.run_log["config_hash"]


def test_dry_run_log_also_includes_config_snapshot_and_hash():
    df = _real_df()
    result = generate_synthetic_dataset(
        df, text_columns=["bio"], model="fake-model", n_rows=3, llm=FakeLLMBackend(), dry_run=True,
    )
    assert result.run_log["dry_run"] is True
    assert "config" in result.run_log
    assert "config_hash" in result.run_log


def test_multi_column_generation_blends_prior_columns_text_similarity(monkeypatch):
    df = _real_df_two_text_columns()
    calls = []
    real_fn = pipeline_module.knn_retrieval_with_text_blocks

    def spy(*args, **kwargs):
        calls.append(kwargs.get("text_blocks"))
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "knn_retrieval_with_text_blocks", spy)

    result = generate_synthetic_dataset(
        df, text_columns=["bio", "notes"], model="fake-model", n_rows=5,
        llm=FakeLLMBackend(), seed=42,
    )
    # Only the second column's stage should trigger the blended KNN call --
    # the first column has no prior text to blend, so it uses the plain
    # tabular-only nn_idx directly (no call at all for stage 1).
    assert len(calls) == 1
    assert len(calls[0]) == 1  # exactly one prior column's block ("bio")
    assert len(result.dataframe) == 5


def test_k0_ablation_skips_text_blending_for_multi_column(monkeypatch):
    df = _real_df_two_text_columns()
    calls = []
    monkeypatch.setattr(
        pipeline_module, "knn_retrieval_with_text_blocks",
        lambda *a, **k: calls.append(1),
    )

    result = generate_synthetic_dataset(
        df, text_columns=["bio", "notes"], model="fake-model", n_rows=3,
        llm=FakeLLMBackend(), k_neighbors=0,
    )
    assert calls == []
    assert len(result.dataframe) == 3


def test_fit_falls_back_to_config_dataset_fields():
    df = _real_df()
    config = GenerationConfig(text_columns=["bio"], target_column="income")
    gen = SyntheticDataGenerator(config)
    gen.fit(df)  # no text_columns/target_column passed -- must come from config
    result = gen.generate(n_rows=4, llm=FakeLLMBackend())
    gen.close()
    assert len(result.dataframe) == 4
    assert result.dataframe["bio"].str.len().gt(0).all()
