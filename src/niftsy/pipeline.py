from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from niftsy.config import GenerationConfig
from niftsy.exceptions import NiftsyError
from niftsy.llm.base import LLMBackend
from niftsy.llm.factory import build_llm_backend, resolve_provider
from niftsy.tabular.pipeline_step import run_tabular_generation
from niftsy.text.generation import generate_free_text_column

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    dataframe: pd.DataFrame
    llm_usage: dict
    knn_indices: np.ndarray | None
    failed_row_indices: list[int]
    run_log: dict = field(default_factory=dict)


class SyntheticDataGenerator:
    def __init__(self, config: GenerationConfig | None = None, gpu_index: int | None = None) -> None:
        self.config = config or GenerationConfig()
        self._gpu_index = gpu_index
        self._df: pd.DataFrame | None = None
        self._text_columns: list[str] = []
        self._target_column: str | None = None
        self._feature_weights: dict[str, float] | None = None
        self._owned_backend: LLMBackend | None = None

    def fit(
        self,
        df: pd.DataFrame,
        text_columns: list[str] | None = None,
        target_column: str | None = None,
        feature_weights: dict[str, float] | None = None,
    ) -> "SyntheticDataGenerator":
        text_columns = text_columns or []
        unknown = [col for col in text_columns if col not in df.columns]
        if target_column is not None and target_column not in df.columns:
            unknown.append(target_column)
        if feature_weights:
            unknown += [col for col in feature_weights if col not in df.columns]
        if unknown:
            raise NiftsyError(f"Unknown column(s) in dataframe: {sorted(set(unknown))}")

        self._df = df
        self._text_columns = text_columns
        self._target_column = target_column
        self._feature_weights = feature_weights
        return self

    def generate(
        self,
        n_rows: int | None = None,
        seed: int | None = None,
        llm: LLMBackend | None = None,
        dry_run: bool = False,
    ) -> GenerationResult:
        if self._df is None:
            raise NiftsyError("Call fit(df, ...) before generate().")

        n_rows = n_rows if n_rows is not None else len(self._df)
        k_neighbors = self.config.tabular.k_neighbors

        if dry_run:
            n_calls = n_rows * len(self._text_columns)
            # Rough words -> tokens heuristic (~1.5 tokens/word); this is a cost
            # estimate for the caller, not a real backend call.
            estimated_tokens = int(n_calls * self.config.llm.max_words_generation * 1.5)
            return GenerationResult(
                dataframe=pd.DataFrame(),
                llm_usage={},
                knn_indices=None,
                failed_row_indices=[],
                run_log={
                    "dry_run": True,
                    "estimated_calls": n_calls,
                    "estimated_tokens": estimated_tokens,
                },
            )

        tabular_cfg = {
            "enforce_min_max_values": self.config.tabular.enforce_min_max_values,
            "epsilon": self.config.tabular.epsilon,
            "knn_k": k_neighbors,
        }
        tabular_result = run_tabular_generation(
            self._df,
            tabular_cfg,
            text_columns=self._text_columns,
            feature_weights=self._feature_weights,
            compute_knn=bool(self._text_columns) and k_neighbors > 0,
            n_rows=n_rows,
            seed=seed,
        )

        dataframe = tabular_result.synthetic_data
        failed_row_indices: list[int] = []
        llm_usage: dict = {}

        if self._text_columns:
            backend = llm or self._get_or_build_backend()
            for text_column in self._text_columns:
                dataframe, failed = generate_free_text_column(
                    backend,
                    original_df=self._df,
                    synthetic_df=dataframe,
                    nn_idx=tabular_result.nn_idx,
                    llm_cfg=self.config.llm,
                    prompt_cfg=self.config.prompt,
                    text_column=text_column,
                    k_neighbors=k_neighbors,
                )
                failed_row_indices = sorted(set(failed_row_indices) | set(failed))
            if hasattr(backend, "usage"):
                llm_usage = backend.usage.summary()

        run_log = {
            "dry_run": False,
            "n_rows": len(dataframe),
            "text_columns": list(self._text_columns),
        }

        return GenerationResult(
            dataframe=dataframe,
            llm_usage=llm_usage,
            knn_indices=tabular_result.nn_idx,
            failed_row_indices=failed_row_indices,
            run_log=run_log,
        )

    def _get_or_build_backend(self) -> LLMBackend:
        if self._owned_backend is None:
            kwargs: dict = {}
            resolved = resolve_provider(self.config.llm.model, self.config.llm.provider)
            if resolved == "local":
                kwargs["gpu_memory_utilization"] = self.config.llm.gpu_memory_utilization
                kwargs["max_model_len"] = self.config.llm.max_model_len
                kwargs["enforce_eager"] = self.config.llm.enforce_eager
                if self._gpu_index is not None:
                    kwargs["gpu_index"] = self._gpu_index
            self._owned_backend = build_llm_backend(
                model=self.config.llm.model,
                provider=self.config.llm.provider,
                **kwargs,
            )
        return self._owned_backend

    def close(self) -> None:
        if self._owned_backend is not None and hasattr(self._owned_backend, "cleanup"):
            self._owned_backend.cleanup()
        self._owned_backend = None


def generate_synthetic_dataset(
    df: pd.DataFrame,
    text_columns: list[str] | None = None,
    target_column: str | None = None,
    feature_weights: dict[str, float] | None = None,
    model: str = "gemini-3.1-flash-lite-preview",
    provider: str = "auto",
    n_rows: int | None = None,
    seed: int | None = None,
    llm: LLMBackend | None = None,
    dry_run: bool = False,
    k_neighbors: int | None = None,
    config: GenerationConfig | None = None,
    gpu_index: int | None = None,
) -> GenerationResult:
    """One-shot convenience wrapper around SyntheticDataGenerator for tier-1 users."""
    config = config or GenerationConfig()
    config.llm.model = model
    config.llm.provider = provider
    if k_neighbors is not None:
        config.tabular.k_neighbors = k_neighbors

    gen = SyntheticDataGenerator(config, gpu_index=gpu_index)
    gen.fit(df, text_columns=text_columns, target_column=target_column, feature_weights=feature_weights)
    try:
        return gen.generate(n_rows=n_rows, seed=seed, llm=llm, dry_run=dry_run)
    finally:
        gen.close()
