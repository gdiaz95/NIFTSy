from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from niftsy.tabular.knn import TabularGenerationResult, knn_retrieval_step1
from niftsy.tabular.npgc import NPGC_special

LOGGER = logging.getLogger(__name__)


def run_tabular_generation(
    real_df: pd.DataFrame,
    tabular_cfg: dict,
    text_columns: list[str],
    feature_weights: dict[str, float] | None = None,
    compute_knn: bool = True,
    max_rows: int | None = None,
) -> TabularGenerationResult:
    """Fit NPGC and generate synthetic tabular rows with neighbor indices."""
    excluded = [col for col in text_columns if col in real_df.columns]
    LOGGER.info(f"- Preparing tabular features (excluding free-text columns: {excluded})...")
    tabular_features = real_df.drop(columns=excluded)
    LOGGER.info(f"- Tabular feature matrix shape: {tabular_features.shape}")

    LOGGER.info("- Initializing NPGC synthesizer...")
    synthesizer = NPGC_special(
        enforce_min_max_values=bool(tabular_cfg["enforce_min_max_values"]),
        epsilon=tabular_cfg["epsilon"],
    )
    LOGGER.info("- Fitting NPGC synthesizer on tabular features...")
    synthesizer.fit(tabular_features)

    real_z = synthesizer._model_state["transformed_data"]
    LOGGER.info("- Sampling synthetic tabular rows...")
    n_samples = min(len(tabular_features), max_rows) if max_rows else len(tabular_features)
    synthetic_data, z_correlated = synthesizer.sample(n_samples)
    LOGGER.info(
        f"- Sampled synthetic rows: {synthetic_data.shape[0]} | "
        f"Columns: {synthetic_data.shape[1]}"
    )

    nn_idx: np.ndarray | None = None
    if text_columns and compute_knn:
        LOGGER.info(f"- Running KNN retrieval in latent space with k={int(tabular_cfg['knn_k'])}...")
        nn_idx, _ = knn_retrieval_step1(
            real_z,
            z_correlated,
            k=int(tabular_cfg["knn_k"]),
            feature_weights=feature_weights,
        )
        LOGGER.info("- Tabular generation and KNN retrieval complete.")
    elif text_columns:
        LOGGER.info("- Skipping KNN retrieval; tabular-only generation requested.")
    else:
        LOGGER.info("- No free-text columns configured; skipping KNN retrieval.")

    return TabularGenerationResult(
        synthetic_data=synthetic_data,
        z_correlated=z_correlated,
        real_z=real_z,
        nn_idx=nn_idx,
    )
