from __future__ import annotations

import pandas as pd


def _word_count_series(series: pd.Series) -> pd.Series:
    non_null = series.dropna().astype(str).str.strip()
    non_empty = non_null[non_null != ""]
    if non_empty.empty:
        return pd.Series(dtype="int64")
    return non_empty.str.split().str.len().astype("int64")


def detect_free_text_columns(
    df: pd.DataFrame,
    threshold: float = 0.8,
    min_avg_words: float = 3.0,
) -> list[str]:
    """Suggest candidate free-text columns.

    A column is flagged when its unique-value ratio is >= ``threshold`` and
    its average word count is >= ``min_avg_words``. This is an opt-in
    convenience only -- it is never called automatically by the pipeline.
    """
    candidates: list[str] = []
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        unique_ratio = non_null.nunique() / len(non_null)
        if unique_ratio < threshold:
            continue

        word_counts = _word_count_series(series)
        if word_counts.empty or word_counts.mean() < min_avg_words:
            continue

        candidates.append(col)

    return candidates
