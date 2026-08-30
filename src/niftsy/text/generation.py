from __future__ import annotations

import logging
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from niftsy.config import LLMConfig, PromptConfig
from niftsy.llm.base import LLMBackend
from niftsy.text.prompting import (
    build_prompt_from_structured_with_neighbors,
    cap_words,
    clean_generated_text,
    looks_like_meta_output,
    sanitize_neighbor_text,
)

LOGGER = logging.getLogger(__name__)

# Substrings that mean "this will keep failing for every remaining row too,
# no matter how small the retry unit is" -- these are account-wide/request-
# wide states (API quota exhaustion, an unsupported sampling parameter for
# this model), so falling back to a one-row-at-a-time retry loop is pointless;
# stop immediately instead. Checked by message content, not exception type.
#
# Deliberately NOT included: local vLLM "out of memory"/CUDA errors. Those
# are batch-size- or contention-dependent, not account-wide -- a single
# oversized prompt or a transient GPU memory spike from another job sharing
# the GPU can fail a whole batch call while every other prompt in it would
# have succeeded fine. The per-row fallback below already retries at a much
# smaller unit (one prompt at a time), which is the right mitigation for
# exactly that case -- stopping immediately would deny it the chance to work.
_SYSTEMIC_ERROR_MARKERS = (
    "unsupported_parameter",
    "unsupported_value",
    "insufficient_quota",
    "quota",
)


def _is_systemic_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _SYSTEMIC_ERROR_MARKERS)


def generate_free_text_column(
    llm_backend: LLMBackend,
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    nn_idx: np.ndarray | None,
    llm_cfg: LLMConfig,
    prompt_cfg: PromptConfig,
    text_column: str,
    k_neighbors: int,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """Generate one free-text column for every synthetic row.

    Returns the generated dataframe and the list of row indices that never
    received a real response after retries (left as empty strings).
    """
    LOGGER.info(f"- Starting free-text generation stage for column '{text_column}'...")
    generated = synthetic_df.copy()
    generated[text_column] = ""

    total_rows = len(generated)
    is_local = getattr(llm_backend, "provider", None) == "local"
    batch_size = max(1, int(llm_cfg.batch_size))
    sleep_sec = float(llm_cfg.sleep_sec)
    max_words_generation = int(llm_cfg.max_words_generation)
    max_words_reader = int(llm_cfg.max_words_reader)

    # Apply local-vLLM overrides when running on a local model.
    if is_local:
        batch_size = max(1, int(llm_cfg.batch_size_local))
        sleep_sec = float(llm_cfg.sleep_sec_local)

    # Adjust parameters for models that don't support certain values
    temp_value = float(llm_cfg.temperature)
    model_name = llm_cfg.model.lower()
    is_newer_model = any(newer_model in model_name for newer_model in ["gpt-5", "gpt-o1"])

    if is_newer_model:
        temp_value = 1.0

    max_tokens = int(llm_cfg.max_tokens)
    if is_local:
        max_tokens = int(llm_cfg.max_tokens_local)

    generation_config = {
        "temperature": temp_value,
        "max_tokens": max_tokens,
        "stop_sequences": llm_cfg.stop_sequences,
    }

    # Newer models don't support top_p
    if not is_newer_model:
        generation_config["top_p"] = float(llm_cfg.top_p)

    LOGGER.info(f"- Total rows to generate: {total_rows}")
    LOGGER.info(f"- Batch size: {batch_size}" + (" [local override]" if is_local else ""))
    LOGGER.info(f"- Sleep between batches: {sleep_sec}s" + (" [local override]" if is_local else ""))
    LOGGER.info(f"- K neighbors used in prompts: {k_neighbors}")
    config_str = (
        f"temperature={generation_config['temperature']}, "
        f"max_tokens={generation_config['max_tokens']}" + (" [local override]" if is_local else "")
    )
    if "top_p" in generation_config:
        config_str += f", top_p={generation_config['top_p']}"
    LOGGER.info(
        "- Generation config: "
        f"{config_str}, "
        f"max_words_generation={max_words_generation}, "
        f"max_words_reader={max_words_reader}"
    )

    batch_log_every = max(0, int(llm_cfg.batch_log_every))

    parallel_api_calls = bool(llm_cfg.parallel_api_calls)
    api_parallel_shards = max(1, int(llm_cfg.api_parallel_shards))
    rerun_failed_rows = bool(llm_cfg.rerun_failed_rows)
    failed_row_retry_passes = max(0, int(llm_cfg.failed_row_retry_passes))

    def _build_prompts_for_indices(batch_indices: list[int]) -> list[str]:
        prompts: list[str] = []
        for i in batch_indices:
            if nn_idx is not None and k_neighbors > 0:
                neighbor_pos = nn_idx[i, :k_neighbors]
                neighbor_rows = original_df.iloc[neighbor_pos].to_dict("records")
            else:
                neighbor_rows = []
            sanitized_neighbor_rows: list[dict[str, Any]] = []
            for row in neighbor_rows:
                sanitized = sanitize_neighbor_text(str(row.get(text_column, "")))
                if not sanitized:
                    continue
                row_copy = dict(row)
                row_copy[text_column] = sanitized
                sanitized_neighbor_rows.append(row_copy)

            neighbor_lengths = [
                len(re.findall(r"\S+", str(row.get(text_column, "")).strip()))
                for row in sanitized_neighbor_rows
                if str(row.get(text_column, "")).strip()
            ]
            if neighbor_lengths:
                avg_neighbor_words = sum(neighbor_lengths) / len(neighbor_lengths)
                std_neighbor_words = statistics.pstdev(neighbor_lengths)
            else:
                avg_neighbor_words = float(max_words_generation)
                std_neighbor_words = 0.0

            target_min_words = max(1, int(round(avg_neighbor_words - (2 * std_neighbor_words))))
            target_max_words = min(
                max_words_generation,
                int(round(avg_neighbor_words + (2 * std_neighbor_words))),
            )
            target_max_words = max(target_min_words, target_max_words)

            prompt = build_prompt_from_structured_with_neighbors(
                fields=generated.iloc[i].to_dict(),
                prompt_template=prompt_cfg.free_text_prompt,
                max_words_generation=max_words_generation,
                text_column=text_column,
                neighbor_rows=sanitized_neighbor_rows,
                max_neighbors=k_neighbors,
                max_words_reader=max_words_reader,
                target_min_words=target_min_words,
                target_max_words=target_max_words,
            )
            prompts.append(prompt)
        return prompts

    def _process_batch(
        batch_indices: list[int],
    ) -> tuple[dict[int, str], dict[int, str]]:
        prompts = _build_prompts_for_indices(batch_indices)
        row_updates: dict[int, str] = {}
        failed_updates: dict[int, str] = {}
        try:
            responses = llm_backend.generate_batch(prompts, config=generation_config)
        except Exception as exc:
            if _is_systemic_error(exc):
                LOGGER.error(f"STOPPING: Systematic error detected on batch: {exc}")
                raise
            # Otherwise treat as batch failure and continue with single-row fallback
            responses = []

        if responses:
            for i, response_text in zip(batch_indices, responses):
                try:
                    cleaned_response = clean_generated_text(response_text)
                    if not cleaned_response or looks_like_meta_output(cleaned_response):
                        failed_updates[i] = "Response was empty or looked like meta-commentary."
                        continue
                    row_updates[i] = cap_words(
                        cleaned_response,
                        max_words_generation,
                    )
                except Exception as exc:
                    failed_updates[i] = f"{type(exc).__name__}: {exc}"

        if not responses or len(responses) != len(batch_indices):
            pending_indices = [
                i for i in batch_indices if i not in row_updates and i not in failed_updates
            ]
            for i in pending_indices:
                text, error = _process_single_row(i)
                if error is None and text is not None:
                    row_updates[i] = text
                else:
                    failed_updates[i] = error or "No response was returned for this row."
        missing_indices = set(batch_indices) - set(row_updates) - set(failed_updates)
        for i in sorted(missing_indices):
            failed_updates[i] = "No response was returned for this row."
        return row_updates, failed_updates

    def _process_single_row(i: int) -> tuple[str | None, str | None]:
        try:
            prompt = _build_prompts_for_indices([i])[0]
            response_text = llm_backend.generate_batch([prompt], config=generation_config)[0]
            cleaned_response = clean_generated_text(response_text)
            if not cleaned_response or looks_like_meta_output(cleaned_response):
                return None, "Response was empty or looked like meta-commentary."
            return cap_words(cleaned_response, max_words_generation), None
        except Exception as exc:
            if _is_systemic_error(exc):
                LOGGER.error(
                    f"STOPPING: Systematic error detected: {exc}. "
                    f"Row {i} encountered it, and it will affect every remaining row too - "
                    "stopping pipeline immediately."
                )
                raise
            return None, f"{type(exc).__name__}: {exc}"

    def _run_sequential() -> tuple[pd.DataFrame, dict[int, str]]:
        failed_rows: dict[int, str] = {}
        batch_starts = range(0, total_rows, batch_size)
        progress = (
            tqdm(batch_starts, desc=f"Generating {text_column}", dynamic_ncols=True)
            if show_progress
            else batch_starts
        )

        for batch_num, batch_start in enumerate(progress, start=1):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_indices = list(range(batch_start, batch_end))
            row_updates, batch_failures = _process_batch(batch_indices)

            if batch_log_every and batch_num % batch_log_every == 0:
                LOGGER.info(
                    "  • Batch "
                    f"{batch_num}: rows [{batch_start}:{batch_end}] "
                    f"-> received {len(batch_indices)} response(s)."
                )

            for i, text in row_updates.items():
                generated.at[generated.index[i], text_column] = text
            failed_rows.update(batch_failures)

            time.sleep(sleep_sec)

        return generated, failed_rows

    def _run_parallel_api() -> tuple[pd.DataFrame, dict[int, str]]:
        shard_indices = [
            list(range(start, total_rows, api_parallel_shards))
            for start in range(api_parallel_shards)
        ]
        shard_indices = [indices for indices in shard_indices if indices]

        progress_bars = []
        if show_progress:
            for shard_id, indices in enumerate(shard_indices):
                bar = tqdm(
                    total=len(indices),
                    desc=f"Shard {shard_id + 1}/{len(shard_indices)}",
                    position=shard_id,
                    leave=True,
                    dynamic_ncols=True,
                )
                progress_bars.append(bar)

        def _run_shard(
            shard_id: int,
            indices: list[int],
        ) -> tuple[int, dict[int, str], dict[int, str]]:
            shard_updates: dict[int, str] = {}
            shard_failures: dict[int, str] = {}
            for batch_num, offset in enumerate(range(0, len(indices), batch_size), start=1):
                batch_indices = indices[offset : offset + batch_size]
                row_updates, batch_failures = _process_batch(batch_indices)
                shard_updates.update(row_updates)
                shard_failures.update(batch_failures)
                if show_progress:
                    progress_bars[shard_id].update(len(batch_indices))
                if batch_log_every and batch_num % batch_log_every == 0:
                    LOGGER.info(
                        f"  • Shard {shard_id + 1} batch {batch_num}: "
                        f"generated {len(batch_indices)} row(s)."
                    )
                time.sleep(sleep_sec)
            return shard_id, shard_updates, shard_failures

        failed_rows: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(shard_indices)) as executor:
            futures = [
                executor.submit(_run_shard, shard_id, indices)
                for shard_id, indices in enumerate(shard_indices)
            ]
            for future in as_completed(futures):
                _, shard_updates, shard_failures = future.result()
                for i, text in shard_updates.items():
                    generated.at[generated.index[i], text_column] = text
                failed_rows.update(shard_failures)

        for bar in progress_bars:
            bar.close()

        return generated, failed_rows

    def _retry_failed_rows(failed_rows: dict[int, str]) -> dict[int, str]:
        remaining_failures = dict(failed_rows)
        if not remaining_failures or not rerun_failed_rows or failed_row_retry_passes <= 0:
            return remaining_failures

        for retry_pass in range(1, failed_row_retry_passes + 1):
            retry_indices = sorted(remaining_failures)
            if not retry_indices:
                break

            LOGGER.info(
                f"- Retry pass {retry_pass}/{failed_row_retry_passes} "
                f"for {len(retry_indices)} failed row(s)."
            )
            next_failures: dict[int, str] = {}
            progress = (
                tqdm(retry_indices, desc=f"Retrying {text_column}", dynamic_ncols=True)
                if show_progress
                else retry_indices
            )
            for i in progress:
                text, error = _process_single_row(i)
                if error is None and text is not None:
                    generated.at[generated.index[i], text_column] = text
                else:
                    next_failures[i] = error or "No response was returned for this row."
                time.sleep(sleep_sec)

            recovered = len(remaining_failures) - len(next_failures)
            LOGGER.info(
                f"  • Retry pass {retry_pass} recovered {recovered} row(s); "
                f"{len(next_failures)} still failing."
            )
            remaining_failures = next_failures

        return remaining_failures

    if getattr(llm_backend, "provider", None) in {"gemini", "openai"} and parallel_api_calls:
        generated, failed_rows = _run_parallel_api()
    else:
        generated, failed_rows = _run_sequential()

    failed_rows = _retry_failed_rows(failed_rows)
    if failed_rows:
        LOGGER.warning(
            f"Free-text generation left {len(failed_rows)} row(s) empty after retries/reruns."
        )
        sample_indices = list(sorted(failed_rows))[:5]
        for i in sample_indices:
            LOGGER.warning(f"   - row {i}: {failed_rows[i]}")
    else:
        LOGGER.info("- All rows generated successfully after retries/reruns.")

    LOGGER.info("- Free-text generation complete.")

    return generated, sorted(failed_rows)
