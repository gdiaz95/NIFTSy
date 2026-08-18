from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Phrases that only ever show up as *trailing* meta-commentary the model
# appends after the real content — truncate from the phrase onward, keeping
# what came before it.
TRAILING_META_PHRASES = [
    "[end]",
    "to ensure adherence",
    "word count",
    "fits constraints",
    "ignore previous",
    # Leaked LLM instruction phrases found in neighbor data
    "focus on mimicking",
    "to sound human",
    "aim for consistency",
    "to apply, just",
]

# Phrases that show up as a *leading* preamble before the real content (e.g.
# "Here's your requested amenities list: <actual list>", or the model
# echoing the prompt's own "Return ONLY the {text_column} text." instruction
# before complying). Truncating to "everything before the phrase" would
# discard the entire real answer for these, so instead strip the preamble
# itself and keep what follows. This was the root cause of a near-total data
# loss on longer/more complex generations (e.g. ~99.75% of an "amenities"
# column silently blanked to empty strings) where a smaller/faster model
# tended to echo "return only" back before actually answering.
LEADING_META_PHRASES = [
    "return only",
    "here's your requested",
    "here goes:",
]

META_PHRASES = TRAILING_META_PHRASES + LEADING_META_PHRASES


def sanitize_neighbor_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    safe_lines: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(phrase in lower for phrase in META_PHRASES):
            continue
        safe_lines.append(line)

    sanitized = " ".join(safe_lines).strip()
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized


def clean_generated_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    lower = cleaned.lower()
    for phrase in TRAILING_META_PHRASES:
        idx = lower.find(phrase)
        if idx != -1:
            cleaned = cleaned[:idx].strip()
            lower = cleaned.lower()

    for phrase in LEADING_META_PHRASES:
        idx = lower.find(phrase)
        if idx != -1:
            after = cleaned[idx + len(phrase):]
            # A leading preamble is often followed by a few more words and a
            # colon before the real content (e.g. "...requested amenities
            # list: <content>") — skip up through the next colon if one
            # appears shortly after the phrase, not just immediately after it.
            colon_pos = after.find(":")
            if 0 <= colon_pos <= 40:
                after = after[colon_pos + 1:]
            cleaned = after.strip()
            lower = cleaned.lower()

    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def looks_like_meta_output(text: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    return any(phrase in lower for phrase in META_PHRASES)

def cap_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", (text or "").strip())
    return " ".join(words[:max_words]).strip()


def build_prompt_from_structured_with_neighbors(
    fields: dict[str, Any],
    prompt_template: str,
    max_words_generation: int,
    text_column: str,
    neighbor_rows: list[dict[str, Any]] | None = None,
    max_neighbors: int = 3,
    max_words_reader: int = 250,
    target_min_words: int | None = None,
    target_max_words: int | None = None,
) -> str:
    clean: dict[str, Any] = {}
    for key, value in fields.items():
        if key == text_column or value is None:
            continue
        if not isinstance(value, (list, tuple, dict, set)) and pd.isna(value):
            continue
        if isinstance(value, str):
            clean[key] = cap_words(value, max_words_reader)
        else:
            clean[key] = value

    snippets: list[str] = []
    if neighbor_rows:
        for idx, row in enumerate(neighbor_rows[:max_neighbors], start=1):
            row_fields = {
                key: value
                for key, value in row.items()
                if key != text_column and value is not None and pd.notna(value)
            }
            excerpt = cap_words(str(row.get(text_column, "")), max_words_reader)
            if excerpt:
                snippets.append(
                    f"- Neighbor {idx}\n"
                    f"  profile: {row_fields}\n"
                    f"  {text_column}: {excerpt}"
                )

    neighbor_block = "\n".join(snippets) if snippets else "(none)"

    calibrated_target_min_words = max(
        1,
        int(target_min_words if target_min_words is not None else max_words_generation),
    )
    calibrated_target_max_words = max(
        calibrated_target_min_words,
        int(target_max_words if target_max_words is not None else max_words_generation),
    )

    return prompt_template.format(
        max_words_generation=max_words_generation,
        max_words_reader=max_words_reader,
        target_min_words=calibrated_target_min_words,
        target_max_words=calibrated_target_max_words,
        text_column=text_column,
        target_profile=clean,
        neighbor_block=neighbor_block,
    ).strip()
