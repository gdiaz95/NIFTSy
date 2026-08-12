"""Local vLLM quickstart: generate using a local Hugging Face model, no API key needed.

Requires a CUDA GPU (vllm + torch are already part of the base install). This is
the heaviest, slowest path (vLLM loads model weights on first call, which can take
a minute or two) and is not exercised in automated tests -- run it manually on a
machine with a GPU. "No GPU detected" surfaces as a NiftsyError with an actionable
message, so this prints and exits 0 rather than crashing when run somewhere
without a GPU.
"""
from __future__ import annotations

import sys

import pandas as pd
from dotenv import load_dotenv

from niftsy import generate_synthetic_dataset
from niftsy.exceptions import NiftsyError


def main() -> int:
    load_dotenv()  # picks up HUGGINGFACE_HUB_TOKEN for gated models, if set
    df = pd.DataFrame({
        "age": [25, 40, 33, 52, 29],
        "income": [45000, 82000, 61000, 95000, 38000],
        "bio": [
            "recent graduate working in retail",
            "senior manager at a logistics company",
            "freelance graphic designer",
            "small business owner in construction",
            "part-time barista studying part time",
        ],
    })

    try:
        result = generate_synthetic_dataset(
            df,
            text_columns=["bio"],
            model="Qwen/Qwen2.5-14B-Instruct",
            provider="local",
            n_rows=3,
        )
    except NiftsyError as exc:
        print(f"Skipping: {exc}")
        return 0

    print(result.dataframe.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
