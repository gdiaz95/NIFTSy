"""Tier-1 quickstart: the generate_synthetic_dataset() one-liner.

Requires a real GEMINI_API_KEY (or GOOGLE_API_KEY) to actually call the model.
Without a key set, this prints a message and exits 0 instead of making a
network call.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv

from niftsy import generate_synthetic_dataset
from niftsy.exceptions import NiftsyError


def main() -> int:
    load_dotenv()
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to run this example. See .env.example.")
        return 0

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
            n_rows=3,
        )
    except NiftsyError as exc:
        print(f"Could not generate: {exc}")
        return 0

    print(result.dataframe.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
