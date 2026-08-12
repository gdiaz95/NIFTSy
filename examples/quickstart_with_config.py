"""Tier-2 quickstart: explicit GenerationConfig (loaded from YAML) + fit/generate/close.

Requires a real GEMINI_API_KEY (or GOOGLE_API_KEY) to actually call the model.
Without a key set, this prints a message and exits 0 instead of making a
network call.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from niftsy import GenerationConfig, SyntheticDataGenerator
from niftsy.exceptions import NiftsyError

CONFIG_PATH = Path(__file__).parent / "quickstart_config.yml"


def main() -> int:
    load_dotenv()
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to run this example. See .env.example.")
        return 0

    if not CONFIG_PATH.exists():
        GenerationConfig().to_yaml(CONFIG_PATH)
        print(f"Wrote a default config to {CONFIG_PATH} (edit it to customize, then re-run).")

    config = GenerationConfig.from_yaml(CONFIG_PATH)

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

    gen = SyntheticDataGenerator(config)
    gen.fit(df, text_columns=["bio"])
    try:
        result = gen.generate(n_rows=3, seed=42)
    except NiftsyError as exc:
        print(f"Could not generate: {exc}")
        return 0
    finally:
        gen.close()

    print(result.dataframe.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
