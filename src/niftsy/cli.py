from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from dotenv import load_dotenv

from niftsy.config import GenerationConfig
from niftsy.pipeline import generate_synthetic_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="niftsy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="Generate a synthetic dataset from a CSV."
    )
    generate.add_argument("input_csv")
    generate.add_argument("-o", "--output", required=True, dest="output_csv")
    generate.add_argument(
        "--text-column", action="append", dest="text_columns", default=[], required=True,
        help="Free-text column to generate. Repeatable.",
    )
    generate.add_argument("--target-column", default=None)
    generate.add_argument("--config", default=None, help="Path to a GenerationConfig YAML file.")
    generate.add_argument("--model", default=None)
    generate.add_argument("--provider", default=None, choices=["auto", "gemini", "openai", "local"])
    generate.add_argument("--k-neighbors", type=int, default=None)
    generate.add_argument("--epsilon", type=float, default=None)
    generate.add_argument("--max-words", type=int, default=None)
    generate.add_argument("--n-rows", type=int, default=None)
    generate.add_argument("--feature-weights-json", default=None)
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--gpu-index", type=int, default=None, help="Local-only.")
    generate.add_argument("--gpu-memory-utilization", type=float, default=None, help="Local-only.")
    generate.add_argument("--run-log", default=None, help="Optional path to write a JSON run log.")

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = GenerationConfig.from_yaml(args.config) if args.config else GenerationConfig()

    if args.model is not None:
        config.llm.model = args.model
    if args.provider is not None:
        config.llm.provider = args.provider
    if args.k_neighbors is not None:
        config.tabular.k_neighbors = args.k_neighbors
    if args.epsilon is not None:
        config.tabular.epsilon = args.epsilon
    if args.max_words is not None:
        config.llm.max_words_generation = args.max_words
    if args.gpu_memory_utilization is not None:
        config.llm.gpu_memory_utilization = args.gpu_memory_utilization

    feature_weights = None
    if args.feature_weights_json:
        with open(args.feature_weights_json) as f:
            feature_weights = json.load(f)

    df = pd.read_csv(args.input_csv)

    result = generate_synthetic_dataset(
        df,
        text_columns=args.text_columns,
        target_column=args.target_column,
        feature_weights=feature_weights,
        n_rows=args.n_rows,
        dry_run=args.dry_run,
        config=config,
        gpu_index=args.gpu_index,
    )

    if args.dry_run:
        print(f"Dry run estimate: {result.run_log}")
        return 0

    result.dataframe.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(result.dataframe)} rows to {args.output_csv}")

    if args.run_log:
        with open(args.run_log, "w") as f:
            json.dump(result.run_log, f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
