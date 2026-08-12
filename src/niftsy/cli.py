from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from dotenv import load_dotenv

from niftsy.config import VALID_PROVIDERS, GenerationConfig, LLMConfig, TabularConfig
from niftsy.exceptions import NiftsyError
from niftsy.pipeline import generate_synthetic_dataset
from niftsy.text.detect import describe_text_columns, detect_free_text_columns


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="niftsy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="Generate a synthetic dataset from a CSV."
    )
    generate.add_argument("input_csv")
    generate.add_argument("-o", "--output", required=True, dest="output_csv")
    generate.add_argument(
        "--text-column", action="append", dest="text_columns", default=[],
        help="Free-text column to generate. Repeatable. Falls back to --config's "
             "text_columns if omitted entirely.",
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

    inspect = subparsers.add_parser(
        "inspect",
        help="Print word-count statistics for text columns. No generation, no LLM calls.",
    )
    inspect.add_argument("input_csv")
    inspect.add_argument(
        "--text-column", action="append", dest="text_columns", default=[], required=True,
        help="Free-text column to describe. Repeatable.",
    )

    setup = subparsers.add_parser(
        "setup",
        help="Interactively build a GenerationConfig YAML for a dataset "
             "(text columns, target column, max words, feature weights).",
    )
    setup.add_argument("input_csv")
    setup.add_argument(
        "-o", "--output", default="niftsy_config.yml", dest="output_yaml",
        help="Path to write the config YAML (default: niftsy_config.yml).",
    )

    return parser


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def _read_csv(path: str):
    try:
        return pd.read_csv(path), None
    except FileNotFoundError:
        return None, _fail(f"input CSV not found: {path}")
    except pd.errors.EmptyDataError:
        return None, _fail(f"input CSV is empty: {path}")
    except pd.errors.ParserError as exc:
        return None, _fail(f"could not parse input CSV {path}: {exc}")


def _run_inspect(args: argparse.Namespace) -> int:
    df, status = _read_csv(args.input_csv)
    if df is None:
        return status

    unknown = [col for col in args.text_columns if col not in df.columns]
    if unknown:
        return _fail(f"unknown column(s) in dataframe: {sorted(set(unknown))}")

    stats = describe_text_columns(df, args.text_columns)
    for col, s in stats.items():
        if s["count"] == 0:
            print(f"{col}: no non-empty rows found.")
            continue
        print(
            f"{col}: count={s['count']}, mean={s['mean']:.1f}, median={s['median']:.1f}, "
            f"p90={s['p90']:.1f}, p95={s['p95']:.1f}, max={s['max']}"
        )
    return 0


def _run_setup(args: argparse.Namespace) -> int:
    df, status = _read_csv(args.input_csv)
    if df is None:
        return status
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {args.input_csv}.\n")

    # 1. Free-text columns.
    detected = detect_free_text_columns(df)
    print("Detected likely free-text columns (unique ratio >= 80%, avg words >= 3):")
    print(f"  {', '.join(detected) if detected else '(none detected)'}")
    print("All columns:")
    for col in df.columns:
        print(f"  - {col}")
    response = input(
        f"Free-text columns, comma-separated [{', '.join(detected)}]: "
    ).strip()
    text_columns = detected if not response else [c.strip() for c in response.split(",") if c.strip()]
    unknown = [c for c in text_columns if c not in df.columns]
    if unknown:
        return _fail(f"unknown free-text column(s): {unknown}")

    # 2. Target column.
    remaining = [c for c in df.columns if c not in text_columns]
    print("\nRemaining columns (candidates for target column):")
    for col in remaining:
        print(f"  - {col}")
    target_response = input("Target column (press Enter to skip): ").strip()
    target_column = target_response or None
    if target_column is not None and target_column not in df.columns:
        return _fail(f"unknown target column: {target_column}")

    # 3. Max words for generation, informed by the same checkup as `inspect`.
    default_max_words = LLMConfig().max_words_generation
    print(f"\nConfigured default max_words_generation: {default_max_words}")
    if text_columns:
        stats = describe_text_columns(df, text_columns)
        for col, s in stats.items():
            if s["count"] == 0:
                print(f"  - {col}: no non-empty rows found; keeping default.")
                continue
            print(
                f"  - {col}: count={s['count']}, mean={s['mean']:.1f}, median={s['median']:.1f}, "
                f"p90={s['p90']:.1f}, p95={s['p95']:.1f}, max={s['max']}"
            )
    max_words_response = input(f"Max words for LLM generation [{default_max_words}]: ").strip()
    if max_words_response:
        try:
            max_words_generation = int(max_words_response)
        except ValueError:
            return _fail(f"invalid max words value: {max_words_response!r}")
        if max_words_generation <= 0:
            return _fail("max_words_generation must be a positive integer.")
    else:
        max_words_generation = default_max_words

    # 4. Feature weights, over the remaining structured columns.
    feature_weights: dict[str, float] = {}
    print(
        "\nFeature weights bias which real rows count as 'nearest neighbors' "
        "during generation. Press Enter to keep the default weight of 1.0."
    )
    apply_weights = input("Set custom feature weights? [y/N]: ").strip().lower()
    if apply_weights in {"y", "yes"}:
        for col in remaining:
            weight_response = input(f"  Weight for '{col}' [1.0]: ").strip()
            if not weight_response:
                continue
            try:
                weight = float(weight_response)
            except ValueError:
                return _fail(f"invalid weight for '{col}': {weight_response!r}")
            if weight != 1.0:
                feature_weights[col] = weight

    default_llm = LLMConfig()
    default_tabular = TabularConfig()

    # 5. Model.
    print(
        "\nModel: which LLM generates the free text. Examples: "
        "gemini-3.1-flash-lite-preview, gpt-4o-mini, Qwen/Qwen2.5-14B-Instruct (local)."
    )
    model_response = input(f"Model [{default_llm.model}]: ").strip()
    model = model_response or default_llm.model

    # 6. Provider.
    print(
        "\nProvider: which backend actually makes the call. 'auto' picks one based "
        f"on the model name; or force one of {sorted(VALID_PROVIDERS)}."
    )
    provider_response = input(f"Provider [{default_llm.provider}]: ").strip()
    provider = provider_response or default_llm.provider
    if provider not in VALID_PROVIDERS:
        return _fail(f"invalid provider: {provider!r}; must be one of {sorted(VALID_PROVIDERS)}")

    # 7. k_neighbors.
    print(
        "\nk_neighbors: how many similar real rows are shown to the LLM as style/"
        "content examples for each synthetic row. Higher = more context and slower; "
        "also a bigger privacy consideration if those real rows are near-unique."
    )
    k_response = input(f"k_neighbors [{default_tabular.k_neighbors}]: ").strip()
    if k_response:
        try:
            k_neighbors = int(k_response)
        except ValueError:
            return _fail(f"invalid k_neighbors: {k_response!r}")
    else:
        k_neighbors = default_tabular.k_neighbors

    # 8. Batch size.
    print(
        "\nBatch size: how many prompts are sent to the LLM per request while "
        "generating free text. Bigger batches finish faster but hit the API/GPU harder."
    )
    batch_response = input(f"Batch size [{default_llm.batch_size}]: ").strip()
    if batch_response:
        try:
            batch_size = int(batch_response)
        except ValueError:
            return _fail(f"invalid batch size: {batch_response!r}")
    else:
        batch_size = default_llm.batch_size

    # 9. Parallelize API calls + shard count.
    print(
        "\nParallelize: split generation across several concurrent shards for "
        "speed. Only applies to gemini/openai (local always runs as one batch)."
    )
    default_parallel_label = "Y/n" if default_llm.parallel_api_calls else "y/N"
    parallel_response = input(f"Parallelize API calls? [{default_parallel_label}]: ").strip().lower()
    if not parallel_response:
        parallel_api_calls = default_llm.parallel_api_calls
    else:
        parallel_api_calls = parallel_response in {"y", "yes"}

    api_parallel_shards = default_llm.api_parallel_shards
    if parallel_api_calls:
        print("How many parallel shards? Each shard generates its own rows independently.")
        shards_response = input(f"Parallel shards [{default_llm.api_parallel_shards}]: ").strip()
        if shards_response:
            try:
                api_parallel_shards = int(shards_response)
            except ValueError:
                return _fail(f"invalid shard count: {shards_response!r}")

    # 10. distance_beta_default.
    print(
        "\ndistance_beta_default: how much weight free-text similarity gets, versus "
        "tabular similarity, when picking a synthetic row's nearest real neighbors. "
        "0 = ignore text similarity entirely; higher = weigh it more heavily."
    )
    beta_response = input(f"distance_beta_default [{default_llm.distance_beta_default}]: ").strip()
    if beta_response:
        try:
            distance_beta_default = float(beta_response)
        except ValueError:
            return _fail(f"invalid distance_beta_default: {beta_response!r}")
    else:
        distance_beta_default = default_llm.distance_beta_default

    # 11. text_vector_hash_dim.
    print(
        "\ntext_vector_hash_dim: size of the hashed vector used internally to "
        "compare free-text columns when finding neighbors. Bigger = finer-grained "
        "text matching, at the cost of more memory."
    )
    hash_response = input(f"text_vector_hash_dim [{default_llm.text_vector_hash_dim}]: ").strip()
    if hash_response:
        try:
            text_vector_hash_dim = int(hash_response)
        except ValueError:
            return _fail(f"invalid text_vector_hash_dim: {hash_response!r}")
    else:
        text_vector_hash_dim = default_llm.text_vector_hash_dim

    config = GenerationConfig(
        text_columns=text_columns,
        target_column=target_column,
        feature_weights=feature_weights,
    )
    config.tabular.k_neighbors = k_neighbors
    config.llm.model = model
    config.llm.provider = provider
    config.llm.max_words_generation = max_words_generation
    config.llm.batch_size = batch_size
    config.llm.parallel_api_calls = parallel_api_calls
    config.llm.api_parallel_shards = api_parallel_shards
    config.llm.distance_beta_default = distance_beta_default
    config.llm.text_vector_hash_dim = text_vector_hash_dim
    config.to_yaml(args.output_yaml)

    print(f"\nWrote config to {args.output_yaml}")
    print("Run generation with:")
    print(f"  niftsy generate {args.input_csv} -o output.csv --config {args.output_yaml}")
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    try:
        config = GenerationConfig.from_yaml(args.config) if args.config else GenerationConfig()
    except FileNotFoundError:
        return _fail(f"config file not found: {args.config}")
    except OSError as exc:
        return _fail(f"could not read config file {args.config}: {exc}")

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
        try:
            with open(args.feature_weights_json) as f:
                feature_weights = json.load(f)
        except FileNotFoundError:
            return _fail(f"feature-weights JSON file not found: {args.feature_weights_json}")
        except json.JSONDecodeError as exc:
            return _fail(f"feature-weights JSON file is not valid JSON: {exc}")

    df, status = _read_csv(args.input_csv)
    if df is None:
        return status

    # An empty list here only ever means "no --text-column flag was passed"
    # (each occurrence appends), so None lets fit() fall back to the config's
    # text_columns instead of silently forcing a tabular-only run.
    text_columns = args.text_columns if args.text_columns else None

    try:
        result = generate_synthetic_dataset(
            df,
            text_columns=text_columns,
            target_column=args.target_column,
            feature_weights=feature_weights,
            n_rows=args.n_rows,
            dry_run=args.dry_run,
            config=config,
            gpu_index=args.gpu_index,
        )
    except NiftsyError as exc:
        return _fail(str(exc))

    if args.run_log:
        with open(args.run_log, "w") as f:
            json.dump(result.run_log, f, indent=2, default=str)

    if args.dry_run:
        print(f"Dry run estimate: {result.run_log}")
        return 0

    result.dataframe.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(result.dataframe)} rows to {args.output_csv}")

    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        return _run_inspect(args)
    if args.command == "setup":
        return _run_setup(args)
    return _run_generate(args)


if __name__ == "__main__":
    sys.exit(main())
