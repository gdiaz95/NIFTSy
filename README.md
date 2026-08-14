# NIFTSy

Generate a synthetic version of a tabular dataset — synthetic structured
columns (age, income, etc.) plus LLM-generated free-text columns (bios, job
descriptions, notes...) that stay consistent with each synthetic row's
profile.

- Structured columns come from a differentially-private Gaussian-copula
  synthesizer (no LLM involved).
- Free-text columns are written by an LLM (Gemini, OpenAI, or a local
  Hugging Face model via vLLM), prompted with real neighbor rows as style
  examples.

---

## Install

```bash
uv sync
```

Requires Python 3.10–3.13. That's it — Gemini, OpenAI, and local-vLLM
support are all installed by default (no extras to remember); which one
actually runs is a config choice, not an install choice.

## Quickstart

**1. Add your API key.** Copy `.env.example` to `.env` and fill in whichever
provider you'll use:

```bash
cp .env.example .env
# edit .env: GEMINI_API_KEY=... (or OPENAI_API_KEY=..., or nothing if using a local model)
```

**2. Build a config interactively.** Point `niftsy setup` at your CSV — it
asks a handful of plain-English questions (which columns are free text,
which model, how many neighbor rows to show it, etc.), showing sensible
defaults at every step, and writes one YAML config file:

```bash
niftsy setup data/my_dataset.csv
```

If you don't pass a path, it asks for a directory (`./data` by default) and
lets you pick a CSV from it. When you're not sure what to answer, just press
Enter — every question has a reasonable default.

**3. Before you generate — check your columns and your cost.** Two quick,
free checks, strongly recommended before any real run:

- **Not sure which columns are free text, or what `--max-words` should be?**

  ```bash
  niftsy inspect data/my_dataset.csv
  ```

  With no flags, it auto-detects likely free-text columns (the same
  heuristic `setup` uses) and prints word-count stats (mean, median,
  p90/p95, max) for each — no LLM calls, no cost — to help you pick sensible
  values before spending on a real run. Pass `--text-column bio --text-column
  notes` explicitly if you want stats for specific columns instead.

- **Always check the cost before committing.** Add `--dry-run` to `generate`
  — it estimates the number of LLM calls and rough token count without
  touching any backend, and still writes the log file so you have a record:

  ```bash
  niftsy generate data/my_dataset.csv --config niftsy_config.yml --dry-run
  ```

**4. Generate.**

```bash
niftsy generate data/my_dataset.csv --config niftsy_config.yml
```

That's the whole workflow. You'll get two files next to your input, named
after it automatically:

- `my_dataset_synthetic_2026-08-14.csv` — the synthetic data.
- `my_dataset_synthetic_2026-08-14.json` — a log of what happened: token
  usage, how long it took, and every parameter used for the run (so you can
  always answer "what settings produced this file?").

**Prefer Python?** Skip the CLI entirely:

```python
import pandas as pd
from niftsy import generate_synthetic_dataset

df = pd.read_csv("data/my_dataset.csv")
result = generate_synthetic_dataset(df, text_columns=["bio"], target_column="income")
result.dataframe.to_csv("synthetic.csv", index=False)
```

---

# Appendix: technical reference

## CLI commands

### `niftsy setup [INPUT_CSV] [-o CONFIG.yml]`

Interactive wizard that writes a single `GenerationConfig` YAML. Walks
through, in order: free-text columns (with auto-detected suggestions),
target column, max words (with a word-count checkup), feature weights,
model, provider, k-neighbors, batch size, API parallelization, and two
advanced knobs (`distance_beta_default`, `text_vector_hash_dim`). Every
question accepts Enter-for-default.

If `INPUT_CSV` is omitted, the first question asks for a data directory
(default `./data`) and lets you pick a CSV from it, or type a path directly.

- `-o, --output` — where to write the config YAML. Default: `niftsy_config.yml`.

### `niftsy generate INPUT_CSV [options]`

| Flag | Meaning | Default |
|---|---|---|
| `-o, --output` | Output CSV path | `<input_dir>/<input_name>_synthetic_<date>.csv` |
| `--text-column` (repeatable) | Free-text column to generate | falls back to `--config`'s `text_columns` |
| `--target-column` | Target/label column, used as context | falls back to config |
| `--config` | Path to a `GenerationConfig` YAML (e.g. from `setup`) | none — built-in defaults |
| `--model` | LLM model name | overrides config; config default is `gemini-3.1-flash-lite-preview` |
| `--provider` | `auto` / `gemini` / `openai` / `local` | `auto` (picks from the model name) |
| `--k-neighbors` | How many real rows shown as style examples per synthetic row | `5` |
| `--epsilon` | Differential-privacy noise for the tabular synthesizer | `1.0` |
| `--max-words` | Max words per generated text field | `80` |
| `--n-rows` | How many synthetic rows to produce (can exceed the real dataset's size) | same as input row count |
| `--feature-weights-json` | Path to a JSON `{"column": weight}` map | none |
| `--dry-run` | Estimate cost, make no LLM calls | off |
| `--gpu-index` | Force a specific GPU (local provider only) | auto-selects a free one |
| `--gpu-memory-utilization` | Fraction of GPU memory vLLM may use (local only) | `0.8` |
| `--run-log` | Path to the JSON log | `<output path>` with `.json` instead of `.csv`; `dry-run_` prefixed if `--dry-run` |

YAML config and flags compose: `--config` loads a full config first, then
any flag you pass overrides just that field. A run log is **always**
written (see [Run log format](#run-log-format)), not only when `--run-log`
is given explicitly.

### `niftsy inspect INPUT_CSV [--text-column COL ...]`

Prints `count`/`mean`/`median`/`p90`/`p95`/`max` word-count stats per column.
No generation, no LLM calls, no cost.

- `--text-column` (repeatable) — which column(s) to describe. **Optional.**
  If omitted entirely, runs the same auto-detection heuristic `setup` uses
  (`detect_free_text_columns`: object/string columns with a unique-value
  ratio ≥ 80% and an average word count ≥ 3) and describes whatever it
  finds, printing the detected list first. If detection finds nothing,
  prints `(none detected)` and exits — no error.

## `GenerationConfig` reference

One YAML file, three sections plus dataset-specific top-level fields.
Produced by `niftsy setup`, or hand-written / loaded via
`GenerationConfig.from_yaml(path)` / `.to_yaml(path)` in Python.

```yaml
text_columns: [bio, job_description]   # which columns are free text
target_column: income                  # optional; shown as context to the LLM
feature_weights: {age: 2.0}            # optional; biases which real rows count as "nearest neighbors"

tabular:
  k_neighbors: 5                # real rows shown as style examples per synthetic row
  epsilon: 1.0                  # differential-privacy noise (lower = more private, less faithful)
  enforce_min_max_values: true  # clip synthetic numeric values to the real data's observed range

llm:
  model: gemini-3.1-flash-lite-preview
  provider: auto                # auto | gemini | openai | local
  gpu_memory_utilization: 0.8   # local-only
  max_model_len: 4096           # local-only
  enforce_eager: false          # local-only; true = skip CUDA-graph capture (slower, less VRAM)
  max_words_generation: 80      # cap on generated text length
  max_words_reader: 250         # cap on how much of a neighbor row's text gets shown as context
  temperature: 0.8
  top_p: 0.95
  max_tokens: 2048              # per-call token ceiling (API providers)
  stop_sequences: [...]         # local-provider stop strings
  sleep_sec: 1.0                # delay between API batches (rate-limit friendliness)
  batch_size: 8                 # prompts per LLM request (API providers)
  max_tokens_local: 320         # token ceiling override for the local provider
  sleep_sec_local: 0.0
  batch_size_local: 64
  batch_log_every: 0            # log a progress line every N batches (0 = off)
  parallel_api_calls: true      # shard rows across concurrent API calls (gemini/openai only)
  api_parallel_shards: 64
  rerun_failed_rows: true       # retry rows that failed generation
  failed_row_retry_passes: 1
  text_vector_hash_dim: 2000    # size of the hashed text vector used for neighbor matching
  distance_alpha: 1.0           # weight of tabular similarity when finding neighbors
  distance_beta_default: 0.01  # weight of text similarity when finding neighbors
  distance_beta_by_column: {}   # per-column override of distance_beta_default

prompt:
  free_text_prompt: "..."       # the template sent to the LLM; edit with care
```

**Privacy note:** `epsilon` and `k_neighbors` are the two levers that trade
fidelity for privacy. Lower `epsilon` = more DP noise = less faithful but
more private. Smaller `k_neighbors` = each synthetic row leans on fewer real
rows for style, reducing (but not eliminating) the chance that generated
text echoes an identifiable real row's phrasing. Niftsy doesn't ship a
membership-inference or privacy-auditing suite — if you need one, that
belongs in a separate evaluation step outside this package.

**Any column you don't list in `text_columns` gets treated as an ordinary
categorical field** — meaning its synthetic value is resampled from the
real dataset's existing unique values, not paraphrased. If a column holds
free text you forgot to declare, its real values can appear verbatim in
the synthetic output. Always list every free-text column.

## Python API

Two tiers:

```python
# Tier 1 — one-liner
from niftsy import generate_synthetic_dataset

result = generate_synthetic_dataset(
    df,
    text_columns=["job_description"],
    target_column="income",
    model="gemini-3.1-flash-lite-preview",  # optional; this is already the default
    provider="auto",
    n_rows=5000,       # can exceed len(df)
    seed=42,           # reproducible tabular sampling (LLM text itself isn't deterministic)
    dry_run=False,
)
result.dataframe.to_csv("synthetic.csv", index=False)
print(result.llm_usage)          # {"prompt_tokens": ..., "completion_tokens": ..., ...}
print(result.failed_row_indices) # rows that never got a real response after retries
print(result.run_log)            # config snapshot, config_hash, duration, n_rows, ...

# Tier 2 — power user: fit once, generate many times, inject your own backend for testing
from niftsy import SyntheticDataGenerator, GenerationConfig

config = GenerationConfig.from_yaml("niftsy_config.yml")
gen = SyntheticDataGenerator(config)
gen.fit(df)   # text_columns/target_column/feature_weights fall back to the config if omitted here
result = gen.generate(n_rows=5000, seed=42)
gen.close()   # releases the LLM backend (matters for local vLLM's GPU memory)
```

`LLMBackend` is a `Protocol` (`generate_batch(prompts, config) -> list[str]`),
injectable via the `llm=` parameter on both `generate_synthetic_dataset` and
`SyntheticDataGenerator.generate`. This is what lets the test suite mock a
backend without ever importing `vllm`/`google-genai`/`openai`.

Raised errors are always `niftsy.NiftsyError` with an actionable message —
missing API key, no GPU detected, gated/nonexistent Hugging Face model,
unknown column name, etc. Never a raw traceback from a third-party library.

## Providers

| Provider | Needs | Notes |
|---|---|---|
| `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env` | Default provider; picked automatically for any model name containing "gemini". |
| `openai` | `OPENAI_API_KEY` in `.env` | Picked automatically for model names starting with "gpt" or containing "openai". |
| `local` | A CUDA GPU; `HUGGINGFACE_HUB_TOKEN` only for license-gated models | Runs via vLLM. First use of a new model downloads it to the standard Hugging Face cache (`~/.cache/huggingface/hub`) and can take a while for large models. `niftsy setup`'s model menu shows which models are already downloaded. |

`--provider auto` (the default) picks based on the model name; pass it
explicitly to override.

**A note on `torch`/`vllm` versions:** these are pinned to a specific minor
version range in `pyproject.toml` (currently `torch>=2.9.1,<2.10`,
`vllm>=0.15.1,<0.16`) rather than left unbounded. An unbounded install can
silently pull a newer `torch` release that requires a newer CUDA driver
than your machine has — this happened in testing (a fresh install pulled
`torch==2.13.0`, which needs a newer driver than a machine running driver
570.211.01 / CUDA 12.8 had, and failed with a driver-version error at
engine init). If you hit a CUDA-driver error on the `local` provider,
check `nvidia-smi`'s reported driver/CUDA version against what your
installed `torch` build expects, or install within the pinned range above.

## Run log format

Every `generate` run writes one JSON file (see the CLI table above for the
default path). A `--dry-run`'s log gets a `dry-run_` prefix on its filename
— e.g. `dry-run_my_dataset_synthetic_2026-08-14.json` — so running a dry-run
and then a real run against the same input on the same day never collide on
the same log file. Example shape (real run):

```json
{
  "dry_run": false,
  "n_rows": 500,
  "text_columns": ["bio"],
  "duration_seconds": 92.4,
  "duration_human": "0h 1m 32s",
  "config": { "...": "the full GenerationConfig used for this run" },
  "config_hash": "a1b2c3d4e5f6",
  "input_csv": "data/my_dataset.csv",
  "output_csv": "data/my_dataset_synthetic_2026-08-14.csv",
  "llm_usage": {"prompt_tokens": 12345, "completion_tokens": 6789, "thinking_tokens": 0, "total_tokens": 19134, "requests": 500},
  "failed_row_indices": []
}
```

`config_hash` is a short fingerprint of the full config — two log files
with the same hash used identical settings.

## Development

```bash
uv sync --group dev
uv run pytest -v            # unit + integration tests, no network/GPU required
uv run pytest --cov=niftsy  # with coverage
uv run ruff check .
uv run mypy src/niftsy
```

Integration tests use an injectable fake `LLMBackend`, so the full suite
runs without any API key or GPU. Live-provider smoke testing (real Gemini/
OpenAI calls, a real local vLLM model) is manual — see `examples/` for
runnable scripts against each provider.
