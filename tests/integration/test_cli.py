import json
import subprocess
import sys

import pandas as pd


def test_cli_dry_run(tmp_path):
    csv_path = tmp_path / "in.csv"
    pd.DataFrame({"age": [20, 30, 40], "bio": ["a", "b", "c"]}).to_csv(csv_path, index=False)
    result = subprocess.run(
        [sys.executable, "-m", "niftsy.cli", "generate", str(csv_path),
         "-o", str(tmp_path / "out.csv"), "--text-column", "bio",
         "--model", "fake-model", "--n-rows", "3", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "estimate" in result.stdout.lower()


def test_inspect_with_no_flags_auto_detects_free_text_columns(tmp_path):
    csv_path = tmp_path / "in.csv"
    pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "bio": [
            "works in tech as a software engineer",
            "teacher at the local elementary school",
            "nurse who specializes in emergency care",
            "student studying computer science full time",
            "retired after forty years of factory work",
        ],
    }).to_csv(csv_path, index=False)

    result = subprocess.run(
        [sys.executable, "-m", "niftsy.cli", "inspect", str(csv_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Detected likely free-text columns" in result.stdout
    assert "bio" in result.stdout
    assert "count=5" in result.stdout


def test_inspect_with_no_flags_and_no_detectable_columns_reports_none(tmp_path):
    csv_path = tmp_path / "in.csv"
    pd.DataFrame({"age": [20, 30, 40], "category": ["a", "a", "b"]}).to_csv(csv_path, index=False)

    result = subprocess.run(
        [sys.executable, "-m", "niftsy.cli", "inspect", str(csv_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "(none detected)" in result.stdout


def test_cli_dry_run_without_explicit_output_writes_auto_named_log(tmp_path):
    csv_path = tmp_path / "in.csv"
    pd.DataFrame({"age": [20, 30, 40], "bio": ["a", "b", "c"]}).to_csv(csv_path, index=False)
    result = subprocess.run(
        [sys.executable, "-m", "niftsy.cli", "generate", str(csv_path),
         "--text-column", "bio", "--model", "fake-model", "--n-rows", "3", "--dry-run"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    # dry-run logs get a distinct "dry-run_" prefix so a later real run on
    # the same day doesn't silently overwrite the dry-run's estimate. Match
    # by glob rather than a hardcoded timestamp to avoid a minute-boundary
    # race between this assertion and the subprocess's own clock read.
    matches = list(tmp_path.glob("dry-run_in_synthetic_*.json"))
    assert len(matches) == 1, matches
    log_path = matches[0]

    log = json.loads(log_path.read_text())
    assert log["dry_run"] is True
    assert log["output_csv"] is None
    assert log["input_csv"].endswith("in.csv")
    assert "config" in log
    assert "config_hash" in log
    assert log["llm_usage"] == {}
    assert log["failed_row_indices"] == []
