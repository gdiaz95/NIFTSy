import json
import subprocess
import sys
from datetime import date

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


def test_cli_dry_run_without_explicit_output_writes_auto_named_log(tmp_path):
    csv_path = tmp_path / "in.csv"
    pd.DataFrame({"age": [20, 30, 40], "bio": ["a", "b", "c"]}).to_csv(csv_path, index=False)
    result = subprocess.run(
        [sys.executable, "-m", "niftsy.cli", "generate", str(csv_path),
         "--text-column", "bio", "--model", "fake-model", "--n-rows", "3", "--dry-run"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    today = date.today().isoformat()
    log_path = tmp_path / f"in_synthetic_{today}.json"
    assert log_path.exists()

    log = json.loads(log_path.read_text())
    assert log["dry_run"] is True
    assert log["output_csv"] is None
    assert log["input_csv"].endswith("in.csv")
    assert "config" in log
    assert "config_hash" in log
    assert log["llm_usage"] == {}
    assert log["failed_row_indices"] == []
