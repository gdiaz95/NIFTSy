import subprocess, sys, pandas as pd

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
