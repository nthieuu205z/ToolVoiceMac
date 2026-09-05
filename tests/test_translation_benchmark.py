from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_translation_benchmark_script_supports_direct_help_invocation():
    result = subprocess.run(
        [sys.executable, "benchmarks/run_translation_baseline.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--batches" in result.stdout
    assert "--workers" in result.stdout


def test_translation_benchmark_report_never_contains_synthetic_line_text(tmp_path: Path):
    report = tmp_path / "translation.json"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/run_translation_baseline.py",
            "--dry-run",
            "--batches",
            "2",
            "--workers",
            "1,2",
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "synthetic benchmark line" not in serialized
    assert "api_key" not in serialized
    assert payload["run_metadata"]["kind"] == "translation-baseline"
    assert len(payload["measurements"]) == 2
