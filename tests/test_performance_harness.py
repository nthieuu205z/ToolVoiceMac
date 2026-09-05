from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.performance_harness import measure, write_json_report


def test_local_baseline_script_supports_direct_help_invocation():
    result = subprocess.run(
        [sys.executable, "benchmarks/run_local_baseline.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--source-wav" in result.stdout


def test_measure_captures_throughput_resources_and_safe_metadata():
    measurement, result = measure(
        "synthetic-stage",
        lambda: (time.sleep(0.01), "ok")[1],
        item_count=4,
        warm=True,
        metadata={
            "stage": "translate",
            "provider": "gemini",
            "device": "cloud",
            "workers": 4,
            "api_key": "[REDACTED]",
            "prompt": "must never be persisted",
        },
    )

    assert result == "ok"
    assert measurement.label == "synthetic-stage"
    assert measurement.item_count == 4
    assert measurement.warm is True
    assert measurement.elapsed_seconds > 0
    assert measurement.items_per_second > 0
    assert measurement.metadata == {
        "stage": "translate",
        "provider": "gemini",
        "device": "cloud",
        "workers": 4,
    }
    assert measurement.resources_before.rss_mb >= 0
    assert measurement.resources_after.rss_mb >= 0
    assert measurement.peak_rss_mb >= measurement.resources_before.rss_mb


def test_measurement_records_error_type_without_persisting_exception_text():
    def broken_operation():
        raise RuntimeError("contains a secret-looking message")

    measurement, result = measure("broken", broken_operation)

    assert result is None
    assert measurement.error_type == "RuntimeError"
    assert measurement.error_message is None


def test_json_report_contains_only_non_secret_measurement_fields(tmp_path: Path):
    measurement, _ = measure(
        "report-case",
        lambda: "done",
        metadata={"stage": "extract", "token": "[REDACTED]"},
    )
    report_path = tmp_path / "report.json"

    write_json_report(report_path, [measurement], run_metadata={"fixture": "synthetic"})

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["run_metadata"] == {"fixture": "synthetic"}
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "api_key" not in serialized
    assert "token" not in serialized
    assert "prompt" not in serialized
    assert "secret-looking" not in serialized
    assert payload["measurements"][0]["label"] == "report-case"
