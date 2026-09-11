"""Verify the real queue renderer keeps new jobs first without replacing cards."""
import subprocess


def test_newest_jobs_first():
    result = subprocess.run(
        ['node', 'tests/job_queue_order.cjs'], text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
