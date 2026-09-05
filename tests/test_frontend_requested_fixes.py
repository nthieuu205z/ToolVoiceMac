from pathlib import Path


ROOT = Path("web/static")


def test_sse_snapshot_renders_queue_after_selected_job_updates():
    source = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "renderGraphFlow(snapshot)" not in source
    assert "updateGraphFlow(snapshot)" in source


def test_selected_job_has_no_duplicate_processing_progress_bar():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    source = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="selectedProgress"' not in html
    assert "selectedProgress" not in source
    assert 'class="progress-track large"' not in html


def test_finished_job_marks_export_node_completed():
    source = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'job.status === "done" ? index <= currentIndex : index < currentIndex' in source


def test_selected_voice_label_fits_inside_action_group():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert "min-width: 72px;" in css
    assert "flex: 0 0 72px;" in css
