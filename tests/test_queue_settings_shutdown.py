"""Regression contracts cho queue, Gemini settings, xóa job và shutdown local tool."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import settings
from backend.job_manager import Job, JobManager
from backend.main import app

ROOT = Path("web/static")


def test_video_queue_is_bounded_and_scrolls_inside_the_panel():
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert ".job-list {" in css
    assert "max-height: 540px;" in css
    assert "overflow-y: auto;" in css
    assert "scrollbar-gutter: stable;" in css
    assert "function syncJobListViewport()" in js
    assert "cards.slice(0, 3)" in js
    assert "list.style.maxHeight" in js
    assert "window.addEventListener(\"resize\", syncJobListViewport)" in js


def test_gemini_settings_form_is_password_only_and_not_persisted_in_browser_storage():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="geminiSettingsForm"' in html
    assert 'id="geminiApiKey"' in html
    assert 'type="password"' in html
    assert 'id="geminiBackend"' in html
    assert 'id="geminiSettingsHint"' in html
    assert "/api/settings/gemini" in js
    assert 'classList.toggle("configured", Boolean(data.configured))' in js
    assert 'localStorage.setItem("gemini' not in js
    assert 'localStorage.setItem("sub.gemini' not in js


def test_gemini_settings_panel_has_a_responsive_form_layout():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert ".settings-panel {" in css
    assert ".settings-form { display: grid;" in css
    assert "grid-template-columns: minmax(0, 1.3fr) minmax(220px, .7fr) auto;" in css
    assert ".settings-form input, .settings-form select" in css


def test_completed_job_cards_expose_a_confirmed_delete_action():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="shutdownButton"' in html
    assert 'data-action="delete-job"' in js
    assert "async function deleteJob" in js
    assert "window.confirm" in js
    assert 'method: "DELETE"' in js
    assert "await refreshJobs()" in js
    assert "const remove = !ACTIVE_STATUSES.has(job.status)" in js


def test_shutdown_button_is_confirmed_and_calls_the_local_shutdown_route():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="shutdownButton"' in html
    assert "setupShutdown" in js
    assert "/api/shutdown" in js
    assert "window.confirm" in js
    assert "Tắt tool" in html


def test_gemini_status_never_returns_the_api_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "[REDACTED]")
    monkeypatch.setattr(settings, "gemini_backend", "vertex")

    with TestClient(app) as client:
        response = client.get("/api/settings/gemini")

    assert response.status_code == 200
    assert response.json() == {"configured": True, "backend": "vertex"}
    assert "[REDACTED]" not in response.text


def test_gemini_settings_update_persists_key_and_backend_without_echoing_it(tmp_path, monkeypatch):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    env_path.write_text("OTHER_SETTING=keep\nGEMINI_BACKEND=developer\n", encoding="utf-8")
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_backend", "developer")

    with TestClient(app) as client:
        response = client.post(
            "/api/settings/gemini",
            json={"api_key": "[REDACTED]", "backend": "vertex"},
        )

    assert response.status_code == 200
    assert response.json() == {"saved": True, "configured": True, "backend": "vertex"}
    assert response.json().get("api_key") is None
    assert env_path.read_text(encoding="utf-8") == (
        "OTHER_SETTING=keep\nGEMINI_BACKEND=vertex\nGEMINI_API_KEY=\"[REDACTED]\"\n"
    )
    assert settings.gemini_api_key == "[REDACTED]"
    settings.gemini_api_key = ""
    settings.gemini_backend = "developer"


def test_gemini_settings_quotes_special_characters_before_writing_env(tmp_path, monkeypatch):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_backend", "developer")

    with TestClient(app) as client:
        response = client.post(
            "/api/settings/gemini",
            json={"api_key": "key-with-\"quote\"-and-#-hash", "backend": "developer"},
        )

    assert response.status_code == 200
    assert env_path.read_text(encoding="utf-8").endswith(
        'GEMINI_API_KEY="key-with-\\"quote\\"-and-#-hash"\n'
    )
    assert settings.gemini_backend == "developer"
    settings.gemini_api_key = ""


def test_gemini_settings_rejects_newline_in_api_key(tmp_path, monkeypatch):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    env_path.write_text("OTHER_SETTING=keep\n", encoding="utf-8")
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_backend", "developer")

    with TestClient(app) as client:
        response = client.post(
            "/api/settings/gemini",
            json={"api_key": "key-part-1\nINJECTED=value", "backend": "developer"},
        )

    assert response.status_code == 400
    assert env_path.read_text(encoding="utf-8") == "OTHER_SETTING=keep\n"
    settings.gemini_api_key = ""
    settings.gemini_backend = "developer"


def test_job_manager_deletes_terminal_job_files_and_registry_entry(tmp_path):
    manager = JobManager(max_workers=1)
    workdir = tmp_path / "job"
    workdir.mkdir()
    (workdir / "job.json").write_text("{}", encoding="utf-8")
    (workdir / "output.mp4").write_bytes(b"video")
    job = Job(
        id="done-job",
        filename="clip.mp4",
        workdir=workdir,
        voice_id="Kore",
        status="done",
    )
    manager._jobs[job.id] = job

    assert manager.delete(job.id) == job.id
    assert manager.get(job.id) is None
    assert not workdir.exists()


def test_job_manager_refuses_to_delete_active_job(tmp_path):
    manager = JobManager(max_workers=1)
    job = Job(
        id="running-job",
        filename="clip.mp4",
        workdir=tmp_path,
        voice_id="Kore",
        status="running",
    )
    manager._jobs[job.id] = job

    try:
        manager.delete(job.id)
    except RuntimeError as exc:
        assert "đang chạy" in str(exc)
    else:
        raise AssertionError("active job deletion should be rejected")


def test_shutdown_process_tree_only_contains_descendants():
    from backend.process_control import _descendants

    table = {
        100: 1,
        101: 100,
        102: 101,
        103: 7,
        104: 100,
    }

    assert _descendants(100, table) == {101, 102, 104}


def test_launcher_replaces_the_shell_with_the_server_process():
    launcher = Path("ChayTool.command").read_text(encoding="utf-8")

    assert "exec .venv/bin/python -m uvicorn backend.main:app --port 8000" in launcher
    assert "python -m uvicorn backend.main:app --port 8000\n\n echo" not in launcher


def test_shutdown_route_is_registered(monkeypatch):
    monkeypatch.setattr("backend.main.schedule_shutdown", lambda: None)

    with TestClient(app) as client:
        response = client.post("/api/shutdown")

    assert response.status_code == 200
    assert response.json() == {"shutting_down": True}
