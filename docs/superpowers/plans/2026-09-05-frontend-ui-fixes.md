# Frontend UI Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make job progress truly live in `VIDEO QUEUE`, remove the duplicate selected-job progress bar, complete the `Export` graph node for finished videos, and keep selected voice controls inside their card.

**Architecture:** Keep the existing SSE stream as the primary realtime transport and correct the exception that currently aborts queue rendering after selected-job rendering. The selected-job panel will retain textual stage/percent/ETA information without a second bar; graph completion will derive from terminal job status, while voice action sizing will reserve enough width for the Vietnamese selected label.

**Tech Stack:** Vanilla JavaScript, HTML, CSS, FastAPI SSE, pytest source contracts, MCP Playwright browser checks.

**Spec:** User-provided request in the current task; no separate feature specification file.

## Global Constraints

- Preserve all unrelated existing migration changes in the dirty worktree.
- Keep `VIDEO QUEUE` as the only progress bar for job processing.
- Verify UI behavior with MCP Playwright at desktop and mobile viewport sizes.
- Do not add frontend build dependencies or change backend job semantics.

### Task 1: Add regression contracts first

**Files:**
- Create: `tests/test_frontend_requested_fixes.py`
- Test: `tests/test_frontend_requested_fixes.py`

**Interfaces:**
- Consumes: current `web/static/app.js`, `web/static/index.html`, and `web/static/style.css`.
- Produces: failing contracts for SSE queue rendering, selected-panel markup, terminal graph state, and voice-button sizing.

- [x] **Step 1: Write the failing tests**

```python
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
```

- [x] **Step 2: Run the focused test file and verify RED**

Run: `./.venv/bin/python -m pytest tests/test_frontend_requested_fixes.py -q`

Expected: FAIL because the current source still calls the undefined `renderGraphFlow`, contains `selectedProgress`, does not complete `mux` nodes, and reserves only 58px for `Đang chọn`.

### Task 2: Implement the four frontend fixes

**Files:**
- Modify: `web/static/app.js:171-196`
- Modify: `web/static/index.html:76-80,130`
- Modify: `web/static/style.css:243-248`

**Interfaces:**
- Consumes: the existing `state`, `renderJobs`, `renderSelectedJob`, `renderGraph`, `ensureJobStream`, and voice-card DOM structure.
- Produces: queue updates after every valid SSE snapshot, one processing progress bar in the queue, completed `Export` state for terminal jobs, and a non-overflowing selected voice button.

- [x] **Step 1: Fix SSE callback and graph completion logic**

Use `updateGraphFlow(snapshot)` in the SSE callback, and mark the current node completed for `done` jobs:

```js
const completed = Boolean(job) && (job.status === "done" ? index <= currentIndex : index < currentIndex);
node.classList.toggle("completed", completed);
```

- [x] **Step 2: Remove the selected-panel progress bar**

Keep the stage, percentage, message, and ETA row in `SELECTED JOB`, but remove the `#selectedProgress` element and its JavaScript width update. Leave the queue card’s `.progress-track` unchanged.

- [x] **Step 3: Reserve width for the selected voice label**

Change `.voice-select-button` to `min-width: 72px; flex: 0 0 72px;` and add `overflow: hidden; text-overflow: ellipsis;` as a defensive boundary for narrow cards.

- [x] **Step 4: Bump static asset cache keys**

Change both asset query strings in `index.html` from `20260901-35` to `20260905-36` so existing browser tabs load the fix.

### Task 3: Verify with MCP Playwright and repository checks

**Files:**
- Modify: none
- Test: `tests/test_frontend_requested_fixes.py` and MCP Playwright session

**Interfaces:**
- Consumes: the fixed static assets served by `http://127.0.0.1:8000/`.
- Produces: browser evidence for queue live updates, single-bar markup, `Export.completed`, and mobile voice-card bounds.

- [x] **Step 1: Run focused pytest and static checks**

Run: `./.venv/bin/python -m pytest tests/test_frontend_requested_fixes.py tests/test_ui_contract.py tests/test_ui_motion_contract.py -q`, `node --check web/static/app.js`, and `git diff --check`.

- [x] **Step 2: Run the MCP Playwright queue regression**

Mock `/api/jobs` with a running job and inject two `EventSource` snapshots (12% then 48%). Assert the queue card changes to `48%` and the selected panel also shows `48%`.

- [x] **Step 3: Run the MCP Playwright visual/DOM checks**

Assert `#selectedProgress` is absent, a finished job has `data-stage="mux"` with class `completed`, and at a 360px viewport the selected voice button has no text overflow, card overflow, or page horizontal overflow.

- [x] **Step 4: Run the broader verification suite**

Run: `./.venv/bin/python -m pytest -q` with the repository’s known proxy override if required by the local environment, then report exact results.
