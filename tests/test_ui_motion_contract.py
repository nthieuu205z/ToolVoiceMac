from pathlib import Path


ROOT = Path("web/static")


def test_voice_cards_stack_actions_inside_the_card():
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "grid-template-columns: 36px minmax(0, 1fr) auto" in css
    assert ".voice-actions { display: flex; flex-wrap: wrap;" in css
    assert ".voice-actions" in css
    assert "overflow: hidden" in css
    assert "voice-actions" in css
    assert "min-width: 0" in css
    assert "voice-actions" in js
    assert "@media (max-width: 900px)" in css


def test_queue_updates_existing_cards_instead_of_rebuilding_the_list():
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "function updateJobCard" in js
    assert "function renderJobs" in js
    assert "card.dataset.jobId" in js
    assert "card.animate" not in js
    assert "card.style.animationDelay" not in js
    render_start = js.index("function renderJobs")
    render_end = js.index("function renderSelectedJob")
    assert "list.textContent = \"\"" not in js[render_start:render_end]


def test_queue_progress_only_updates_the_progress_bar_without_card_animation():
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "updateJobCard" in js
    assert "style.width" in js
    assert "renderJobs();" in js


def test_execution_graph_has_directional_flow_classes_and_motion():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="pipelineGraph"' in html
    assert "renderGraphNodes(job)" in js
    assert "graph-flow" in js
    assert "flow-active" in css
    assert "flow-complete" in css
    assert "flow-active" in js
    assert "flow-complete" in js
    assert "graph-flow" in css
    assert "flow-active" in js
    assert "flow-complete" in js


def test_execution_graph_uses_a_track_and_layered_flow_effect():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'class="graph-track"' in html
    assert "graph-step" in js
    assert "graph-rail" in js
    assert ".graph-track {" in css
    assert ".graph-step {" in css
    assert ".graph-link::before" in css
    assert ".graph-flow::before" in css
    assert "animation: graph-packet" in css
    assert "column-gap: var(--graph-gap);" in css
    assert "right: calc(-1 * var(--graph-gap)); left: 100%;" in css
    assert ".graph-link.flow-complete .graph-flow { opacity: .3; animation: none; }" in css
    assert "--graph-gap: clamp(18px, 2vw, 28px);" in css
    assert "column-gap: var(--graph-gap);" in css
    assert ".graph-node { position: relative; z-index: 2; width: 100%;" in css
    assert ".graph-rail { position: absolute; z-index: 1; top: 50%; right: calc(-1 * var(--graph-gap)); left: 100%;" in css
    assert ".node-index::after" in css


def test_execution_graph_mobile_scroll_is_explicit_and_afforded():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert 'class="graph-scroll-shell"' in html
    assert 'class="graph-scroll-cue"' in html
    assert 'aria-hidden="true"' in html
    assert ".graph-scroll-shell {" in css
    assert ".graph-scroll-shell::after" in css
    assert "width: 30px;" in css
    assert "rgba(7,11,18,.78)" in css
    assert ".graph-scroll-cue" in css
    assert "overflow-x: auto;" in css
    assert "scrollbar-width: thin;" in css
    assert ".graph-scroll-shell.is-scrollable::after" in css or ".graph-scroll-shell::after { opacity: 1; }" in css
    assert ".graph-scroll-cue { opacity: .9; }" in css
    assert ".graph-scroll-shell.is-at-end::after" in css
    assert "  .graph-scroll-shell::after { opacity: 1; }" in css
    assert "  .graph-scroll-cue { position: absolute;" in css


def test_execution_graph_rail_connects_nodes_without_vertical_drift():
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert ".graph-track { position: relative;" in css
    assert ".graph-node.active {" in css
    assert "transform: none;" in css
    assert ".graph-rail { position: absolute; z-index: 1; top: 50%; right: calc(-1 * var(--graph-gap)); left: 100%;" in css
    assert "function syncGraphScrollAffordance()" in js
    assert "track.scrollWidth > shell.clientWidth + 1" in js
    assert "window.addEventListener(\"resize\", syncGraphScrollAffordance)" in js
    assert "graphTrack?.addEventListener(\"scroll\", syncGraphScrollAffordance" in js


def test_execution_graph_active_flow_uses_the_incoming_edge():
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "const flowIndex = currentIndex - 1;" in js
    assert "index === flowIndex && ACTIVE_STATUSES.has(job.status)" in js
    assert "index < flowIndex" in js
    assert "graph-frame-light" not in js


def test_execution_graph_active_link_has_a_between_node_light_sweep():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "graph-frame-light" not in html
    assert "graph-frame-light" not in css
    assert 'shell?.classList.toggle("is-running"' not in js
    assert ".graph-link.flow-active::before" in css
    assert "animation: link-edge-sweep" in css
    assert "@keyframes link-edge-sweep" in css
    assert ".graph-link.flow-active .graph-flow" in css


def test_execution_graph_scroll_cue_is_hidden_on_wide_layouts():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert ".graph-scroll-cue { display: none; }" in css
    assert ".graph-scroll-cue { position: absolute;" in css


def test_graph_flow_respects_reduced_motion():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert "prefers-reduced-motion" in css
    assert ".graph-flow" in css
    assert ".graph-flow { animation: none !important; opacity: .35; }" in css
