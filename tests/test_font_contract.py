from pathlib import Path


ROOT = Path("web/static")


def test_ui_uses_a_vietnamese_safe_ui_font_and_mono_only_for_technical_data():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert "Be+Vietnam+Pro" in html
    assert "display=swap" in html
    assert '--sans: "Be Vietnam Pro"' in css
    assert '--mono: "JetBrains Mono"' in css
    assert "font-family: var(--sans);" in css


def test_voice_card_controls_do_not_use_tracking_heavy_monospace_text():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert ".voice-preview" in css
    assert ".voice-select-button" in css
    assert "font: 600 10px var(--sans)" in css
    assert "letter-spacing: 0" in css


def test_voice_cards_reserve_space_for_name_and_actions():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert "grid-template-columns: repeat(auto-fit, minmax(420px, 1fr))" in css
    assert "grid-template-columns: 30px minmax(0, 1fr) 193px" in css
    assert "text-overflow: ellipsis" in css


def test_static_asset_version_includes_the_multilingual_composer_release():
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert 'href="style.css?v=20260906-39"' in html
    assert 'src="app.js?v=20260906-39"' in html
