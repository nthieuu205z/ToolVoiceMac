from pathlib import Path


def test_bootstrap_binds_document_without_passing_it_to_selector_helper():
    source = Path("web/static/app.js").read_text(encoding="utf-8")

    assert '$(document).addEventListener' not in source
    assert 'document.addEventListener("click"' in source


def test_bootstrap_has_one_document_click_delegation():
    source = Path("web/static/app.js").read_text(encoding="utf-8")

    assert source.count('document.addEventListener("click"') == 1
