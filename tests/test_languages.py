import pytest

from pipeline.languages import available_languages, normalize_language_code, require_language


def test_initial_registry_uses_canonical_bcp47_codes():
    assert [item.code for item in available_languages()] == ["vi-VN", "en-US"]
    assert require_language("vi-VN").omnivoice_name == "Vietnamese"
    assert require_language("en-US").gemini_tts_code == "en-US"


@pytest.mark.parametrize(("raw", "expected"), [
    ("vi", "vi-VN"), ("vi-vn", "vi-VN"),
    ("en", "en-US"), ("EN-us", "en-US"),
])
def test_language_aliases_normalize_at_boundaries(raw, expected):
    assert normalize_language_code(raw) == expected


def test_unknown_language_is_rejected():
    with pytest.raises(ValueError, match="Ngôn ngữ không được hỗ trợ"):
        normalize_language_code("fr")
