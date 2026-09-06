"""Phần trăm phải tăng đều và không bao giờ lùi khi chuyển bước."""

from __future__ import annotations

import re

import pytest

from pipeline.models import (
    STAGES,
    STAGES_BY_JOB_TYPE,
    STAGE_WEIGHTS_BY_JOB_TYPE,
    TEXT_STAGES,
    VIDEO_STAGES,
    overall_percent,
)


def test_first_stage_starts_at_zero():
    assert overall_percent("extract", 0.0) == 0.0


def test_last_stage_ends_at_one_hundred():
    assert overall_percent("mux", 1.0) == 100.0


def test_percent_never_decreases_across_stages():
    values = [overall_percent(stage, f) for stage in STAGES for f in (0.0, 0.5, 1.0)]
    assert values == sorted(values)


def test_stage_end_equals_next_stage_start():
    for stage, nxt in zip(STAGES, STAGES[1:]):
        assert overall_percent(stage, 1.0) == overall_percent(nxt, 0.0)


@pytest.mark.parametrize("fraction", [-1.0, 2.0])
def test_out_of_range_fractions_are_clamped(fraction):
    assert 0.0 <= overall_percent("translate", fraction) <= 100.0


def test_stage_names_match_the_frontend_data_attributes():
    """app.js và index.html dựa vào đúng bảy tên này, theo đúng thứ tự này."""
    assert STAGES == ["extract", "transcribe", "translate", "synthesize",
                      "subtitle", "assemble", "mux"]


def test_subtitles_are_built_after_speech():
    """Cue phụ đề bám theo thời lượng đọc thật, nên phải chạy sau bước tạo giọng đọc."""
    assert STAGES.index("subtitle") > STAGES.index("synthesize")


def test_text_jobs_use_their_own_ordered_stages_and_weights():
    assert TEXT_STAGES == ("prepare", "synthesize", "assemble", "export")
    assert STAGES_BY_JOB_TYPE["text_to_voice"] == TEXT_STAGES
    assert sum(STAGE_WEIGHTS_BY_JOB_TYPE["text_to_voice"].values()) == 100
    assert overall_percent("prepare", 0.0, job_type="text_to_voice") == 0.0
    assert overall_percent("export", 1.0, job_type="text_to_voice") == 100.0


def test_video_stage_alias_retains_the_legacy_list_contract():
    assert tuple(STAGES) == VIDEO_STAGES
    assert STAGES_BY_JOB_TYPE["video_dubbing"] == VIDEO_STAGES


def test_unknown_restored_stage_returns_its_persisted_percentage():
    assert overall_percent(
        "retired-stage",
        0.5,
        job_type="text_to_voice",
        fallback=42.5,
    ) == 42.5


def test_unknown_job_type_without_a_fallback_is_neutral():
    assert overall_percent("extract", 0.5, job_type="future_job") == 0.0


def test_frontend_stage_labels_match_the_backend_order():
    """app.js hiển thị nhãn từng bước trên thẻ job — thiếu bước nào là thẻ hiện tên thô."""
    from pathlib import Path

    js = Path("web/static/app.js").read_text(encoding="utf-8")
    # Frontend mới dùng metadata có nhãn tiếng Việt thay vì map trạng thái cũ.
    video_order = re.search(r'const STAGES = \[([^\]]+)\]', js)
    text_order = re.search(r'const TEXT_STAGES = \[([^\]]+)\]', js)
    assert video_order is not None and re.findall(r'"(\w+)"', video_order.group(1)) == STAGES
    assert text_order is not None and re.findall(r'"(\w+)"', text_order.group(1)) == list(TEXT_STAGES)
    for stage in (*STAGES, *TEXT_STAGES):
        assert re.search(rf"^\s{{2}}{stage}: \{{ label:", js, flags=re.MULTILINE)
