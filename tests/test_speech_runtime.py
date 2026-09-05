"""Process-wide provider activity gives production priority over previews."""

from __future__ import annotations

from threading import Event, Thread

import pytest

from pipeline.speech_runtime import PreviewBusyError, SpeechActivity, speech_activity


def test_preview_is_rejected_while_production_is_active():
    activity = SpeechActivity()

    with activity.production("omnivoice"):
        with pytest.raises(PreviewBusyError):
            with activity.preview("omnivoice"):
                raise AssertionError("preview body must not run")


def test_parallel_production_is_allowed_across_threads():
    activity = SpeechActivity()
    entered = Event()
    release = Event()

    def run_production() -> None:
        with activity.production("edge"):
            entered.set()
            release.wait()

    worker = Thread(target=run_production)
    try:
        with activity.production("edge"):
            worker.start()
            assert entered.wait(timeout=1)
            assert activity.active_production("edge") == 2
    finally:
        release.set()
        worker.join(timeout=1)
    assert not worker.is_alive()


def test_preview_is_single_slot():
    activity = SpeechActivity()

    with activity.preview("edge"):
        with pytest.raises(PreviewBusyError):
            with activity.preview("edge"):
                raise AssertionError("second preview body must not run")


def test_different_providers_do_not_block_each_other():
    activity = SpeechActivity()

    with activity.production("omnivoice"):
        with activity.preview("edge"):
            assert activity.active_production("omnivoice") == 1


def test_production_waits_only_until_an_active_preview_finishes():
    activity = SpeechActivity()
    attempted = Event()
    entered = Event()
    release = Event()

    def run_production() -> None:
        attempted.set()
        with activity.production("edge"):
            entered.set()
            release.wait()

    worker = Thread(target=run_production)
    try:
        with activity.preview("edge"):
            worker.start()
            assert attempted.wait(timeout=1)
            assert not entered.is_set()

        assert entered.wait(timeout=1)
        with pytest.raises(PreviewBusyError):
            with activity.preview("edge"):
                raise AssertionError("preview must yield to resumed production")
    finally:
        release.set()
        worker.join(timeout=1)
    assert not worker.is_alive()


def test_module_exports_one_process_wide_activity_instance():
    assert isinstance(speech_activity, SpeechActivity)
