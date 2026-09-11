"""Projects preserve text and publish only complete, ordered audio exports."""

import importlib
import json
import threading
import wave
import zipfile

import numpy as np
import pytest

from pipeline.errors import JobCancelledError, PipelineError
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.text_to_voice import TextToVoiceOptions


def project_module():
    return importlib.import_module("pipeline.speech_project")


@pytest.mark.parametrize("text", [
    ("A paragraph. More words!\n\n" * 9000)[:200_000],
    "ạ" * 200_000,
    "hello" + " " * 10_001 + "world",
], ids=["paragraphs", "unbroken-unicode", "whitespace"])
def test_partition_covers_every_character_once_at_200k(text):
    parts = project_module().split_project_text(text)
    assert "".join(parts) == text
    assert all(0 < len(part) <= 5000 for part in parts)


def test_partition_prefers_paragraph_then_sentence_then_whitespace():
    split = project_module().split_project_text
    assert split("First.\n\nSecond sentence.", max_characters=16) == ["First.\n\n", "Second sentence."]
    assert split("First. Second sentence.", max_characters=16) == ["First. ", "Second sentence."]
    assert split("first second third", max_characters=13) == ["first second ", "third"]


class SequenceSpeech:
    engine = "test"

    def __init__(self, *, fail_at=None, cancel=None):
        self.calls = []
        self.fail_at = fail_at
        self.cancel = cancel
        self.active = 0
        self.maximum_active = 0

    def synthesize(self, text, voice, *, language):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            self.calls.append((text, voice, language))
            if self.cancel and len(self.calls) == 2:
                self.cancel.set()
            if len(self.calls) == self.fail_at:
                raise RuntimeError("speech failed")
            return np.full(120, len(self.calls), dtype="<i2").tobytes()
        finally:
            self.active -= 1


@pytest.fixture
def cheap_encoding(monkeypatch):
    # Keep real sentence synthesis/WAV assembly; only substitute external ffmpeg.
    monkeypatch.setattr("pipeline.text_to_voice.encode_mp3", lambda wav, mp3: mp3.write_bytes(wav.read_bytes()))


def test_project_archives_ordered_audio_text_and_complete_manifest(tmp_path, cheap_encoding):
    backend = SequenceSpeech()
    text = "Alpha. " + "x" * 4990 + "\n\nBeta."
    result = project_module().run_speech_project(
        backend, text, tmp_path, TextToVoiceOptions("voice", "en-US"), name="../My / book",
    )
    assert backend.maximum_active == 1
    assert all(voice == "voice" and language == "en-US" for _, voice, language in backend.calls)
    assert result.attempted_count == result.spoken_count == len(backend.calls)
    with zipfile.ZipFile(result.zip_path) as archive:
        assert all(not name.startswith("/") and ".." not in name.split("/") for name in archive.namelist())
        manifest = json.loads(archive.read("My book/manifest.json"))
        assert manifest["status"] == "done"
        assert manifest["completed_parts"] == manifest["total_parts"] == 2
        assert "".join(archive.read("My book/" + p["files"]["text"]).decode() for p in manifest["parts"]) == text
        last_sample = 0
        for part in manifest["parts"]:
            with archive.open("My book/" + part["files"]["wav"]) as stream, wave.open(stream) as wav:
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            nonzero = samples[samples != 0]
            assert nonzero[0] > last_sample
            assert np.all(nonzero[1:] >= nonzero[:-1])
            last_sample = nonzero[-1]


def test_project_does_not_publish_zip_after_a_partial_speech_failure(tmp_path, cheap_encoding):
    backend = SequenceSpeech(fail_at=2)
    with pytest.raises(PipelineError):
        project_module().run_speech_project(
            backend, "First. Second.", tmp_path, TextToVoiceOptions("voice", "en-US"), name="Book",
        )
    assert not (tmp_path / "project.zip").exists()
    manifest = json.loads((tmp_path / "project.json").read_text())
    assert manifest["status"] == "error"
    assert manifest["parts"][0]["spoken_count"] < manifest["parts"][0]["attempted_count"]


def test_cancel_retains_diagnostic_manifest_without_publishing_zip(tmp_path, cheap_encoding):
    cancel = threading.Event()
    with pytest.raises(JobCancelledError):
        project_module().run_speech_project(
            SequenceSpeech(cancel=cancel), "First. Second.", tmp_path,
            TextToVoiceOptions("voice", "en-US"), name="Book", should_cancel=cancel.is_set,
        )
    assert not (tmp_path / "project.zip").exists()
    manifest = json.loads((tmp_path / "project.json").read_text())
    assert manifest["status"] == "cancelled"
    assert manifest["parts"][0]["status"] == "cancelled"


def test_200k_project_keeps_audio_parts_bounded_and_in_exact_order(tmp_path, cheap_encoding):
    backend = SequenceSpeech()
    text = "ạ" * 200_000
    result = project_module().run_speech_project(backend, text, tmp_path, TextToVoiceOptions("voice", "en-US"), name="Long")
    assert "".join(call[0] for call in backend.calls) == text
    assert max(len(call[0]) for call in backend.calls) == 1000
    assert backend.maximum_active == 1
    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("Long/manifest.json"))
        assert manifest["total_parts"] == 40
        assert all(part["characters"] == 5000 for part in manifest["parts"])
        for index, part in enumerate(manifest["parts"]):
            with archive.open("Long/" + part["files"]["wav"]) as stream, wave.open(stream) as wav:
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            assert samples[0] == index * 5 + 1
            assert samples[-1] == index * 5 + 5


def test_cancel_while_packaging_removes_archive_and_marks_manifest(tmp_path, cheap_encoding, monkeypatch):
    cancel = threading.Event()
    real_write = zipfile.ZipFile.write
    def write_and_cancel(self, *args, **kwargs):
        result = real_write(self, *args, **kwargs)
        cancel.set()
        return result
    monkeypatch.setattr(zipfile.ZipFile, "write", write_and_cancel)
    with pytest.raises(JobCancelledError):
        project_module().run_speech_project(SequenceSpeech(), "Hello.", tmp_path, TextToVoiceOptions("voice", "en-US"), name="Book", should_cancel=cancel.is_set)
    assert not (tmp_path / "project.zip").exists()
    assert not (tmp_path / ".project.zip.tmp").exists()
    assert json.loads((tmp_path / "project.json").read_text())["status"] == "cancelled"


def test_failed_second_part_stops_later_parts_and_retains_first_audio(tmp_path, cheap_encoding):
    backend = SequenceSpeech(fail_at=6)
    with pytest.raises(PipelineError):
        project_module().run_speech_project(backend, "x" * 12_000, tmp_path, TextToVoiceOptions("voice", "en-US"), name="Book")
    manifest = json.loads((tmp_path / "project.json").read_text())
    assert [part["status"] for part in manifest["parts"]] == ["done", "error", "pending"]
    assert manifest["completed_parts"] == 1
    assert manifest["attempted_count"] == 10
    assert manifest["spoken_count"] == 9
    assert len(backend.calls) == 10
    assert (tmp_path / "project_parts" / manifest["parts"][0]["files"]["wav"]).exists()
    assert not (tmp_path / "project.zip").exists()


def test_manifest_readers_never_observe_partial_json(tmp_path, cheap_encoding):
    module = project_module()
    options = TextToVoiceOptions("voice", "en-US")
    module.initialize_project("x" * 50_000, tmp_path, options, name="Book")
    stop = threading.Event()
    errors = []
    observations = []
    def read_repeatedly():
        while not stop.is_set():
            try:
                manifest = module.read_project_manifest(tmp_path)
                observations.append(manifest["status"])
                assert sum(part["characters"] for part in manifest["parts"]) == 50_000
            except Exception as exc:
                errors.append(exc)
    reader = threading.Thread(target=read_repeatedly)
    reader.start()
    try:
        module.run_speech_project(SequenceSpeech(), "x" * 50_000, tmp_path, options, name="Book")
    finally:
        stop.set()
        reader.join(timeout=2)
    assert not errors
    assert observations
    assert module.read_project_manifest(tmp_path)["status"] == "done"


@pytest.mark.parametrize("target,key,value", [
    ("project", "status", "unknown"), ("project", "name", None),
    ("project", "total_parts", 2), ("project", "total_characters", "6"),
    ("project", "completed_parts", True), ("project", "attempted_count", -1),
    ("project", "spoken_count", 1), ("project", "voice_id", []),
    ("project", "language", None), ("project", "error", {}),
    ("part", "index", 0), ("part", "title", None),
    ("part", "characters", 0), ("part", "status", None),
    ("part", "attempted_count", "0"), ("part", "spoken_count", 1),
    ("part", "files", {}), ("part", "error", []),
], ids=lambda value: str(value))
def test_reader_rejects_malformed_required_progress_fields(tmp_path, target, key, value):
    module = project_module()
    manifest = module.initialize_project("Hello.", tmp_path, TextToVoiceOptions("voice", "en-US"), name="Book")
    record = manifest if target == "project" else manifest["parts"][0]
    record[key] = value
    (tmp_path / "project.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.read_project_manifest(tmp_path)
