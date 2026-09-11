"""Standalone CLI tests with synthetic inputs; no real model or application."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest


def cli_module():
    assert importlib.util.find_spec("run_omnivoice") is not None, "standalone CLI is missing"
    return importlib.import_module("run_omnivoice")


def arguments(tmp_path):
    paths = {name: tmp_path / name for name in ("input.json", "reference.wav", "transcript.txt", "output")}
    paths["input.json"].write_text(json.dumps(["Second input", "First input"]), encoding="utf-8")
    paths["reference.wav"].write_bytes(b"synthetic reference")
    paths["transcript.txt"].write_text("Explicit synthetic transcript", encoding="utf-8")
    argv = ["--input", str(paths["input.json"]), "--reference-audio", str(paths["reference.wav"]),
            "--reference-text", str(paths["transcript.txt"]), "--language", "vi-VN",
            "--output-dir", str(paths["output"]), "--checkpoint", "k2-fsa/OmniVoice", "--seed", "1234"]
    return argv, paths


def fake_result(count=2):
    return SimpleNamespace(pcm=tuple(b"\x00\x00\xff\x7f" for _ in range(count)), sample_rate=24000,
                           timings={"load_seconds": 1.0, "prompt_seconds": 2.0, "generate_seconds": 3.0,
                                    "pcm_seconds": 0.01, "total_seconds": 6.01},
                           provenance={"runtime": "omnivoice", "device": "mps", "dtype": "float16",
                                       "generation_num_step": 32, "generation_guidance_scale": 2.0})


def test_cli_forwards_one_request_and_publishes_ordered_wav_metadata(tmp_path, capsys):
    module = cli_module()
    argv, paths = arguments(tmp_path)
    settings = tmp_path / "settings.json"
    settings.write_text('{"speed":1.25,"num_step":40,"preprocess_prompt":false}', encoding="utf-8")
    calls = []

    class Runtime:
        def __init__(self, reference_audio, reference_text, **kwargs):
            calls.append((reference_audio, reference_text, kwargs))

        def generate(self, texts, **kwargs):
            calls.append((texts, kwargs))
            assert not paths["output"].exists()
            return fake_result()

    assert module.main([*argv, "--settings", str(settings)], runtime_factory=Runtime) == 0
    assert len(calls) == 2
    assert calls[0][0] == paths["reference.wav"]
    assert calls[0][1] == "Explicit synthetic transcript"
    assert calls[0][2]["checkpoint"] == "k2-fsa/OmniVoice"
    assert calls[0][2]["settings"].speed == 1.25
    assert calls[0][2]["settings"].num_step == 40
    assert calls[0][2]["settings"].preprocess_prompt is False
    assert calls[1] == (["Second input", "First input"], {"language": "vi-VN", "seed": 1234})
    assert sorted(p.name for p in paths["output"].iterdir()) == ["0000.wav", "0001.wav", "metadata.json"]
    with wave.open(str(paths["output"] / "0000.wav"), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (24000, 1, 2, 2)
        assert wav.readframes(2) == b"\x00\x00\xff\x7f"
    metadata = json.loads((paths["output"] / "metadata.json").read_text())
    assert metadata["count"] == 2 and metadata["sample_rate"] == 24000
    assert metadata["provenance"]["generation_num_step"] == 32
    assert metadata["timings"]["generate_seconds"] == 3.0
    assert metadata["timings"]["write_seconds"] >= 0
    serialized = json.dumps(metadata)
    assert str(tmp_path) not in serialized and "transcript" not in serialized and "Second input" not in serialized
    assert capsys.readouterr().out == "Wrote 2 WAV files\n"


@pytest.mark.parametrize("case", ["invalid-json", "object", "empty", "blank", "nonstring", "many", "long",
    "missing-audio", "audio-directory", "blank-transcript", "missing-transcript", "missing-input",
    "invalid-utf8", "language", "seed", "negative-seed", "settings", "unknown-setting", "duration-batch",
    "duration-long", "empty-checkpoint", "existing-output", "existing-file", "dangling-output"])
def test_invalid_cli_input_rejected_before_runtime_factory(tmp_path, case, capsys):
    module = cli_module()
    argv, paths = arguments(tmp_path)
    if case == "invalid-json":
        paths["input.json"].write_text("private invalid JSON")
    elif case in {"object", "empty", "blank", "nonstring", "many", "long"}:
        value = {"object": {}, "empty": [], "blank": [" "], "nonstring": [1], "many": ["x"] * 65,
                 "long": ["x" * 50001]}[case]
        paths["input.json"].write_text(json.dumps(value))
    elif case == "missing-audio":
        paths["reference.wav"].unlink()
    elif case == "audio-directory":
        paths["reference.wav"].unlink()
        paths["reference.wav"].mkdir()
    elif case == "blank-transcript":
        paths["transcript.txt"].write_text(" \n ")
    elif case == "missing-transcript":
        paths["transcript.txt"].unlink()
    elif case == "missing-input":
        paths["input.json"].unlink()
    elif case == "invalid-utf8":
        paths["transcript.txt"].write_bytes(b"\xff")
    elif case in {"language", "seed", "negative-seed", "empty-checkpoint"}:
        key, value = {"language": ("--language", "private-language"), "seed": ("--seed", "private-seed"),
                      "negative-seed": ("--seed", "-1"), "empty-checkpoint": ("--checkpoint", "")}[case]
        argv[argv.index(key) + 1] = value
    elif case in {"settings", "unknown-setting", "duration-batch", "duration-long"}:
        config = tmp_path / "settings.json"
        value = {"settings": '{"speed":NaN}', "unknown-setting": '{"private":1}',
                 "duration-batch": '{"duration":2.0}', "duration-long": '{"duration":2.0}'}[case]
        config.write_text(value)
        argv += ["--settings", str(config)]
        if case == "duration-long":
            paths["input.json"].write_text(json.dumps(["x" * 1001]))
    elif case == "existing-output":
        paths["output"].mkdir()
    elif case == "existing-file":
        paths["output"].write_text("keep")
    elif case == "dangling-output":
        paths["output"].symlink_to(tmp_path / "missing-target", target_is_directory=True)
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("runtime must not be constructed")

    assert module.main(argv, runtime_factory=forbidden) != 0
    assert called == []
    assert not list(tmp_path.glob(".omnivoice-*"))
    output = capsys.readouterr()
    assert "private" not in output.out + output.err and str(tmp_path) not in output.out + output.err
    if case == "existing-output":
        assert list(paths["output"].iterdir()) == []
    elif case == "existing-file":
        assert paths["output"].read_text() == "keep"
    elif case == "dangling-output":
        assert paths["output"].is_symlink()
    else:
        assert not paths["output"].exists()


@pytest.mark.parametrize("case", ["runtime", "count", "rate", "empty", "odd", "wrong-type", "timing", "wav-write", "metadata-write", "interrupted"])
def test_failure_never_publishes_and_cleans_own_staging(tmp_path, monkeypatch, capsys, case):
    module = cli_module()
    argv, paths = arguments(tmp_path)
    unrelated = tmp_path / ".omnivoice-unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("keep")
    result = fake_result()
    if case == "count":
        result.pcm = result.pcm[:1]
    elif case == "rate":
        result.sample_rate = 16000
    elif case == "empty":
        result.pcm = (b"", b"12")
    elif case == "odd":
        result.pcm = (b"x", b"12")
    elif case == "wrong-type":
        result.pcm = ([1, 2], b"12")
    elif case == "timing":
        result.timings["load_seconds"] = float("nan")
    elif case == "wav-write":
        original = wave.open
        def broken(file, mode):
            if str(file).endswith("0001.wav"):
                raise OSError("private write failure")
            return original(file, mode)
        monkeypatch.setattr(wave, "open", broken)
    elif case == "metadata-write":
        original_write = Path.write_text
        def broken_write(path, *args, **kwargs):
            if path.name == "metadata.json":
                raise OSError("private metadata failure")
            return original_write(path, *args, **kwargs)
        monkeypatch.setattr(Path, "write_text", broken_write)

    class Runtime:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            if case in {"runtime", "interrupted"}:
                import logging
                import sys
                print("private stdout")
                print("private stderr", file=sys.stderr)
                logging.warning("private logging")
                raise RuntimeError("private exception") if case == "runtime" else KeyboardInterrupt()
            return result

    assert module.main(argv, runtime_factory=Runtime) != 0
    assert not paths["output"].exists()
    assert list(tmp_path.glob(".omnivoice-*")) == [unrelated]
    assert (unrelated / "keep").read_text() == "keep"
    captured = capsys.readouterr()
    assert "private" not in captured.out + captured.err and str(tmp_path) not in captured.out + captured.err
    assert captured.out.startswith("Error: ") and captured.err == ""


@pytest.mark.parametrize("occupied", [False, True])
def test_destination_created_during_generation_is_never_replaced(tmp_path, occupied, capsys):
    module = cli_module()
    argv, paths = arguments(tmp_path)
    class Runtime:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            paths["output"].mkdir()
            if occupied:
                (paths["output"] / "keep").write_text("untouched")
            return fake_result()
    assert module.main(argv, runtime_factory=Runtime) != 0
    assert sorted(p.name for p in paths["output"].iterdir()) == (["keep"] if occupied else [])
    assert not list(tmp_path.glob(".omnivoice-*"))


def test_cli_sanitizes_metadata_even_for_injected_runtime(tmp_path):
    module = cli_module()
    argv, paths = arguments(tmp_path)
    result = fake_result()
    result.provenance.update({"reference_text": "private transcript", "device": "/private/path", "nested": {"private": 1}})
    result.timings.update({"private path": "private value"})
    class Runtime:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            return result
    assert module.main(argv, runtime_factory=Runtime) == 0
    assert "private" not in (paths["output"] / "metadata.json").read_text()


@pytest.mark.parametrize("argv", [[], ["--unknown", "private-value"], ["--seed", "private-seed"]])
def test_parser_errors_are_class_only(argv, capsys):
    module = cli_module()
    assert module.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == "Error: ValueError\n" and captured.err == ""


def test_help_exits_without_model_factory(capsys):
    module = cli_module()
    assert module.main(["--help"], runtime_factory=lambda *a, **k: pytest.fail("no model")) == 0
    help_text = capsys.readouterr().out
    assert "--reference-text" in help_text
    assert "--optimization" in help_text and "experimental" in help_text.lower()


@pytest.mark.parametrize("optimization", [None, "none", "split-cfg"])
def test_cli_optimization_routes_and_publishes_sanitized_provenance(tmp_path, optimization):
    module = cli_module()
    argv, paths = arguments(tmp_path)
    selected = optimization or "none"
    calls = []
    result = fake_result()
    result.provenance.update(optimization=selected,
                             optimization_source_sha256="c" * 64 if selected == "split-cfg" else None)

    class Runtime:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def generate(self, texts, **kwargs):
            assert texts == ["Second input", "First input"]
            return result

    if optimization is not None:
        argv += ["--optimization", optimization]
    assert module.main(argv, runtime_factory=Runtime) == 0
    assert len(calls) == 1 and calls[0]["optimization"] == selected
    metadata = json.loads((paths["output"] / "metadata.json").read_text())
    assert metadata["provenance"]["optimization"] == selected
    assert metadata["provenance"]["optimization_source_sha256"] == result.provenance["optimization_source_sha256"]
    assert metadata["provenance"]["generation_num_step"] == 32
    assert metadata["provenance"]["generation_guidance_scale"] == 2.0


def test_cli_invalid_optimization_never_constructs_runtime(tmp_path, capsys):
    module = cli_module()
    argv, paths = arguments(tmp_path)
    assert module.main([*argv, "--optimization", "private-invalid"],
                       runtime_factory=lambda *a, **k: pytest.fail("must not load")) == 2
    assert capsys.readouterr().out == "Error: ValueError\n"
    assert not paths["output"].exists()
