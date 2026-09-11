#!/usr/bin/env python3
"""Standalone offline OmniVoice CLI; the caller owns process/time/memory guards."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import ctypes
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
import wave

from pipeline.omnivoice_runtime import (
    DEFAULT_CHECKPOINT, SAMPLE_RATE, OmniVoiceRuntime, sanitize_provenance,
    validate_reference, validate_request,
)
from pipeline.omnivoice_settings import OmniVoiceSettings


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's normal diagnostics echo paths and arbitrary input values.
        raise ValueError("Invalid command line")


def _publish(stage: Path, destination: Path) -> None:
    """macOS exclusive atomic rename: even a racing empty directory is protected."""
    if sys.platform != "darwin":
        raise RuntimeError("Atomic publication requires macOS")
    rename = ctypes.CDLL(None, use_errno=True).renamex_np
    rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # RENAME_EXCL from the macOS SDK sys/stdio.h.
    if rename(os.fsencode(stage), os.fsencode(destination), 0x00000004) != 0:
        raise OSError(ctypes.get_errno(), "Atomic publication failed")


def _run(args, runtime_factory) -> int:
    destination = args.output_dir.absolute()
    if os.path.lexists(destination):
        raise FileExistsError("Output already exists")
    if not destination.parent.is_dir():
        raise FileNotFoundError("Output parent must exist")
    texts = json.loads(args.input.read_text(encoding="utf-8"))
    transcript = args.reference_text.read_text(encoding="utf-8")
    settings = OmniVoiceSettings.model_validate_json(args.settings.read_text(encoding="utf-8")) if args.settings else OmniVoiceSettings()
    validate_reference(args.reference_audio, transcript)
    language = validate_request(texts, args.language, args.seed, settings)
    if not args.checkpoint.strip():
        raise ValueError("Explicit checkpoint required")
    # Upstream diagnostics may include input text or local paths. The standalone
    # CLI reports only sanitized metadata; library callers still receive errors.
    previous_logging = logging.root.manager.disable
    try:
        logging.disable(sys.maxsize)
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            runtime = runtime_factory(args.reference_audio, transcript, settings=settings,
                                      checkpoint=args.checkpoint, optimization=args.optimization)
            result = runtime.generate(texts, language=language.code, seed=args.seed)
    finally:
        logging.disable(previous_logging)
    if result.sample_rate != SAMPLE_RATE or not isinstance(result.pcm, tuple) or len(result.pcm) != len(texts):
        raise ValueError("Invalid runtime result")
    if any(not isinstance(pcm, bytes) or not pcm or len(pcm) % 2 for pcm in result.pcm):
        raise ValueError("Invalid PCM")
    timings = {}
    for key in ("load_seconds", "prompt_seconds", "generate_seconds", "pcm_seconds", "total_seconds"):
        value = result.timings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid runtime timing")
        timings[key] = value
    start = perf_counter()
    stage = Path(tempfile.mkdtemp(prefix=".omnivoice-", dir=destination.parent))
    try:
        for index, pcm in enumerate(result.pcm):
            with wave.open(str(stage / f"{index:04d}.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(pcm)
        # WAV write time only; metadata serialization and final rename excluded.
        timings["write_seconds"] = perf_counter() - start
        metadata = {"count": len(result.pcm), "sample_rate": SAMPLE_RATE,
                    "timings": timings, "provenance": sanitize_provenance(result.provenance)}
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _publish(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"Wrote {len(result.pcm)} WAV files")
    return 0


def main(argv=None, *, runtime_factory=OmniVoiceRuntime) -> int:
    parser = _Parser(prog="run_omnivoice.py", description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 JSON list of ordered text strings")
    parser.add_argument("--reference-audio", required=True, type=Path, help="Explicit local reference audio")
    parser.add_argument("--reference-text", required=True, type=Path, help="UTF-8 reference transcript file")
    parser.add_argument("--language", required=True, help="vi-VN or en-US")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory with existing parent; never overwrite")
    parser.add_argument("--settings", type=Path, help="Optional OmniVoiceSettings JSON object")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Local checkpoint with audio_tokenizer, or offline cached repo ID")
    parser.add_argument("--optimization", choices=("none", "split-cfg"), default="none",
                        help="Optional experimental split-cfg; default none keeps original upstream behavior")
    parser.add_argument("--seed", default=1234, type=int)
    try:
        try:
            args = parser.parse_args(argv)
        except SystemExit as exit:
            return int(exit.code or 0)
        return _run(args, runtime_factory)
    except KeyboardInterrupt:
        print("Error: KeyboardInterrupt")
        return 130
    except Exception as error:
        # Use known class labels, not arbitrary upstream class names/messages.
        label = next((cls.__name__ for cls in (FileNotFoundError, FileExistsError, ValueError, OSError, RuntimeError)
                      if isinstance(error, cls)), "Exception")
        print(f"Error: {label}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
