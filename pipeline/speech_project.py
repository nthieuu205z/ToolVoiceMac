"""Bounded, sequential speech parts with an atomic manifest and complete ZIP."""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from .errors import InvalidTextError, JobCancelledError, PipelineError
from .models import CancelFn, ProgressFn, never_cancel, noop_progress
from .text_to_voice import TextToVoiceOptions, chunk_text, run_text_to_voice

MAX_PART_CHARACTERS = 5_000
MAX_PROJECT_CHARACTERS = 200_000
MANIFEST_FILENAME = "project.json"


def safe_download_stem(value: str, *, limit: int = 40) -> str:
    cleaned = re.sub(r"[^\w .-]+", "", value[:limit], flags=re.UNICODE)
    return " ".join(cleaned.split()).strip(" .-") or "speech"


def split_project_text(text: str, *, max_characters: int = MAX_PART_CHARACTERS) -> list[str]:
    """Partition without trimming/reordering; even whitespace remains in text exports."""
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    parts = []
    offset = 0
    while offset < len(text):
        window = text[offset:offset + max_characters]
        end = len(window)
        if offset + end < len(text):
            paragraphs = list(re.finditer(r"\n[ \t]*\n", window))
            sentences = list(re.finditer(r"[.!?。！？][ \t\n]*", window))
            whitespace = list(re.finditer(r"\s+", window))
            boundaries = paragraphs or sentences or whitespace
            if boundaries:
                end = boundaries[-1].end()
        parts.append(text[offset:offset + end])
        offset += end
    return parts


def _write_manifest(root: Path, manifest: dict) -> None:
    """Readers observe either the previous or the next full manifest."""
    temporary = root / ".project.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(root / MANIFEST_FILENAME)


def read_project_manifest(root: Path) -> dict:
    path = root / MANIFEST_FILENAME
    if path.is_symlink() or path.resolve().parent != root.resolve() or path.stat().st_size > 2_000_000:
        raise ValueError("Unsafe project manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1 or not isinstance(manifest.get("parts"), list):
        raise ValueError("Invalid project manifest")
    _validate_progress_record(manifest, {"queued", "running", "done", "error", "cancelled"})
    for field in ("name", "voice_id", "language"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Invalid project {field}")
    if len(manifest["name"]) > 80 or manifest["language"] not in {"vi-VN", "en-US"}:
        raise ValueError("Invalid project name or language")
    _require_count(manifest, "total_characters", minimum=1, maximum=MAX_PROJECT_CHARACTERS)
    _require_count(manifest, "total_parts", minimum=1, maximum=manifest["total_characters"])
    _require_count(manifest, "completed_parts", maximum=manifest["total_parts"])
    if len(manifest["parts"]) != manifest["total_parts"]:
        raise ValueError("Invalid project part count")
    for index, part in enumerate(manifest["parts"], 1):
        _validate_progress_record(part, {"pending", "running", "done", "error", "cancelled"})
        _require_count(part, "index", minimum=index, maximum=index)
        _require_count(part, "characters", minimum=1, maximum=MAX_PART_CHARACTERS)
        if not isinstance(part.get("title"), str) or not part["title"] or len(part["title"]) > 40:
            raise ValueError("Invalid project part title")
        if not isinstance(part.get("files"), dict) or set(part["files"]) not in ({"text"}, {"text", "wav", "mp3"}):
            raise ValueError("Invalid project part")
        for kind, name in part["files"].items():
            if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."} or "\\" in name:
                raise ValueError("Unsafe project filename")
            extension = "txt" if kind == "text" else kind
            if not name.startswith(f"{index:03d}-") or not name.endswith(f".{extension}"):
                raise ValueError("Invalid numbered project filename")
    for total, part_field in (("total_characters", "characters"), ("attempted_count", "attempted_count"), ("spoken_count", "spoken_count")):
        if manifest[total] != sum(part[part_field] for part in manifest["parts"]):
            raise ValueError("Inconsistent project progress counts")
    if manifest["completed_parts"] != sum(part["status"] == "done" for part in manifest["parts"]):
        raise ValueError("Inconsistent completed part count")
    if manifest["status"] == "done" and manifest["completed_parts"] != manifest["total_parts"]:
        raise ValueError("Incomplete project marked done")
    return manifest


def _require_count(record: dict, key: str, *, minimum: int = 0, maximum: int = MAX_PROJECT_CHARACTERS) -> None:
    value = record.get(key)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid project {key}")


def _validate_progress_record(record: object, statuses: set[str]) -> None:
    if not isinstance(record, dict) or not isinstance(record.get("status"), str) or record["status"] not in statuses:
        raise ValueError("Invalid project progress status")
    _require_count(record, "attempted_count")
    _require_count(record, "spoken_count", maximum=record["attempted_count"])
    if "error" in record and not isinstance(record["error"], str):
        raise ValueError("Invalid project error message")


def initialize_project(text: str, root: Path, options: TextToVoiceOptions, *, name: str) -> dict:
    parts = split_project_text(text)
    if not text.strip():
        raise InvalidTextError()
    if len(text) > MAX_PROJECT_CHARACTERS:
        raise PipelineError(user_message="Nội dung vượt quá giới hạn 200.000 ký tự.")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, part in enumerate(parts, 1):
        title = " ".join(part.split())[:40] or "Whitespace"
        stem = f"{index:03d}-{safe_download_stem(title)}"
        records.append({
            "index": index, "title": title, "characters": len(part), "status": "pending",
            "attempted_count": 0, "spoken_count": 0,
            "files": {extension: f"{stem}.{extension}" for extension in ("text", "wav", "mp3")},
        })
        records[-1]["files"]["text"] = f"{stem}.txt"
    manifest = {
        "version": 1, "name": name, "status": "queued", "voice_id": options.voice_id,
        "language": options.language, "total_characters": len(text), "total_parts": len(parts),
        "completed_parts": 0, "attempted_count": 0, "spoken_count": 0, "parts": records,
    }
    _write_manifest(root, manifest)
    return manifest


def fail_project(root: Path, status: str, message: str) -> None:
    """Also used when queue/backend/final publication fails outside this runner."""
    for name in ("project.zip", ".project.zip.tmp"):
        (root / name).unlink(missing_ok=True)
    manifest = read_project_manifest(root)
    manifest.update(status=status, error=message)
    for part in manifest["parts"]:
        if part["status"] == "running":
            part.update(status=status, error=message)
    _write_manifest(root, manifest)


@dataclass
class SpeechProjectResult:
    zip_path: str
    filename: str
    attempted_count: int
    spoken_count: int
    warnings: list[str] = field(default_factory=list)


def run_speech_project(
    synthesizer, text: str, workdir: Path, options: TextToVoiceOptions, *, name: str,
    progress: ProgressFn = noop_progress, should_cancel: CancelFn = never_cancel,
) -> SpeechProjectResult:
    root = Path(workdir)
    manifest = initialize_project(text, root, options, name=name)
    parts = split_project_text(text)
    output = root / "project_parts"
    output.mkdir(exist_ok=True)
    archive_path = root / "project.zip"
    temporary_archive = root / ".project.zip.tmp"
    archive_path.unlink(missing_ok=True)
    warnings = []

    def check_cancel():
        if should_cancel():
            raise JobCancelledError()

    try:
        check_cancel()
        manifest["status"] = "running"
        _write_manifest(root, manifest)
        progress("prepare", 1.0, f"Đã chuẩn bị {len(parts)} phần")
        for position, (text_part, record) in enumerate(zip(parts, manifest["parts"])):
            check_cancel()
            record["status"] = "running"
            _write_manifest(root, manifest)
            (output / record["files"]["text"]).write_text(text_part, encoding="utf-8")
            part_dir = output / f"part-{position + 1:03d}"
            part_dir.mkdir(exist_ok=True)

            def part_progress(stage, fraction, message):
                # Each short runner completes its own export. Keep the parent stage
                # in synthesize until every part finishes to avoid an early 100%.
                weights = {"prepare": (0, .05), "synthesize": (.05, .85), "assemble": (.9, .05), "export": (.95, .05)}
                begin, span = weights.get(stage, (0, 1))
                progress("synthesize", (position + begin + fraction * span) / len(parts),
                         f"Phần {position + 1}/{len(parts)} · {message}")

            if text_part.strip():
                result = run_text_to_voice(
                    synthesizer, text_part, part_dir, replace(options, max_characters=MAX_PART_CHARACTERS),
                    progress=part_progress, should_cancel=should_cancel,
                )
                record.update(attempted_count=result.attempted_count, spoken_count=result.spoken_count)
                manifest["attempted_count"] += result.attempted_count
                manifest["spoken_count"] += result.spoken_count
                warnings.extend(result.warnings)
                _write_manifest(root, manifest)
                expected = len(chunk_text(text_part, options.language, max_chars=options.max_chunk_characters))
                if result.attempted_count != expected or result.spoken_count != expected:
                    raise PipelineError(user_message=f"Phần {position + 1} thiếu giọng đọc; dự án chưa được xuất.")
                for extension, source in (("wav", result.wav_path), ("mp3", result.mp3_path)):
                    Path(source).replace(output / record["files"][extension])
            else:
                # A >5k whitespace run has no speech, but its exact text belongs in
                # the archive. Explicitly omit audio filenames for such a part.
                record["files"] = {"text": record["files"]["text"]}
            part_dir.rmdir()
            check_cancel()
            record["status"] = "done"
            manifest["completed_parts"] += 1
            _write_manifest(root, manifest)

        progress("assemble", 1.0, "Đã tạo đủ giọng đọc cho mọi phần")
        check_cancel()
        progress("export", 0.0, "Đang đóng gói dự án")
        complete = {**manifest, "status": "done"}
        folder = safe_download_stem(name)
        # ZipFile.write streams file contents; no project-wide audio array exists.
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for record in manifest["parts"]:
                check_cancel()
                for filename in record["files"].values():
                    archive.write(output / filename, f"{folder}/{filename}")
            archive.writestr(f"{folder}/manifest.json", json.dumps(complete, ensure_ascii=False, indent=2))
        check_cancel()
        temporary_archive.replace(archive_path)
        _write_manifest(root, complete)
        check_cancel()
        progress("export", 1.0, "Đã đóng gói toàn bộ dự án")
        return SpeechProjectResult(str(archive_path), f"{folder}.zip", manifest["attempted_count"], manifest["spoken_count"], warnings)
    except Exception as exc:
        status = "cancelled" if isinstance(exc, JobCancelledError) else "error"
        fail_project(root, status, getattr(exc, "user_message", "Không thể hoàn tất dự án."))
        raise
