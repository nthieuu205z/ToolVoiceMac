"""Quản lý các model tải từ Hugging Face về máy, có đo tiến trình.

Đo bằng tổng byte thật trong cache Hugging Face (bỏ qua symlink, nên không đếm trùng
`snapshots/` và `blobs/`). Cách này không phụ thuộc vào thư viện bên dưới có chịu phơi
ra thanh tiến trình hay không: `faster_whisper` chặn cứng thanh tiến trình bằng
`tqdm_class=disabled_tqdm`.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)
_accel_probe_lock = threading.Lock()
OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"
OMNIVOICE_MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"


@lru_cache(maxsize=1)
def accel_device() -> str | None:
    """Trả MPS sau khi phép tính Metal chạy được; None nghĩa là dùng CPU.

    Cờ is_available chưa đủ để biết kernel có hoạt động. Probe bằng một phép
    nhân ma trận nhỏ; model cụ thể vẫn phải qua kiểm tra riêng khi nạp/chạy.
    """
    # lru_cache không tự khóa phần thân khi hai request đầu tiên chạy đồng thời. MPS
    # có thể abort native runtime nếu hai thread cùng khởi tạo context, nên serialize
    # cả phép probe (sau đó kết quả vẫn được cache).
    with _accel_probe_lock:
        try:
            import torch
        except Exception as exc:
            # torch là phụ thuộc tùy chọn: edge/Gemini vẫn phải khởi động được trên
            # máy không cài extra OmniVoice.
            log.debug("Không có torch để dò thiết bị gia tốc: %s", exc)
            return None

        try:
            mps_available = torch.backends.mps.is_available()
        except Exception as exc:
            log.debug("Không dò được MPS: %s", exc)
            mps_available = False

        if mps_available:
            try:
                probe = torch.zeros(8, 8, device="mps")
                float((probe @ probe).sum().cpu())
                return "mps"
            except Exception as exc:
                log.warning("Có Apple Silicon nhưng bản torch này không chạy được Metal (%s). "
                            "Lùi về CPU.", exc)
        return None


@lru_cache(maxsize=1)
def uses_gpu() -> bool:
    """Có Metal trên Apple Silicon chạy thật được không.

    Giữ tên cũ cho tương thích; nguồn sự thật là `accel_device()` — xem chú thích ở đó
    về vì sao phải probe bằng phép tính thật thay vì hỏi `torch.*.is_available()`.
    """
    return accel_device() is not None


@dataclass(frozen=True)
class RepoSpec:
    """Một repo Hugging Face, kèm bộ file cần lấy. `patterns=None` nghĩa là lấy hết."""

    repo_id: str
    patterns: list[str] | None = None
    revision: str | None = None  # Exact commit hash; None follows the default revision.


@dataclass(frozen=True)
class ModelSpec:
    key: str            # định danh dùng trong API: "whisper" | "omnivoice"
    label: str          # hiển thị cho người dùng
    repos: list[RepoSpec] = field(default_factory=list)


# Đúng bộ file mà faster-whisper tải; giữ khớp để đo và tải cùng một tập.
_WHISPER_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]

_total_bytes_cache: dict[tuple[str, tuple[str, ...] | None, str | None], int] = {}


@dataclass(frozen=True)
class ModelInfo:
    key: str
    label: str
    ready: bool
    downloaded_bytes: int
    total_bytes: int

    @property
    def percent(self) -> float:
        if self.ready:
            return 100.0
        if self.total_bytes <= 0:
            return 0.0
        return round(min(100.0, self.downloaded_bytes / self.total_bytes * 100), 1)


# ─── định nghĩa model ───────────────────────────────────────────────

def whisper_spec(name: str) -> ModelSpec:
    from faster_whisper.utils import _MODELS

    repo = name if "/" in name else _MODELS.get(name)
    if repo is None:
        raise ValueError(f"Model Whisper không hợp lệ: {name}. Chọn một trong: {', '.join(_MODELS)}")
    return ModelSpec(
        key="whisper",
        label=f"Nhận diện giọng nói — Whisper '{name}'",
        repos=[RepoSpec(repo, _WHISPER_PATTERNS)],
    )


def omnivoice_spec() -> ModelSpec:
    """OmniVoice tự chứa trong một repo (~3,3 GB)."""
    dev = accel_device()
    label_dev = dev.upper() if dev else "CPU"
    return ModelSpec(
        key="omnivoice",
        label=f"Giọng nhân bản — OmniVoice ({label_dev})",
        repos=[RepoSpec(OMNIVOICE_MODEL_ID, revision=OMNIVOICE_MODEL_REVISION)],
    )


# ─── đo và tải ──────────────────────────────────────────────────────

def _cache_dir(repo_id: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def _active_snapshots(repo_dir: Path, revision: str | None = None) -> list[Path]:
    """Snapshot đã ghim; nếu không ghim thì ưu tiên `refs/main`.

    Cache Hugging Face có thể giữ nhiều revision cũ trong cùng một repo. Chỉ cộng
    revision đang dùng; nếu cộng tất cả, cache cũ hoặc hai nhánh ONNX/PyTorch sẽ làm
    thanh tiến trình nhảy quá 100% và có thể báo xong dù revision hiện tại còn thiếu file.
    """
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return []

    if revision is not None:
        snapshot = snapshots / revision
        return [snapshot] if snapshot.is_dir() else []

    refs = repo_dir / "refs"
    revisions: list[str] = []
    main_ref = refs / "main"
    if main_ref.is_file():
        try:
            revision = main_ref.read_text(encoding="utf-8").strip()
        except OSError:
            revision = ""
        if revision:
            revisions.append(revision)

    active = [snapshots / revision for revision in revisions
              if (snapshots / revision).is_dir()]
    if active:
        return active

    # Test fixtures and detached revisions may not have a refs/main file. Chọn snapshot
    # mới nhất thay vì cộng cả cache cũ; snapshot_download mặc định cũng chỉ dùng một
    # revision tại một thời điểm.
    try:
        candidates = [p for p in snapshots.iterdir() if p.is_dir()]
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[:1]
    except OSError:
        return []


def _file_is_complete(path: Path) -> bool:
    """Snapshot file có thể đọc đầy đủ, kể cả trên filesystem không hỗ trợ symlink."""
    if path.is_symlink():
        return path.exists()
    return path.is_file()


def _cached_snapshot_files(snapshot: Path, repo: RepoSpec) -> dict[Path, int]:
    """Các file khớp pattern, lập chỉ mục theo blob thật để không cộng trùng."""
    files: dict[Path, int] = {}
    for path in snapshot.rglob("*"):
        if not _file_is_complete(path):
            continue
        relative = path.relative_to(snapshot).as_posix()
        if repo.patterns is not None and not any(
            fnmatch.fnmatch(relative, pattern) for pattern in repo.patterns
        ):
            continue
        try:
            blob = path.resolve(strict=True)
            files[blob] = blob.stat().st_size
        except OSError:
            continue
    return files


def _incomplete_bytes(repo_dir: Path, repo: RepoSpec) -> int:
    """Byte đang tải dở nếu không cần đoán nhánh file từ hash opaque."""
    total = 0
    for path in repo_dir.joinpath("blobs").glob("*.incomplete"):
        try:
            if repo.patterns is not None or repo.revision is not None:
                # Tên `.incomplete` chỉ là etag/hash, không chứa rfilename từ xa.
                # Không thể biết nó thuộc update/ hay onnx_update/; bỏ qua để không
                # làm progress ăn nhầm file hoặc revision khác.
                continue
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _downloaded_bytes(repo: RepoSpec) -> int:
    """Byte của file khớp `RepoSpec.patterns` đã có trong snapshot hiện hành.

    Đếm từ snapshot thay vì cộng toàn bộ `blobs/`: cache có thể chứa revision cũ, nhánh
    ONNX và nhánh PyTorch cùng lúc. File symlink được quy về blob thật và mỗi blob chỉ
    tính một lần; file thật trong snapshots cũng được tính.

    File `.incomplete` không gắn được chắc chắn với tên file từ xa nếu chỉ nhìn cache.
    Chỉ cộng file dở khi repo không lọc pattern và không ghim revision; các repo khác
    chỉ báo file đã hoàn tất để không tính nhầm file/revision khác.
    """
    repo_dir = _cache_dir(repo.repo_id)
    if not repo_dir.is_dir():
        return 0

    counted: dict[Path, int] = {}
    for snapshot in _active_snapshots(repo_dir, repo.revision):
        counted.update(_cached_snapshot_files(snapshot, repo))
    total = sum(counted.values())

    total += _incomplete_bytes(repo_dir, repo)
    return total


def _total_bytes(repo: RepoSpec) -> int:
    """Hỏi Hugging Face tổng dung lượng cần tải. Cần mạng; trả 0 nếu hỏi không được."""
    cache_key = (
        repo.repo_id,
        tuple(sorted(repo.patterns)) if repo.patterns is not None else None,
        repo.revision,
    )
    if cache_key in _total_bytes_cache:
        return _total_bytes_cache[cache_key]
    try:
        from huggingface_hub import HfApi

        revision_args = {"revision": repo.revision} if repo.revision is not None else {}
        info = HfApi().model_info(repo.repo_id, files_metadata=True, **revision_args)
        files = info.siblings
        if repo.patterns is not None:
            files = [s for s in files
                     if any(fnmatch.fnmatch(s.rfilename, p) for p in repo.patterns)]
        total = sum(s.size or 0 for s in files)
    except Exception as exc:
        log.warning("Không hỏi được dung lượng %s: %s", repo.repo_id, exc)
        return 0

    _total_bytes_cache[cache_key] = total
    return total


def _repo_ready(repo: RepoSpec) -> bool:
    from huggingface_hub import snapshot_download

    try:
        revision_args = {"revision": repo.revision} if repo.revision is not None else {}
        snapshot_download(repo.repo_id, allow_patterns=repo.patterns, local_files_only=True, **revision_args)
        return True
    except Exception:
        return False


def is_ready(spec: ModelSpec) -> bool:
    """Model đã đủ file trên đĩa chưa — kiểm tra offline, không đụng mạng."""
    return all(_repo_ready(r) for r in spec.repos)


def describe(spec: ModelSpec) -> ModelInfo:
    ready = is_ready(spec)
    downloaded = sum(_downloaded_bytes(r) for r in spec.repos)
    total = downloaded if ready else sum(_total_bytes(r) for r in spec.repos)
    return ModelInfo(key=spec.key, label=spec.label, ready=ready,
                     downloaded_bytes=downloaded, total_bytes=total)


def download(spec: ModelSpec) -> None:
    """Tải mọi repo của model. Chạy đồng bộ — người gọi tự đặt nó vào luồng nền."""
    from huggingface_hub import snapshot_download

    for repo in spec.repos:
        log.info("Đang tải %s", repo.repo_id)
        revision_args = {"revision": repo.revision} if repo.revision is not None else {}
        snapshot_download(repo.repo_id, allow_patterns=repo.patterns, **revision_args)
