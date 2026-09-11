"""Kiểm tra dò thiết bị gia tốc mà không đụng model hay mạng."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


class _ProbeValue:
    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def cpu(self):
        return self

    def __float__(self):
        return 0.0


class _FakeTorch:
    def __init__(self, *, cuda_available=False,
                 mps_available=False, mps_fails=False):
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available),
        )
        self._mps_fails = mps_fails

    def zeros(self, *shape, device):
        if device == "mps" and self._mps_fails:
            raise RuntimeError("MPS probe failed")
        return _ProbeValue()


@pytest.fixture(autouse=True)
def _clear_device_caches():
    from pipeline import model_store

    def clear_cache(function):
        cache_clear = getattr(function, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    clear_cache(model_store.accel_device)
    clear_cache(model_store.uses_gpu)
    model_store._total_bytes_cache.clear()
    yield
    clear_cache(model_store.accel_device)
    clear_cache(model_store.uses_gpu)
    model_store._total_bytes_cache.clear()


def test_accel_device_falls_back_to_cpu_when_torch_is_missing(monkeypatch):
    from pipeline.model_store import accel_device

    monkeypatch.setitem(sys.modules, "torch", None)

    assert accel_device() is None


def test_accel_device_falls_back_when_torch_import_is_broken(monkeypatch):
    import builtins

    from pipeline.model_store import accel_device

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError("libtorch không khởi động được")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)

    assert accel_device() is None


def test_accel_device_uses_cpu_when_only_non_mps_acceleration_is_available(monkeypatch):
    from pipeline.model_store import accel_device

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(cuda_available=True))

    assert accel_device() is None


def test_accel_device_selects_mps_after_a_real_probe(monkeypatch):
    from pipeline.model_store import accel_device

    monkeypatch.setitem(
        sys.modules, "torch",
        _FakeTorch(mps_available=True),
    )

    assert accel_device() == "mps"


def test_accel_device_returns_cpu_when_mps_computation_fails(monkeypatch):
    from pipeline.model_store import accel_device

    monkeypatch.setitem(
        sys.modules, "torch",
        _FakeTorch(mps_available=True, mps_fails=True),
    )

    assert accel_device() is None


def test_accel_device_uses_cpu_when_mps_availability_check_fails(monkeypatch):
    from pipeline.model_store import accel_device

    class BrokenMps(_FakeTorch):
        def __init__(self):
            super().__init__(mps_available=True)
            self.backends.mps.is_available = lambda: (_ for _ in ()).throw(
                RuntimeError("Metal unavailable")
            )

    monkeypatch.setitem(sys.modules, "torch", BrokenMps())

    assert accel_device() is None


def test_accel_device_probe_is_serialized(monkeypatch):
    from pipeline import model_store

    class ProbeTorch(_FakeTorch):
        active = 0
        maximum = 0

        def zeros(self, *shape, device):
            type(self).active += 1
            type(self).maximum = max(type(self).maximum, type(self).active)
            try:
                return super().zeros(*shape, device=device)
            finally:
                type(self).active -= 1

    monkeypatch.setitem(sys.modules, "torch", ProbeTorch(mps_available=True))

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda _: model_store.accel_device(), range(2)))
    assert result == ["mps", "mps"]
    assert ProbeTorch.maximum == 1


def test_accel_device_prefers_mps_even_when_other_acceleration_is_available(monkeypatch):
    from pipeline import model_store

    class CudaTorch(_FakeTorch):
        def __init__(self):
            super().__init__(cuda_available=True, mps_available=True)
            self.mps_checked = False
            self.backends.mps.is_available = self._mps_is_available

        def _mps_is_available(self):
            self.mps_checked = True
            return True

    fake_torch = CudaTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert model_store.accel_device() == "mps"
    assert fake_torch.mps_checked is True


def test_uses_gpu_matches_the_successful_device_probe(monkeypatch):
    from pipeline.model_store import accel_device, uses_gpu

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(mps_available=True))

    assert accel_device() == "mps"
    assert uses_gpu() is True


def test_omnivoice_spec_uses_the_single_model_repository():
    from pipeline import model_store

    spec = model_store.omnivoice_spec()

    assert spec.key == "omnivoice"
    assert [repo.repo_id for repo in spec.repos] == ["k2-fsa/OmniVoice"]
    assert getattr(spec.repos[0], "revision", None) == "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"


def test_downloaded_bytes_only_counts_files_matching_repo_patterns(monkeypatch, tmp_path):
    from pipeline import model_store

    repo_dir = tmp_path / "models--repo"
    snapshot = repo_dir / "snapshots" / "revision"
    blobs = repo_dir / "blobs"

    update_blob = blobs / "update-hash"
    onnx_blob = blobs / "onnx-hash"
    stale_blob = blobs / "stale-hash"
    update_blob.parent.mkdir(parents=True)
    update_blob.write_bytes(b"u" * 23)
    onnx_blob.write_bytes(b"o" * 31)
    stale_blob.write_bytes(b"s" * 47)

    snapshot.mkdir(parents=True)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("revision\n", encoding="utf-8")
    for relative, target, fallback in (
        ("update/model.safetensors", update_blob, b"u" * 23),
        ("onnx_update/model.onnx", onnx_blob, b"o" * 31),
        ("stale.bin", stale_blob, b"s" * 47),
    ):
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(target)
        except OSError:
            path.write_bytes(fallback)

    monkeypatch.setattr(model_store, "_cache_dir", lambda repo_id: repo_dir)

    assert model_store._downloaded_bytes(
        model_store.RepoSpec("repo", ["update/*"])
    ) == 23
    assert model_store._downloaded_bytes(
        model_store.RepoSpec("repo", ["onnx_update/*"])
    ) == 31


def test_downloaded_bytes_uses_only_the_active_revision(monkeypatch, tmp_path):
    from pipeline import model_store

    repo_dir = tmp_path / "models--repo"
    active = repo_dir / "snapshots" / "active"
    stale = repo_dir / "snapshots" / "stale"
    active.mkdir(parents=True)
    stale.mkdir(parents=True)
    (active / "model.bin").write_bytes(b"a" * 11)
    (stale / "model.bin").write_bytes(b"s" * 97)
    refs = repo_dir / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("active\n", encoding="utf-8")

    monkeypatch.setattr(model_store, "_cache_dir", lambda repo_id: repo_dir)

    assert model_store._downloaded_bytes(
        model_store.RepoSpec("repo", ["model.bin"])
    ) == 11


def test_downloaded_bytes_counts_incomplete_files_only_for_unfiltered_repos(
    monkeypatch, tmp_path,
):
    from pipeline import model_store

    repo_dir = tmp_path / "models--repo"
    incomplete = repo_dir / "blobs" / "unknown.incomplete"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_bytes(b"x" * 17)
    monkeypatch.setattr(model_store, "_cache_dir", lambda repo_id: repo_dir)

    assert model_store._downloaded_bytes(model_store.RepoSpec("repo", None)) == 17
    assert model_store._downloaded_bytes(
        model_store.RepoSpec("repo", ["update/*"])
    ) == 0


def test_downloaded_bytes_ignores_an_incomplete_file_for_a_filtered_repo(
    monkeypatch, tmp_path,
):
    from pipeline import model_store

    repo_dir = tmp_path / "models--repo"
    incomplete = repo_dir / "blobs" / "unknown.incomplete"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_bytes(b"x" * 19)
    monkeypatch.setattr(model_store, "_cache_dir", lambda repo_id: repo_dir)

    assert model_store._downloaded_bytes(
        model_store.RepoSpec("repo", ["update/*"])
    ) == 0


def test_total_bytes_cache_is_scoped_by_repo_patterns(monkeypatch):
    from pipeline import model_store

    calls = []

    class FakeApi:
        def model_info(self, repo_id, files_metadata):
            calls.append((repo_id, files_metadata))
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename="update/model.safetensors", size=23),
                SimpleNamespace(rfilename="onnx_update/model.onnx", size=31),
            ])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))

    update = model_store.RepoSpec("repo", ["update/*"])
    onnx = model_store.RepoSpec("repo", ["onnx_update/*"])

    assert model_store._total_bytes(update) == 23
    assert model_store._total_bytes(onnx) == 31
    assert len(calls) == 2


def test_pinned_progress_ignores_main_and_unattributed_partial_files(monkeypatch, tmp_path):
    from pipeline import model_store

    pinned = tmp_path / "snapshots" / "frozen"
    latest = tmp_path / "snapshots" / "latest"
    pinned.mkdir(parents=True)
    latest.mkdir(parents=True)
    (pinned / "model.bin").write_bytes(b"p" * 11)
    (latest / "model.bin").write_bytes(b"n" * 97)
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "main").write_text("latest")
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "unknown.incomplete").write_bytes(b"x" * 23)
    monkeypatch.setattr(model_store, "_cache_dir", lambda repo_id: tmp_path)

    assert model_store._downloaded_bytes(model_store.RepoSpec("repo", revision="frozen")) == 11
    assert model_store._downloaded_bytes(model_store.RepoSpec("repo", revision="missing")) == 0


def test_total_bytes_uses_requested_revision_and_separates_cached_sizes(monkeypatch):
    from pipeline import model_store

    class FakeApi:
        def model_info(self, repo_id, files_metadata, revision=None):
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename="model.bin", size={None: 97, "frozen": 11}[revision]),
            ])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    assert model_store._total_bytes(model_store.RepoSpec("repo")) == 97
    assert model_store._total_bytes(model_store.RepoSpec("repo", revision="frozen")) == 11


def test_pinned_readiness_does_not_accept_a_different_cached_revision(monkeypatch):
    from pipeline import model_store

    def snapshot_download(repo_id, *, allow_patterns, local_files_only, revision=None):
        if revision is not None:
            raise FileNotFoundError("Only main is cached")
        return "/cache/snapshots/main"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    assert model_store._repo_ready(model_store.RepoSpec("repo")) is True
    assert model_store._repo_ready(model_store.RepoSpec("repo", revision="frozen")) is False


def test_pinned_download_selects_the_frozen_revision(monkeypatch):
    from pipeline import model_store

    downloaded = []

    def snapshot_download(repo_id, *, allow_patterns, revision=None):
        downloaded.append((repo_id, revision, allow_patterns))
        return "/cache/snapshots/frozen"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr(model_store, "accel_device", lambda: "mps")
    model_store.download(model_store.omnivoice_spec())
    assert downloaded == [("k2-fsa/OmniVoice", "c5fdb5ccb189668d56333f77ba2629f4cd7535f4", None)]
