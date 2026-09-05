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
    def __init__(self, *, cuda_available=False, cuda_fails=False,
                 mps_available=False, mps_fails=False):
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available),
        )
        self._cuda_fails = cuda_fails
        self._mps_fails = mps_fails

    def zeros(self, *shape, device):
        if device == "cuda" and self._cuda_fails:
            raise RuntimeError("CUDA probe failed")
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


def test_accel_device_selects_cuda_after_a_real_probe(monkeypatch):
    from pipeline.model_store import accel_device

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(cuda_available=True))

    assert accel_device() == "cuda"


def test_accel_device_tries_mps_when_cuda_probe_fails(monkeypatch):
    from pipeline.model_store import accel_device

    monkeypatch.setitem(
        sys.modules, "torch",
        _FakeTorch(cuda_available=True, cuda_fails=True, mps_available=True),
    )

    assert accel_device() == "mps"


def test_accel_device_returns_cpu_when_every_accelerator_probe_fails(monkeypatch):
    from pipeline.model_store import accel_device

    monkeypatch.setitem(
        sys.modules, "torch",
        _FakeTorch(cuda_available=True, cuda_fails=True,
                   mps_available=True, mps_fails=True),
    )

    assert accel_device() is None


def test_accel_device_ignores_a_broken_cuda_probe_and_uses_mps(monkeypatch):
    from pipeline.model_store import accel_device

    class BrokenCuda(_FakeTorch):
        def __init__(self):
            super().__init__(mps_available=True)
            self.cuda.is_available = lambda: (_ for _ in ()).throw(
                RuntimeError("CUDA driver unavailable")
            )

    monkeypatch.setitem(sys.modules, "torch", BrokenCuda())

    assert accel_device() == "mps"


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


def test_accel_device_does_not_probe_mps_when_cuda_succeeds(monkeypatch):
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

    assert model_store.accel_device() == "cuda"
    assert fake_torch.mps_checked is False


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
