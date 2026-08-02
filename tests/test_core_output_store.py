import os
import tempfile
import pytest

from fusion_comfyui.core.output_store import (
    init_store, get_store_dir, resolve_path, list_outputs, save_bytes,
)


class TestOutputStore:
    def test_init_and_dir(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        import fusion_comfyui.core.output_store as mod
        mod._store_dir = out
        init_store()
        assert out.exists()

    def test_save_and_resolve(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        import fusion_comfyui.core.output_store as mod
        mod._store_dir = out
        saved = save_bytes(b"hello", "test.bin")
        assert saved.exists()
        assert resolve_path("test.bin") is not None
        assert resolve_path("test.bin").read_bytes() == b"hello"

    def test_resolve_missing(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        import fusion_comfyui.core.output_store as mod
        mod._store_dir = out
        assert resolve_path("nope.bin") is None

    def test_list_outputs(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        import fusion_comfyui.core.output_store as mod
        mod._store_dir = out
        save_bytes(b"a", "a.png")
        save_bytes(b"b", "b.png")
        files = list_outputs()
        assert "a.png" in files
        assert "b.png" in files

    def test_subfolder(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        import fusion_comfyui.core.output_store as mod
        mod._store_dir = out
        save_bytes(b"sub", "sub.png", subfolder="inner")
        assert resolve_path("sub.png", subfolder="inner") is not None
        assert resolve_path("sub.png") is None

    def test_get_store_dir(self, tmp_path):
        out = tmp_path / "store"
        import fusion_comfyui.core.output_store as mod
        mod._store_dir = out
        assert get_store_dir() == out
