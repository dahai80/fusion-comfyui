import pytest
from pathlib import Path

from fusion_comfyui.server.static_files import (
    init_output_dir, get_output_dir, get_frontend_dir, view_file,
)


class TestStaticFiles:
    def test_init_output_dir(self, tmp_path):
        import fusion_comfyui.server.static_files as mod
        mod._output_dir = tmp_path / "out"
        init_output_dir()
        assert (tmp_path / "out").exists()

    def test_get_output_dir(self, tmp_path):
        import fusion_comfyui.server.static_files as mod
        mod._output_dir = tmp_path
        assert get_output_dir() == tmp_path

    def test_get_frontend_dir_env(self, tmp_path):
        import fusion_comfyui.server.static_files as mod
        mod._frontend_dir = tmp_path
        assert get_frontend_dir() == tmp_path

    def test_get_frontend_dir_bundled(self, tmp_path):
        import fusion_comfyui.server.static_files as mod
        mod._frontend_dir = Path("/nonexistent")
        bundled = Path(__file__).parent.parent / "fusion_comfyui" / "frontend"
        result = get_frontend_dir()
        if bundled.exists():
            assert result == bundled
        else:
            assert result == Path("")

    def test_view_file_found(self, tmp_path):
        import fusion_comfyui.server.static_files as mod
        mod._output_dir = tmp_path
        (tmp_path / "test.png").write_bytes(b"img")
        resp = view_file("test.png")
        assert resp is not None

    def test_view_file_not_found(self, tmp_path):
        import fusion_comfyui.server.static_files as mod
        mod._output_dir = tmp_path
        with pytest.raises(Exception):
            view_file("missing.png")

    def test_view_file_with_subfolder(self, tmp_path):
        import fusion_comfyui.server.static_files as mod
        sub = tmp_path / "sub"
        sub.mkdir()
        mod._output_dir = tmp_path
        (sub / "deep.png").write_bytes(b"deep")
        resp = view_file("deep.png", subfolder="sub")
        assert resp is not None

    def test_view_file_rejects_traversal(self, tmp_path):
        import fusion_comfyui.server.static_files as mod
        from fastapi import HTTPException
        secret = tmp_path.parent / "secret.txt"
        secret.write_bytes(b"secret")
        mod._output_dir = tmp_path
        with pytest.raises(HTTPException) as exc:
            view_file("../secret.txt")
        assert exc.value.status_code == 404

    def test_get_frontend_dir_empty_env_falls_back(self, tmp_path):
        import fusion_comfyui.server.static_files as mod
        mod._frontend_dir = Path("")
        result = get_frontend_dir()
        bundled = Path(__file__).parent.parent / "fusion_comfyui" / "frontend"
        if bundled.exists():
            assert result == bundled
        else:
            assert result == Path("")
