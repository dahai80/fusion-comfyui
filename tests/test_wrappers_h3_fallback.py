import pytest


@pytest.fixture
def h3_model_installed(monkeypatch):
    import fusion_comfyui_plugin.core.wrappers as w
    monkeypatch.setattr(w, "_available_video_models", lambda: ["Wan2.2-5B", "minimax-h3"])
    return "minimax-h3"


class TestH3Fallback:
    def test_fallback_h3_to_minimax(self, h3_model_installed):
        from fusion_comfyui_plugin.core.wrappers import _fallback_model
        # available[0]=Wan2.2-5B, so without an explicit H3 branch this would
        # wrongly return Wan2.2-5B. The H3 branch must surface minimax-h3.
        assert _fallback_model("h3-14B") == "minimax-h3"
        assert _fallback_model("minimax-h3") == "minimax-h3"

    def test_fallback_minimax_exact_dir_short_circuits(self, h3_model_installed, monkeypatch):
        from fusion_comfyui_plugin.core.wrappers import _fallback_model
        import os
        orig_isdir = os.path.isdir
        def fake_isdir(p):
            if p.endswith("minimax-h3"):
                return True
            return orig_isdir(p)
        monkeypatch.setattr(os.path, "isdir", fake_isdir)
        assert _fallback_model("minimax-h3") == "minimax-h3"

    def test_fallback_h3_not_installed_falls_through(self, monkeypatch):
        import fusion_comfyui_plugin.core.wrappers as w
        monkeypatch.setattr(w, "_available_video_models", lambda: ["Wan2.2-5B"])
        from fusion_comfyui_plugin.core.wrappers import _fallback_model
        resolved = _fallback_model("h3-14B")
        assert resolved == "Wan2.2-5B", resolved
