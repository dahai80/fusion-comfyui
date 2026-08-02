import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestFusionLipsyncLoader:
    def test_input_types(self):
        from nodes.talking_head import FusionLipsyncLoader
        inputs = FusionLipsyncLoader.INPUT_TYPES()
        assert "required" in inputs
        assert "model_name" in inputs["required"]
        assert "dtype" in inputs["required"]

    def test_return_types(self):
        from nodes.talking_head import FusionLipsyncLoader
        assert FusionLipsyncLoader.RETURN_TYPES == ("FUSION_LIPSYNC_MODEL",)
        assert FusionLipsyncLoader.FUNCTION == "load_lipsync"
        assert FusionLipsyncLoader.CATEGORY == "Fusion-MLX/Talking-Head"

    def test_load_lipsync(self):
        from nodes.talking_head import FusionLipsyncLoader
        mock_pipeline = MagicMock()
        with patch.object(
            FusionLipsyncLoader, "_load_pipeline",
            new_callable=AsyncMock, return_value=mock_pipeline,
        ), patch("core.lifecycle.FusionMemoryGuardian.purge_memory"):
            node = FusionLipsyncLoader()
            result = node.load_lipsync("latentsync_unet", "float16")
        assert len(result) == 1
        assert result[0] is mock_pipeline


class TestFusionLipsyncApply:
    def test_input_types(self):
        from nodes.talking_head import FusionLipsyncApply
        inputs = FusionLipsyncApply.INPUT_TYPES()
        assert "required" in inputs
        assert "lipsync_model" in inputs["required"]
        assert "video_path" in inputs["required"]
        assert "audio_path" in inputs["required"]

    def test_return_types(self):
        from nodes.talking_head import FusionLipsyncApply
        assert FusionLipsyncApply.RETURN_TYPES == ("IMAGE",)
        assert FusionLipsyncApply.FUNCTION == "apply_lipsync"

    def test_missing_video_raises(self):
        from nodes.talking_head import FusionLipsyncApply
        node = FusionLipsyncApply()
        with pytest.raises(ValueError, match="video_path"):
            node.apply_lipsync(MagicMock(), "", "/tmp/audio.wav")

    def test_missing_audio_raises(self):
        from nodes.talking_head import FusionLipsyncApply
        node = FusionLipsyncApply()
        with pytest.raises(ValueError, match="audio_path"):
            node.apply_lipsync(MagicMock(), "/tmp/video.mp4", "")

    def test_apply_lipsync_success(self):
        from nodes.talking_head import FusionLipsyncApply
        mock_frames = np.random.rand(25, 256, 256, 3).astype(np.float32)
        with patch(
            "nodes.talking_head._video_frames_to_image_array",
            return_value=mock_frames,
        ), patch.object(
            FusionLipsyncApply, "_run_lipsync",
            new_callable=AsyncMock, return_value="/tmp/output.mp4",
        ), patch("core.lifecycle.FusionMemoryGuardian.purge_memory"):
            node = FusionLipsyncApply()
            result = node.apply_lipsync(
                MagicMock(), "/tmp/video.mp4", "/tmp/audio.wav",
            )
        assert len(result) == 1
        assert result[0].shape == (25, 256, 256, 3)

    def test_save_audio_to_temp(self):
        from nodes.talking_head import FusionLipsyncApply
        import os
        audio_np = np.zeros(24000, dtype=np.int16)
        path = FusionLipsyncApply._save_audio_to_temp((audio_np, 24000))
        assert path.endswith(".wav")
        assert os.path.exists(path)
        os.unlink(path)


class TestResolvePaths:
    def test_resolve_latentsync_path(self):
        from nodes.talking_head import _resolve_latentsync_path
        result = _resolve_latentsync_path("latentsync_unet")
        assert "latentsync_unet" in result

    def test_resolve_audio_path_file(self):
        from nodes.talking_head import _resolve_audio_path
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = _resolve_audio_path(tmp_path)
            assert result == tmp_path
        finally:
            os.unlink(tmp_path)
