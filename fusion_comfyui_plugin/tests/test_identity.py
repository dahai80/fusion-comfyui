import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestFusionIdentityLoader:
    def test_input_types(self):
        from nodes.identity import FusionIdentityLoader
        inputs = FusionIdentityLoader.INPUT_TYPES()
        assert "required" in inputs
        assert "model_name" in inputs["required"]
        assert "dtype" in inputs["required"]

    def test_return_types(self):
        from nodes.identity import FusionIdentityLoader
        assert FusionIdentityLoader.RETURN_TYPES == ("FUSION_IDENTITY_MODEL",)
        assert FusionIdentityLoader.FUNCTION == "load_identity"
        assert FusionIdentityLoader.CATEGORY == "Fusion-MLX/Identity"

    def test_load_identity(self):
        from nodes.identity import FusionIdentityLoader
        mock_pipeline = MagicMock()
        with patch.object(
            FusionIdentityLoader, "_load_pipeline",
            new_callable=AsyncMock, return_value=mock_pipeline,
        ), patch("core.lifecycle.FusionMemoryGuardian.purge_memory"):
            node = FusionIdentityLoader()
            result = node.load_identity("pulid_flux_v0.9.1", "float16")
        assert len(result) == 1
        assert result[0] is mock_pipeline


class TestFusionIdentityApply:
    def test_input_types(self):
        from nodes.identity import FusionIdentityApply
        inputs = FusionIdentityApply.INPUT_TYPES()
        assert "required" in inputs
        assert "identity_model" in inputs["required"]
        assert "image" in inputs["required"]
        assert "weight" in inputs["required"]

    def test_return_types(self):
        from nodes.identity import FusionIdentityApply
        assert FusionIdentityApply.RETURN_TYPES == ("FUSION_IDENTITY_EMBED",)
        assert FusionIdentityApply.FUNCTION == "apply_identity"

    def test_apply_success(self):
        from nodes.identity import FusionIdentityApply
        import mlx.core as mx
        mock_model = MagicMock()
        embed = mx.zeros((1, 32, 2048))
        mock_model.extract_id_embedding.return_value = embed
        mock_model.attn_processors = {}
        image = np.random.rand(1, 512, 512, 3).astype(np.float32)
        node = FusionIdentityApply()
        result = node.apply_identity(mock_model, image, weight=0.8, start_at=0.1, end_at=0.9)
        assert len(result) == 1
        assert result[0]["weight"] == 0.8
        assert result[0]["start_at"] == 0.1
        assert result[0]["end_at"] == 0.9

    def test_apply_no_face_raises(self):
        from nodes.identity import FusionIdentityApply
        mock_model = MagicMock()
        mock_model.extract_id_embedding.return_value = None
        image = np.random.rand(1, 64, 64, 3).astype(np.float32)
        node = FusionIdentityApply()
        with pytest.raises(RuntimeError, match="No face"):
            node.apply_identity(mock_model, image)


class TestFusionIdentityGenerate:
    def test_input_types(self):
        from nodes.identity import FusionIdentityGenerate
        inputs = FusionIdentityGenerate.INPUT_TYPES()
        assert "required" in inputs
        assert "pipeline" in inputs["required"]
        assert "identity_model" in inputs["required"]
        assert "reference_image" in inputs["required"]
        assert "prompt" in inputs["required"]

    def test_return_types(self):
        from nodes.identity import FusionIdentityGenerate
        assert FusionIdentityGenerate.RETURN_TYPES == ("IMAGE",)
        assert FusionIdentityGenerate.FUNCTION == "generate"

    def test_generate_success(self):
        from nodes.identity import FusionIdentityGenerate
        import mlx.core as mx
        from PIL import Image
        import io
        mock_pipeline = MagicMock()
        mock_identity = MagicMock()
        embed = mx.zeros((1, 32, 2048))
        mock_identity.extract_id_embedding.return_value = embed
        img = Image.new("RGB", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fake_bytes = [buf.getvalue()]
        with patch.object(
            FusionIdentityGenerate, "_generate_with_identity",
            new_callable=AsyncMock, return_value=fake_bytes,
        ), patch("core.lifecycle.FusionMemoryGuardian.purge_memory"):
            node = FusionIdentityGenerate()
            ref_image = np.random.rand(1, 256, 256, 3).astype(np.float32)
            result = node.generate(
                mock_pipeline, mock_identity, ref_image,
                prompt="a person", seed=42,
            )
        assert len(result) == 1
        assert result[0].ndim == 4
        mock_identity.inject_id.assert_called_once_with(embed)
        mock_identity.clear_id.assert_called_once()

    def test_generate_no_face_raises(self):
        from nodes.identity import FusionIdentityGenerate
        mock_pipeline = MagicMock()
        mock_identity = MagicMock()
        mock_identity.extract_id_embedding.return_value = None
        ref_image = np.random.rand(1, 64, 64, 3).astype(np.float32)
        with patch("core.lifecycle.FusionMemoryGuardian.purge_memory"):
            node = FusionIdentityGenerate()
            with pytest.raises(RuntimeError, match="No face"):
                node.generate(
                    mock_pipeline, mock_identity, ref_image,
                    prompt="test", seed=1,
                )
        mock_identity.clear_id.assert_called_once()


class TestImageToBGR:
    def test_float_input(self):
        from nodes.identity import _image_to_bgr
        arr = np.random.rand(1, 100, 100, 3).astype(np.float32)
        bgr = _image_to_bgr(arr)
        assert bgr.dtype == np.uint8
        assert bgr.ndim == 3

    def test_uint8_input(self):
        from nodes.identity import _image_to_bgr
        arr = (np.random.rand(1, 100, 100, 3) * 255).astype(np.uint8)
        bgr = _image_to_bgr(arr)
        assert bgr.dtype == np.uint8
        assert bgr.ndim == 3

    def test_rgba_input(self):
        from nodes.identity import _image_to_bgr
        arr = np.random.rand(1, 100, 100, 4).astype(np.float32)
        bgr = _image_to_bgr(arr)
        assert bgr.shape[2] == 3


class TestResolvePulidPath:
    def test_default_model(self):
        from nodes.identity import _resolve_pulid_path
        result = _resolve_pulid_path("pulid_flux_v0.9.1")
        assert "pulid_flux_v0.9.1" in result
