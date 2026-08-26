import numpy as np
from unittest.mock import MagicMock, patch


class TestCLIPTextEncode:
    def test_input_types(self):
        from nodes.conditioning import CLIPTextEncode
        inputs = CLIPTextEncode.INPUT_TYPES()
        assert "required" in inputs
        assert "text" in inputs["required"]

    def test_encode_with_fusion_clip(self):
        from nodes.conditioning import CLIPTextEncode
        mock_clip = MagicMock()
        mock_clip.encode_text.return_value = np.zeros((1, 77, 768))
        node = CLIPTextEncode()
        result = node.encode(mock_clip, "test prompt")
        assert result is not None

    def test_encode_with_non_fusion_clip(self):
        from nodes.conditioning import CLIPTextEncode
        mock_clip = MagicMock(spec=[])
        mock_clip.encode_from_tokens_scheduled = MagicMock(return_value=MagicMock(cond=np.zeros((1, 77, 768))))
        node = CLIPTextEncode()
        result = node.encode(mock_clip, "test prompt")
        assert result is not None

    def test_return_types(self):
        from nodes.conditioning import CLIPTextEncode
        assert "CONDITIONING" in CLIPTextEncode.RETURN_TYPES


class TestFusionTextEncoderNode:
    def test_input_types(self):
        from nodes.conditioning import FusionTextEncoderNode
        inputs = FusionTextEncoderNode.INPUT_TYPES()
        assert "required" in inputs

    def test_return_types(self):
        from nodes.conditioning import FusionTextEncoderNode
        assert "FUSION_COND" in FusionTextEncoderNode.RETURN_TYPES

    def test_encode_success(self):
        from nodes.conditioning import FusionTextEncoderNode
        from fusion_comfyui.core.wrappers import FusionCLIPWrapper
        mock_clip = MagicMock(spec=FusionCLIPWrapper)
        with patch("fusion_comfyui.core.async_utils.run_async", return_value={"embed": MagicMock()}):
            node = FusionTextEncoderNode()
            result = node.encode(mock_clip, "test prompt")
        assert result is not None

    def test_encode_fallback_on_failure(self):
        from nodes.conditioning import FusionTextEncoderNode
        mock_clip = MagicMock()
        mock_clip.encode_text.side_effect = RuntimeError("fail")
        mock_clip.encode_from_tokens_scheduled = MagicMock(return_value=MagicMock(cond=np.zeros((1, 77, 768))))
        node = FusionTextEncoderNode()
        result = node.encode(mock_clip, "test prompt")
        assert result is not None

    def test_encode_no_encode_text(self):
        from nodes.conditioning import FusionTextEncoderNode
        mock_clip = MagicMock(spec=["tokenize", "encode_from_tokens_scheduled"])
        mock_clip.tokenize.return_value = {"text": "hello"}
        mock_clip.encode_from_tokens_scheduled.return_value = {"text": "hello", "clip": mock_clip}
        with patch("fusion_comfyui.core.async_utils.run_async", return_value={"embed": MagicMock()}):
            node = FusionTextEncoderNode()
            result = node.encode(mock_clip, "test prompt")
        assert result is not None
