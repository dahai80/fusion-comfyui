import numpy as np
from unittest.mock import patch


class TestFusionSubtitleOverlayNode:
    def test_input_types(self):
        from nodes.postprocess import FusionSubtitleOverlayNode
        inputs = FusionSubtitleOverlayNode.INPUT_TYPES()
        assert "required" in inputs

    def test_overlay(self):
        from nodes.postprocess import FusionSubtitleOverlayNode
        frames = np.random.randint(0, 255, (4, 512, 512, 3), dtype=np.uint8)
        node = FusionSubtitleOverlayNode()
        with patch("fusion_comfyui.core.bridge.to_numpy", return_value=frames):
            result = node.overlay(frames, "Hello", 10, 40, (255, 255, 255))
            assert result is not None
