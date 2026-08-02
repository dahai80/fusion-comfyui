from unittest.mock import MagicMock, patch


class TestUNETLoader:
    def test_input_types(self):
        from nodes.loaders import UNETLoader
        inputs = UNETLoader.INPUT_TYPES()
        assert "required" in inputs

    def test_load_unet(self):
        from nodes.loaders import UNETLoader
        with patch("core.wrappers._resolve_model_path", return_value="/tmp/m"), \
             patch("core.wrappers._map_unet_name_to_model_name", return_value="test"), \
             patch("core.wrappers._infer_model_type", return_value="video"), \
             patch("core.wrappers.FusionModelWrapper") as MockWrap:
            MockWrap.return_value = MagicMock()
            node = UNETLoader()
            result = node.load_unet("test-model")
            assert result is not None


class TestCLIPLoader:
    def test_input_types(self):
        from nodes.loaders import CLIPLoader
        inputs = CLIPLoader.INPUT_TYPES()
        assert "required" in inputs

    def test_load_clip(self):
        from nodes.loaders import CLIPLoader
        with patch("core.wrappers._resolve_model_path", return_value="/tmp/m"), \
             patch("core.wrappers._map_clip_type_to_model_name", return_value="test"), \
             patch("core.wrappers.FusionCLIPWrapper") as MockWrap:
            MockWrap.return_value = MagicMock()
            node = CLIPLoader()
            result = node.load_clip("test-model", "wan")
            assert result is not None


class TestDualCLIPLoader:
    def test_input_types(self):
        from nodes.loaders import DualCLIPLoader
        inputs = DualCLIPLoader.INPUT_TYPES()
        assert "required" in inputs

    def test_load_clip(self):
        from nodes.loaders import DualCLIPLoader
        with patch("core.wrappers._resolve_model_path", return_value="/tmp/m"), \
             patch("core.wrappers._map_clip_type_to_model_name", return_value="test"), \
             patch("core.wrappers.FusionCLIPWrapper") as MockWrap:
            MockWrap.return_value = MagicMock()
            node = DualCLIPLoader()
            result = node.load_clip("test-model", "wan", "test-model2")
            assert result is not None


class TestVAELoader:
    def test_input_types(self):
        from nodes.loaders import VAELoader
        inputs = VAELoader.INPUT_TYPES()
        assert "required" in inputs

    def test_load_vae(self):
        from nodes.loaders import VAELoader
        with patch("core.wrappers._resolve_model_path", return_value="/tmp/m"), \
             patch("core.wrappers._map_vae_name_to_model_name", return_value="test"), \
             patch("core.wrappers.FusionVAEWrapper") as MockWrap:
            MockWrap.return_value = MagicMock()
            node = VAELoader()
            result = node.load_vae("test-model")
            assert result is not None


class TestCheckpointLoaderSimple:
    def test_input_types(self):
        from nodes.loaders import CheckpointLoaderSimple
        inputs = CheckpointLoaderSimple.INPUT_TYPES()
        assert "required" in inputs

    def test_load_checkpoint(self):
        from nodes.loaders import CheckpointLoaderSimple
        with patch("core.wrappers._resolve_model_path", return_value="/tmp/m"), \
             patch("core.wrappers._map_checkpoint_to_model_name", return_value="test"), \
             patch("core.wrappers._infer_model_type", return_value="video"), \
             patch("core.wrappers._map_clip_type_to_model_name", return_value="test"), \
             patch("core.wrappers._map_vae_name_to_model_name", return_value="test"), \
             patch("core.wrappers.FusionModelWrapper") as MockModel, \
             patch("core.wrappers.FusionCLIPWrapper") as MockClip, \
             patch("core.wrappers.FusionVAEWrapper") as MockVae:
            MockModel.return_value = MagicMock()
            MockClip.return_value = MagicMock()
            MockVae.return_value = MagicMock()
            node = CheckpointLoaderSimple()
            result = node.load_checkpoint("test-model")
            assert result is not None


class TestFusionModelLoaderNode:
    def test_input_types(self):
        from nodes.loaders import FusionModelLoaderNode
        inputs = FusionModelLoaderNode.INPUT_TYPES()
        assert "required" in inputs

    def test_load_pipeline(self):
        from nodes.loaders import FusionModelLoaderNode
        with patch("core.wrappers._resolve_model_path", return_value="/tmp/m"), \
             patch("core.wrappers._infer_model_type", return_value="video"), \
             patch("core.wrappers._map_clip_type_to_model_name", return_value="test"), \
             patch("core.wrappers._map_vae_name_to_model_name", return_value="test"), \
             patch("core.wrappers.FusionModelWrapper") as MockModel, \
             patch("core.wrappers.FusionCLIPWrapper") as MockClip, \
             patch("core.wrappers.FusionVAEWrapper") as MockVae:
            MockModel.return_value = MagicMock()
            MockClip.return_value = MagicMock()
            MockVae.return_value = MagicMock()
            node = FusionModelLoaderNode()
            result = node.load_pipeline("test-model", "sequential", "fp8_e4m3")
            assert result is not None
