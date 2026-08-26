import numpy as np
import tempfile
from unittest.mock import MagicMock, patch


class TestSaveWEBM:
    def test_input_types(self):
        from nodes.video_io import SaveWEBM
        inputs = SaveWEBM.INPUT_TYPES()
        assert "required" in inputs

    def test_save_webm(self):
        from nodes.video_io import SaveWEBM
        frames = np.random.randint(0, 255, (4, 512, 512, 3), dtype=np.uint8)
        tmpdir = tempfile.mkdtemp()
        with patch("folder_paths.get_output_directory", return_value=tmpdir), \
             patch("folder_paths.get_save_image_path", return_value=(tmpdir, "test", 1, "", "ComfyUI")), \
             patch("av.open") as mock_av:
            mock_container = MagicMock()
            mock_av.return_value = mock_container
            mock_stream = MagicMock()
            mock_container.add_stream.return_value = mock_stream
            mock_container.__enter__ = MagicMock(return_value=mock_container)
            mock_container.__exit__ = MagicMock(return_value=False)
            node = SaveWEBM()
            result = node.save_video(frames, filename_prefix="test")
            assert result is not None


class TestSaveAnimatedWEBP:
    def test_input_types(self):
        from nodes.video_io import SaveAnimatedWEBP
        inputs = SaveAnimatedWEBP.INPUT_TYPES()
        assert "required" in inputs


class TestFusionSaveVideoNode:
    def test_input_types(self):
        from nodes.video_io import FusionSaveVideoNode
        inputs = FusionSaveVideoNode.INPUT_TYPES()
        assert "required" in inputs

    def test_write_frames(self):
        from nodes.video_io import FusionSaveVideoNode
        frames = np.random.randint(0, 255, (4, 512, 512, 3), dtype=np.uint8)
        tmpdir = tempfile.mkdtemp()
        with patch("folder_paths.get_output_directory", return_value=tmpdir), \
             patch("folder_paths.get_save_image_path", return_value=(tmpdir, "test", 1, "", "ComfyUI")), \
             patch("fusion_comfyui.core.bridge.to_numpy", return_value=frames), \
             patch("av.open") as mock_av:
            mock_container = MagicMock()
            mock_av.return_value = mock_container
            mock_stream = MagicMock()
            mock_container.add_stream.return_value = mock_stream
            mock_container.__enter__ = MagicMock(return_value=mock_container)
            mock_container.__exit__ = MagicMock(return_value=False)
            node = FusionSaveVideoNode()
            result = node.save_video(frames, filename_prefix="test")
            assert result is not None


class TestFusionVideoConcatNode:
    def test_input_types(self):
        from nodes.video_io import FusionVideoConcatNode
        inputs = FusionVideoConcatNode.INPUT_TYPES()
        assert "required" in inputs

    def test_concat(self):
        from nodes.video_io import FusionVideoConcatNode
        v1 = np.random.rand(4, 512, 512, 3).astype(np.float32)
        v2 = np.random.rand(4, 512, 512, 3).astype(np.float32)
        node = FusionVideoConcatNode()
        result = node.concat(v1, v2)
        assert result is not None

    def test_concat_5d(self):
        from nodes.video_io import FusionVideoConcatNode
        v1 = np.random.rand(1, 4, 512, 512, 3).astype(np.float32)
        v2 = np.random.rand(1, 4, 512, 512, 3).astype(np.float32)
        node = FusionVideoConcatNode()
        result = node.concat(v1, v2)
        assert result is not None

    def test_concat_shape_mismatch_resizes(self):
        from nodes.video_io import FusionVideoConcatNode
        v1 = np.random.rand(4, 512, 512, 3).astype(np.float32)
        v2 = np.random.rand(4, 256, 256, 3).astype(np.float32)
        node = FusionVideoConcatNode()
        result = node.concat(v1, v2)
        assert result is not None
