import numpy as np
import os
from unittest.mock import MagicMock

import mlx.core as mx


class TestEmptyLatentImage:
    def test_input_types(self):
        from nodes.latent import EmptyLatentImage
        inputs = EmptyLatentImage.INPUT_TYPES()
        assert "required" in inputs

    def test_generate(self):
        from nodes.latent import EmptyLatentImage
        node = EmptyLatentImage()
        result = node.generate(width=512, height=512, batch_size=1)
        assert result is not None
        assert "samples" in result[0]
        assert "width" in result[0]
        assert "height" in result[0]

    def test_generate_custom_size(self):
        from nodes.latent import EmptyLatentImage
        node = EmptyLatentImage()
        result = node.generate(width=1024, height=768, batch_size=2)
        samples = result[0]["samples"]
        assert samples.shape[0] == 2


class TestEmptyHunyuanLatentVideo:
    def test_input_types(self):
        from nodes.latent import EmptyHunyuanLatentVideo
        inputs = EmptyHunyuanLatentVideo.INPUT_TYPES()
        assert "required" in inputs

    def test_generate(self):
        from nodes.latent import EmptyHunyuanLatentVideo
        node = EmptyHunyuanLatentVideo()
        result = node.generate(width=848, height=480, length=33, batch_size=1)
        assert result is not None
        assert "samples" in result[0]
        assert "num_frames" in result[0]
        assert result[0]["num_frames"] == 33
        assert result[0]["downscale_ratio_spacial"] == 8

    def test_generate_custom_length(self):
        from nodes.latent import EmptyHunyuanLatentVideo
        node = EmptyHunyuanLatentVideo()
        result = node.generate(width=848, height=480, length=65, batch_size=1)
        samples = result[0]["samples"]
        assert samples.shape[2] == (65 - 1) // 4 + 1


class TestEmptyCosmosLatentVideo:
    def test_input_types(self):
        from nodes.latent import EmptyCosmosLatentVideo
        inputs = EmptyCosmosLatentVideo.INPUT_TYPES()
        assert "required" in inputs

    def test_generate(self):
        from nodes.latent import EmptyCosmosLatentVideo
        node = EmptyCosmosLatentVideo()
        result = node.generate(width=1280, height=704, length=121, batch_size=1)
        assert result is not None
        assert "samples" in result[0]
        assert "num_frames" in result[0]
        assert result[0]["num_frames"] == 121

    def test_generate_custom_length(self):
        from nodes.latent import EmptyCosmosLatentVideo
        node = EmptyCosmosLatentVideo()
        result = node.generate(width=1280, height=704, length=57, batch_size=1)
        samples = result[0]["samples"]
        assert samples.shape[2] == max(2, (57 // 4 // 2) * 2)


class TestWan22ImageToVideoLatent:
    def test_input_types(self):
        from nodes.latent import Wan22ImageToVideoLatent
        inputs = Wan22ImageToVideoLatent.INPUT_TYPES()
        assert "required" in inputs
        assert "optional" in inputs

    def test_generate_t2v(self):
        from nodes.latent import Wan22ImageToVideoLatent
        mock_vae = MagicMock()
        node = Wan22ImageToVideoLatent()
        result = node.generate(mock_vae, width=1280, height=704, length=49, batch_size=1)
        assert len(result) == 1
        assert "samples" in result[0]
        assert "num_frames" in result[0]
        assert "_i2v_image_path" not in result[0]

    def test_generate_i2v_numpy(self):
        from nodes.latent import Wan22ImageToVideoLatent
        mock_vae = MagicMock()
        start_image = np.zeros((1, 704, 1280, 3), dtype=np.float32)
        node = Wan22ImageToVideoLatent()
        result = node.generate(mock_vae, width=1280, height=704, length=49, batch_size=1, start_image=start_image)
        assert "_i2v_image_path" in result[0]
        tmp_path = result[0]["_i2v_image_path"]
        assert os.path.exists(tmp_path)
        os.unlink(tmp_path)

    def test_generate_i2v_mx_array(self):
        from nodes.latent import Wan22ImageToVideoLatent
        mock_vae = MagicMock()
        start_image = mx.array(np.zeros((1, 704, 1280, 3), dtype=np.float32))
        node = Wan22ImageToVideoLatent()
        result = node.generate(mock_vae, width=1280, height=704, length=49, batch_size=1, start_image=start_image)
        assert "_i2v_image_path" in result[0]
        os.unlink(result[0]["_i2v_image_path"])

    def test_generate_i2v_3d_image(self):
        from nodes.latent import Wan22ImageToVideoLatent
        mock_vae = MagicMock()
        start_image = np.zeros((704, 1280, 3), dtype=np.float32)
        node = Wan22ImageToVideoLatent()
        result = node.generate(mock_vae, width=1280, height=704, length=49, batch_size=1, start_image=start_image)
        assert "_i2v_image_path" in result[0]
        os.unlink(result[0]["_i2v_image_path"])


class TestWanImageToVideo:
    def test_input_types(self):
        from nodes.latent import WanImageToVideo
        inputs = WanImageToVideo.INPUT_TYPES()
        assert "required" in inputs
        assert "optional" in inputs

    def test_generate_t2v(self):
        from nodes.latent import WanImageToVideo
        positive = MagicMock()
        negative = MagicMock()
        mock_vae = MagicMock()
        node = WanImageToVideo()
        result = node.generate(positive, negative, mock_vae, width=832, height=480, length=81, batch_size=1)
        assert len(result) == 3
        assert "samples" in result[2]
        assert "_i2v_image_path" not in result[2]

    def test_generate_i2v(self):
        from nodes.latent import WanImageToVideo
        positive = MagicMock()
        negative = MagicMock()
        mock_vae = MagicMock()
        start_image = np.zeros((1, 480, 832, 3), dtype=np.float32)
        node = WanImageToVideo()
        result = node.generate(positive, negative, mock_vae, width=832, height=480, length=81, batch_size=1, start_image=start_image)
        assert len(result) == 3
        assert "_i2v_image_path" in result[2]
        os.unlink(result[2]["_i2v_image_path"])

    def test_generate_i2v_mx_array(self):
        from nodes.latent import WanImageToVideo
        positive = MagicMock()
        negative = MagicMock()
        mock_vae = MagicMock()
        start_image = mx.array(np.zeros((1, 480, 832, 3), dtype=np.float32))
        node = WanImageToVideo()
        result = node.generate(positive, negative, mock_vae, width=832, height=480, length=81, batch_size=1, start_image=start_image)
        assert len(result) == 3
        assert "_i2v_image_path" in result[2]
        os.unlink(result[2]["_i2v_image_path"])

    def test_generate_i2v_3d_image(self):
        from nodes.latent import WanImageToVideo
        positive = MagicMock()
        negative = MagicMock()
        mock_vae = MagicMock()
        start_image = np.zeros((480, 832, 3), dtype=np.float32)
        node = WanImageToVideo()
        result = node.generate(positive, negative, mock_vae, width=832, height=480, length=81, batch_size=1, start_image=start_image)
        assert len(result) == 3
        assert "_i2v_image_path" in result[2]
        os.unlink(result[2]["_i2v_image_path"])


class TestLTXVImgToVideo:
    def test_input_types(self):
        from nodes.latent import LTXVImgToVideo
        inputs = LTXVImgToVideo.INPUT_TYPES()
        assert "required" in inputs

    def test_generate_with_image(self):
        from nodes.latent import LTXVImgToVideo
        positive = MagicMock()
        negative = MagicMock()
        mock_vae = MagicMock()
        mock_image = np.zeros((1, 512, 768, 3), dtype=np.float32)
        node = LTXVImgToVideo()
        result = node.generate(positive, negative, mock_vae, mock_image, width=768, height=512, length=97, batch_size=1)
        assert len(result) == 3
        assert "_i2v_image_path" in result[2]
        assert "_i2v_image_strength" in result[2]
        assert result[2]["_i2v_image_strength"] == 1.0
        os.unlink(result[2]["_i2v_image_path"])

    def test_generate_with_custom_strength(self):
        from nodes.latent import LTXVImgToVideo
        positive = MagicMock()
        negative = MagicMock()
        mock_vae = MagicMock()
        mock_image = np.zeros((1, 512, 768, 3), dtype=np.float32)
        node = LTXVImgToVideo()
        result = node.generate(positive, negative, mock_vae, mock_image, width=768, height=512, length=97, batch_size=1, strength=0.7)
        assert result[2]["_i2v_image_strength"] == 0.7
        os.unlink(result[2]["_i2v_image_path"])

    def test_generate_no_image(self):
        from nodes.latent import LTXVImgToVideo
        positive = MagicMock()
        negative = MagicMock()
        mock_vae = MagicMock()
        node = LTXVImgToVideo()
        result = node.generate(positive, negative, mock_vae, None, width=768, height=512, length=97, batch_size=1)
        assert len(result) == 3
        assert "_i2v_image_path" not in result[2]

    def test_generate_mx_image(self):
        from nodes.latent import LTXVImgToVideo
        positive = MagicMock()
        negative = MagicMock()
        mock_vae = MagicMock()
        mock_image = mx.array(np.zeros((1, 512, 768, 3), dtype=np.float32))
        node = LTXVImgToVideo()
        result = node.generate(positive, negative, mock_vae, mock_image, width=768, height=512, length=97, batch_size=1)
        assert "_i2v_image_path" in result[2]
        os.unlink(result[2]["_i2v_image_path"])

    def test_generate_3d_image(self):
        from nodes.latent import LTXVImgToVideo
        positive = MagicMock()
        negative = MagicMock()
        mock_vae = MagicMock()
        mock_image = np.zeros((512, 768, 3), dtype=np.float32)
        node = LTXVImgToVideo()
        result = node.generate(positive, negative, mock_vae, mock_image, width=768, height=512, length=97, batch_size=1)
        assert "_i2v_image_path" in result[2]
        os.unlink(result[2]["_i2v_image_path"])


class TestFusionEmptyLatentNode:
    def test_input_types(self):
        from nodes.latent import FusionEmptyLatentNode
        inputs = FusionEmptyLatentNode.INPUT_TYPES()
        assert "required" in inputs

    def test_generate_image(self):
        from nodes.latent import FusionEmptyLatentNode
        node = FusionEmptyLatentNode()
        result = node.generate(width=1024, height=1024, batch_size=1, num_frames=1)
        assert result is not None
        samples = result[0]["samples"]
        assert samples.ndim == 4
        assert result[0]["num_frames"] == 1

    def test_generate_video(self):
        from nodes.latent import FusionEmptyLatentNode
        node = FusionEmptyLatentNode()
        result = node.generate(width=1024, height=1024, batch_size=1, num_frames=41)
        assert result is not None
        samples = result[0]["samples"]
        assert samples.ndim == 5
        assert result[0]["num_frames"] == 41
        latent_t = (41 - 1) // 4 + 1
        assert samples.shape[2] == latent_t
