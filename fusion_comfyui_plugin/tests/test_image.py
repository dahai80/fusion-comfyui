import numpy as np
import os
import tempfile
from unittest.mock import patch
from PIL import Image


class TestLoadImage:
    def test_input_types(self):
        from nodes.image import LoadImage
        with patch("folder_paths.get_input_directory", return_value=tempfile.mkdtemp()):
            inputs = LoadImage.INPUT_TYPES()
            assert "required" in inputs

    def test_load_image(self):
        from nodes.image import LoadImage
        tmpdir = tempfile.mkdtemp()
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img_path = os.path.join(tmpdir, "test.png")
        img.save(img_path)
        with patch("folder_paths.get_annotated_filepath", return_value=img_path):
            node = LoadImage()
            result = node.load_image("test.png")
            assert result is not None
            image_t, mask_t = result
            assert isinstance(image_t, np.ndarray)
            assert image_t.shape == (1, 64, 64, 3)
            assert image_t.dtype == np.float32
            assert isinstance(mask_t, np.ndarray)
            assert mask_t.ndim == 3


class TestSaveImage:
    def test_input_types(self):
        from nodes.image import SaveImage
        inputs = SaveImage.INPUT_TYPES()
        assert "required" in inputs

    def test_save_images(self):
        from nodes.image import SaveImage
        images = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
        tmpdir = tempfile.mkdtemp()
        with patch("folder_paths.get_output_directory", return_value=tmpdir), \
             patch("folder_paths.get_save_image_path", return_value=(tmpdir, "test", 1, "", "ComfyUI")), \
             patch("comfy.cli_args") as mock_args:
            mock_args.disable_metadata = False
            node = SaveImage()
            result = node.save_images(images, filename_prefix="test")
            assert result is not None


class TestPreviewImage:
    def test_input_types(self):
        from nodes.image import PreviewImage
        inputs = PreviewImage.INPUT_TYPES()
        assert "required" in inputs
