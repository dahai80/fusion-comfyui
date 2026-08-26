# fusion_comfyui_plugin/tests/test_image_transform.py
import numpy as np


def _img(batch=1, h=32, w=32, c=3):
    rng = np.random.default_rng(7)
    return rng.random((batch, h, w, c), dtype=np.float32)


class TestImageScale:
    def test_upscale(self):
        from nodes.image_transform import ImageScale
        out = ImageScale().upscale(_img(1, 32, 32), "bilinear", 64, 64, "disabled")
        assert out[0].shape == (1, 64, 64, 3)
        assert out[0].dtype == np.float32

    def test_zero_dims_passthrough(self):
        from nodes.image_transform import ImageScale
        src = _img(1, 32, 32)
        out = ImageScale().upscale(src, "bilinear", 0, 0, "disabled")
        assert out[0].shape == src.shape


class TestImageScaleBy:
    def test_scale_by(self):
        from nodes.image_transform import ImageScaleBy
        out = ImageScaleBy().upscale(_img(1, 20, 20), "bilinear", 2.0)
        assert out[0].shape == (1, 40, 40, 3)


class TestImageBatch:
    def test_same_channels(self):
        from nodes.image_transform import ImageBatch
        a = _img(2, 16, 16, 3)
        b = _img(3, 16, 16, 3)
        out = ImageBatch().batch(a, b)
        assert out[0].shape == (5, 16, 16, 3)

    def test_channel_pad(self):
        from nodes.image_transform import ImageBatch
        a = _img(1, 16, 16, 3)
        b = _img(1, 16, 16, 4)
        out = ImageBatch().batch(a, b)
        assert out[0].shape == (2, 16, 16, 4)
        # padded alpha channel (row 0, was 3-channel -> 4-channel) must be 1.0,
        # not 0.0 — distinguishes the correct pad value from a wrong one.
        assert np.all(out[0][0, :, :, 3] == 1.0), "padded alpha should be 1.0, got {}".format(out[0][0, :, :, 3])


class TestEmptyImage:
    def test_generate(self):
        from nodes.image_transform import EmptyImage
        out = EmptyImage().generate(64, 64, batch_size=2, color=0xFF0000)
        assert out[0].shape == (2, 64, 64, 3)
        assert out[0].dtype == np.float32
        assert np.allclose(out[0][0, 0, 0], [1.0, 0.0, 0.0])


class TestImagePadForOutpaint:
    def test_expand(self):
        from nodes.image_transform import ImagePadForOutpaint
        src = _img(1, 16, 16, 3)
        img, mask = ImagePadForOutpaint().expand_image(src, 4, 4, 4, 4, 0)
        assert img.shape == (1, 24, 24, 3)
        assert mask.shape == (1, 24, 24)


class TestLoadImageMask:
    def test_input_types(self):
        from nodes.image_transform import LoadImageMask
        with __import__("unittest.mock").mock.patch(
            "folder_paths.get_input_directory", return_value="/tmp"
        ):
            inputs = LoadImageMask.INPUT_TYPES()
            assert "channel" in inputs["required"]
