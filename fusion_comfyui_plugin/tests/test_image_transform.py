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


class TestPainterNode:
    def test_input_types(self):
        from nodes.image_transform import PainterNode
        inputs = PainterNode.INPUT_TYPES()
        assert "mask" in inputs["required"]
        assert "width" in inputs["required"]
        assert "height" in inputs["required"]
        assert "bg_color" in inputs["required"]
        assert "image" in inputs.get("optional", {})
        assert PainterNode.RETURN_TYPES == ("IMAGE", "MASK")

    def test_blank_canvas_color(self):
        from nodes.image_transform import PainterNode
        img, mask = PainterNode().paint("", 64, 48, "#ff0000")
        assert img.shape == (1, 48, 64, 3)
        assert img.dtype == np.float32
        assert mask.shape == (1, 48, 64)
        assert mask.dtype == np.float32
        assert np.allclose(img[0, 0, 0], [1.0, 0.0, 0.0]), "bg_color #ff0000 -> red canvas"
        assert np.all(mask == 0.0), "no mask file -> all-zero mask"

    def test_blank_canvas_uses_optional_image_dims(self):
        from nodes.image_transform import PainterNode
        src = _img(1, 20, 30, 3)
        img, mask = PainterNode().paint("", 64, 64, "#000000", image=src)
        assert img.shape == (1, 20, 30, 3), "dims come from optional image, not width/height"

    def test_composite_alpha_and_mask(self, tmp_path):
        from PIL import Image
        from nodes.image_transform import PainterNode
        painter_img = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
        mask_path = str(tmp_path / "paint.png")
        painter_img.save(mask_path)
        with __import__("unittest.mock").mock.patch(
            "folder_paths.get_annotated_filepath", return_value=mask_path
        ):
            img, mask = PainterNode().paint("paint.png", 8, 8, "#000000")
        assert img.shape == (1, 8, 8, 3)
        assert mask.shape == (1, 8, 8)
        assert np.allclose(img[0, 0, 0], [10 / 255.0, 20 / 255.0, 30 / 255.0]), "fully opaque paint overwrites bg"
        assert np.all(mask == 1.0), "opaque alpha channel -> all-one mask"

    def test_no_torch_import(self):
        import sys
        before = set(sys.modules)
        from nodes.image_transform import PainterNode  # noqa: F401
        leaked = [m for m in ("torch",) if m in sys.modules and m not in before]
        assert not leaked, "PainterNode import leaked torch: {}".format(leaked)
