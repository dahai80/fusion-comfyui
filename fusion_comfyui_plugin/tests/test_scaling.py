# fusion_comfyui_plugin/tests/test_scaling.py
import numpy as np


def _img(batch=2, c=3, h=64, w=64):
    rng = np.random.default_rng(42)
    return rng.random((batch, c, h, w), dtype=np.float32)


class TestLanczos:
    def test_shape_4d(self):
        from nodes._scaling import lanczos
        out = lanczos(_img(), 128, 96)
        assert out.shape == (2, 3, 96, 128)
        assert out.dtype == np.float32

    def test_range_preserved(self):
        from nodes._scaling import lanczos
        src = np.zeros((1, 3, 32, 32), dtype=np.float32)
        out = lanczos(src, 64, 64)
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestBislerp:
    def test_shape_4d(self):
        from nodes._scaling import bislerp
        out = bislerp(_img(), 128, 96)
        assert out.shape == (2, 3, 96, 128)
        assert out.dtype == np.float32

    def test_identity_upscale(self):
        from nodes._scaling import bislerp
        src = _img(1, 3, 16, 16)
        out = bislerp(src, 16, 16)
        assert np.allclose(out, src, atol=1e-4)


class TestCommonUpscale:
    def test_disabled_crop_grows(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 32, 32), 64, 64, "bilinear", "disabled")
        assert out.shape == (1, 3, 64, 64)

    def test_center_crop_aspect(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 64, 32), 32, 32, "bilinear", "center")
        assert out.shape == (1, 3, 32, 32)

    def test_5d_video(self):
        from nodes._scaling import common_upscale
        src = _img(1, 4, 32, 32)[:, :, None, :, :]  # B,C,T,H,W = (1,4,1,32,32)
        src = np.ascontiguousarray(src)
        out = common_upscale(src, 64, 64, "bilinear", "disabled")
        assert out.shape == (1, 4, 1, 64, 64)

    def test_method_lanczos(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 32, 32), 64, 64, "lanczos", "disabled")
        assert out.shape == (1, 3, 64, 64)

    def test_method_bislerp(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 32, 32), 64, 64, "bislerp", "disabled")
        assert out.shape == (1, 3, 64, 64)

    def test_nearest_exact(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 32, 32), 64, 64, "nearest-exact", "disabled")
        assert out.shape == (1, 3, 64, 64)
