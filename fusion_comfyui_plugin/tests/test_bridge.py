import numpy as np
from unittest.mock import MagicMock, patch
import sys


def _get_mx_array_cls():
    return sys.modules["mlx.core"].array


class TestToMlxArray:
    def test_numpy_passthrough(self):
        import fusion_comfyui.core.bridge as bridge
        MxArray = _get_mx_array_cls()
        arr = np.zeros((2, 3), dtype=np.float32)
        result = bridge.to_mlx_array(arr)
        assert isinstance(result, MxArray)

    def test_mx_array_passthrough(self):
        import fusion_comfyui.core.bridge as bridge
        MxArray = _get_mx_array_cls()
        mx_arr = MxArray()
        result = bridge.to_mlx_array(mx_arr)
        assert result is mx_arr

    def test_array_interface(self):
        import fusion_comfyui.core.bridge as bridge
        MxArray = _get_mx_array_cls()
        arr = np.zeros((2, 3), dtype=np.float32)
        class FakeTensor:
            __array_interface__ = arr.__array_interface__
        result = bridge.to_mlx_array(FakeTensor())
        assert isinstance(result, MxArray)

    def test_numpy_method(self):
        import fusion_comfyui.core.bridge as bridge
        MxArray = _get_mx_array_cls()
        fake = MagicMock()
        fake.numpy = MagicMock(return_value=np.zeros((2, 3)))
        del fake.__array_interface__
        result = bridge.to_mlx_array(fake)
        assert isinstance(result, MxArray)

    def test_cpu_numpy_method(self):
        import fusion_comfyui.core.bridge as bridge
        MxArray = _get_mx_array_cls()
        fake = MagicMock(spec=[])
        fake.cpu = MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.zeros((2, 3)))))
        fake.numpy = MagicMock(return_value=np.zeros((2, 3)))
        result = bridge.to_mlx_array(fake)
        assert isinstance(result, MxArray)

    def test_fallback_to_np_array(self):
        import fusion_comfyui.core.bridge as bridge
        MxArray = _get_mx_array_cls()
        result = bridge.to_mlx_array([1, 2, 3])
        assert isinstance(result, MxArray)

    def test_non_contiguous(self):
        import fusion_comfyui.core.bridge as bridge
        MxArray = _get_mx_array_cls()
        arr = np.zeros((3, 4), dtype=np.float32)
        non_contig = arr[:, ::2]
        assert not non_contig.flags["C_CONTIGUOUS"]
        result = bridge.to_mlx_array(non_contig)
        assert isinstance(result, MxArray)


class TestToNumpy:
    def test_numpy_passthrough(self):
        import fusion_comfyui.core.bridge as bridge
        arr = np.zeros((2, 3), dtype=np.float32)
        result = bridge.to_numpy(arr)
        assert result is arr

    def test_mx_array(self):
        import fusion_comfyui.core.bridge as bridge
        with patch("numpy.array", side_effect=lambda *a, **kw: np.asarray(*a)):
            MxArray = _get_mx_array_cls()
            mx_arr = MxArray(np.zeros((2, 3), dtype=np.float32))
            result = bridge.to_numpy(mx_arr)
            assert isinstance(result, np.ndarray)

    def test_cpu_numpy(self):
        import fusion_comfyui.core.bridge as bridge
        expected = np.zeros((2, 3))
        class FakeCpuTensor:
            def cpu(self):
                class Inner:
                    def numpy(self):
                        return expected
                return Inner()
            def numpy(self):
                return expected
        fake = FakeCpuTensor()
        result = bridge.to_numpy(fake)
        np.testing.assert_array_equal(result, expected)

    def test_fallback_asarray(self):
        import fusion_comfyui.core.bridge as bridge
        result = bridge.to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)


class TestToImageArray:
    def test_float32_passthrough(self):
        import fusion_comfyui.core.bridge as bridge
        arr = np.zeros((1, 512, 512, 3), dtype=np.float32)
        result = bridge.to_image_array(arr)
        assert result.shape == (1, 512, 512, 3)

    def test_5d_reshape(self):
        import fusion_comfyui.core.bridge as bridge
        arr = np.random.rand(1, 4, 3, 512, 512).astype(np.float32)
        with patch.object(bridge, "to_numpy", return_value=arr):
            result = bridge.to_image_array(MagicMock())
            assert result.ndim == 4

    def test_chw_to_hwc(self):
        import fusion_comfyui.core.bridge as bridge
        arr = np.random.rand(1, 3, 512, 512).astype(np.float32)
        with patch.object(bridge, "to_numpy", return_value=arr):
            result = bridge.to_image_array(MagicMock())
            assert result.shape[-1] == 3

    def test_3d_hwc_expand(self):
        import fusion_comfyui.core.bridge as bridge
        arr = np.random.rand(3, 512, 512).astype(np.float32)
        with patch.object(bridge, "to_numpy", return_value=arr):
            result = bridge.to_image_array(MagicMock())
            assert result.ndim == 4

    def test_uint8_normalize(self):
        import fusion_comfyui.core.bridge as bridge
        arr = np.full((1, 64, 64, 3), 200, dtype=np.float32)
        with patch.object(bridge, "to_numpy", return_value=arr):
            result = bridge.to_image_array(MagicMock())
            assert result.max() <= 1.0

    def test_non_float32_cast(self):
        import fusion_comfyui.core.bridge as bridge
        arr = np.zeros((1, 64, 64, 3), dtype=np.uint8)
        with patch.object(bridge, "to_numpy", return_value=arr):
            result = bridge.to_image_array(MagicMock())
            assert result.dtype == np.float32


def test_to_image_tensor_returns_numpy():
    from fusion_comfyui.core.bridge import to_image_tensor
    src = np.random.default_rng(1).random((2, 8, 8, 3)).astype(np.float32)
    out = to_image_tensor(src)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (2, 8, 8, 3)


def test_to_mask_numpy():
    from fusion_comfyui.core.bridge import to_mask_numpy
    src = np.random.default_rng(2).random((4, 8, 8)).astype(np.float32)
    out = to_mask_numpy(src)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.ndim == 3


def test_to_image_numpy_alias():
    from fusion_comfyui.core.bridge import to_image_numpy, to_image_tensor
    assert to_image_numpy is to_image_tensor


def test_to_mask_numpy_squeezes_singleton_channel():
    from fusion_comfyui.core.bridge import to_mask_numpy
    src = np.random.default_rng(4).random((2, 8, 8, 1)).astype(np.float32)
    out = to_mask_numpy(src)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (2, 8, 8), "4D [B,H,W,1] must squeeze to 3D [B,H,W], got {}".format(out.shape)


def test_to_mask_numpy_expands_2d():
    from fusion_comfyui.core.bridge import to_mask_numpy
    src = np.random.default_rng(5).random((8, 8)).astype(np.float32)
    out = to_mask_numpy(src)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (1, 8, 8), "2D [H,W] must expand to 3D [1,H,W], got {}".format(out.shape)


def test_to_mask_numpy_does_not_squeeze_image_channels():
    from fusion_comfyui.core.bridge import to_mask_numpy
    src = np.random.default_rng(6).random((2, 8, 8, 3)).astype(np.float32)
    out = to_mask_numpy(src)
    assert out.shape == (2, 8, 8, 3), "4D [B,H,W,3] must NOT be squeezed, got {}".format(out.shape)
