import numpy as np

from fusion_comfyui_plugin.nodes.sr import super_resolve_frames


def test_sr_mode_0_passthrough():
    imgs = np.random.rand(3, 16, 20, 3).astype(np.float32)
    out = super_resolve_frames(imgs, mode="0", scale=4)
    assert out.shape == (3, 16, 20, 3)
    assert np.array_equal(out, imgs)


def test_sr_mock_resize_4x_dims():
    imgs = np.random.rand(2, 16, 20, 3).astype(np.float32)
    out = super_resolve_frames(imgs, mode="mock", scale=4)
    assert out.shape == (2, 64, 80, 3), out.shape
    assert out.dtype == np.float32


def test_sr_mock_resize_2x_dims():
    imgs = np.zeros((1, 32, 40, 3), dtype=np.float32)
    out = super_resolve_frames(imgs, mode="mock", scale=2)
    assert out.shape == (1, 64, 80, 3), out.shape


def test_sr_mock_preserves_range():
    imgs = np.zeros((1, 12, 12, 3), dtype=np.float32)
    out = super_resolve_frames(imgs, mode="mock", scale=4)
    assert np.isfinite(out).all()
