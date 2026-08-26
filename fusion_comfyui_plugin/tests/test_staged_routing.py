import numpy as np
import pytest
from unittest.mock import MagicMock


def _model(model_type):
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    m = MagicMock(spec=FusionModelWrapper)
    m.model_type = model_type
    return m


def _latent(**extra):
    base = {"samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32)}
    base.update(extra)
    return base


@pytest.mark.parametrize("model_type,latent_extra,denoise,positive_extra,expected", [
    ("video", {}, 1.0, {}, True),                                   # T2V pure text -> staged
    ("video", {"_i2v_image_path": "/tmp/x.png"}, 1.0, {}, False),   # I2V -> monolith
    ("video", {"_vace_control_video": "/tmp/v.mp4"}, 1.0, {}, False),  # VACE ctrl -> monolith
    ("video", {"_vace_control_mask": "/tmp/m.png"}, 1.0, {}, False),   # VACE mask -> monolith
    ("video", {"_vace_reference_images": "/tmp/r.png"}, 1.0, {}, False),  # VACE ref -> monolith
    ("image", {"_image_init_path": "/tmp/i.png"}, 0.5, {}, False),  # img2img (denoise<1) -> monolith
    ("image", {"_image_init_path": "/tmp/i.png"}, 1.0, {}, True),   # txt2img w/ stale init key, denoise=1 -> staged
    ("image", {}, 1.0, {}, True),                                   # txt2img FLUX.2 -> staged
    ("image", {}, 1.0, {"stable_cascade_prior": np.zeros((64, 64, 3))}, False),  # cascade stage_b -> pass-through
])
def test_should_use_staged_matrix(model_type, latent_extra, denoise, positive_extra, expected):
    from nodes.samplers import _should_use_staged
    model = _model(model_type)
    positive = {"prompt": "p", **positive_extra}
    negative = {"prompt": "n"}
    latent = _latent(**latent_extra)
    assert _should_use_staged(model, positive, negative, latent, denoise) is expected


def test_should_use_staged_negative_none():
    from nodes.samplers import _should_use_staged
    model = _model("video")
    assert _should_use_staged(model, {"prompt": "p"}, None, _latent(), 1.0) is True


def test_should_use_staged_unknown_type():
    from nodes.samplers import _should_use_staged
    model = _model("audio")  # neither video nor image
    assert _should_use_staged(model, {"prompt": "p"}, None, _latent(), 1.0) is False


def test_staged_pixels_video_mx_array():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    # VAE decode returns float [0,1]; THWC
    pixels = mx.array(np.random.rand(4, 512, 768, 3).astype(np.float32))
    out = _staged_pixels_to_numpy(pixels, "video")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (4, 512, 768, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_staged_pixels_video_ndim5_squeezes():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    pixels = mx.array(np.random.rand(1, 4, 512, 768, 3).astype(np.float32))
    out = _staged_pixels_to_numpy(pixels, "video")
    assert out.shape == (4, 512, 768, 3)


def test_staged_pixels_image_nchw_to_hwc():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    # Image decode returns [batch,c,h,w]
    pixels = mx.array(np.random.rand(1, 3, 512, 512).astype(np.float32))
    out = _staged_pixels_to_numpy(pixels, "image")
    assert out.shape == (512, 512, 3)
    assert out.dtype == np.float32


def test_staged_pixels_image_4ch_slices_to_3():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    pixels = mx.array(np.random.rand(1, 4, 64, 64).astype(np.float32))
    out = _staged_pixels_to_numpy(pixels, "image")
    assert out.shape == (64, 64, 3)


def test_staged_pixels_clamps_out_of_range():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    pixels = mx.array(np.full((2, 8, 8, 3), 2.0, dtype=np.float32))
    out = _staged_pixels_to_numpy(pixels, "video")
    assert out.max() <= 1.0


def test_staged_pixels_uint8_fallback_divides():
    from nodes.samplers import _staged_pixels_to_numpy
    # Defensive: if decode ever returns uint8 numpy, divide like monolith
    pixels = np.full((2, 8, 8, 3), 255, dtype=np.uint8)
    out = _staged_pixels_to_numpy(pixels, "video")
    assert out.dtype == np.float32
    assert out.max() <= 1.0
