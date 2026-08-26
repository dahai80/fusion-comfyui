import asyncio

import mlx.core as mx
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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


def _staged_mock_model(model_type="video"):
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    mock = MagicMock(spec=FusionModelWrapper)
    mock.model_type = model_type
    mock.model_name = f"staged-{model_type}"
    engine = MagicMock()
    engine.ensure_started = AsyncMock()
    engine._run_staged_pipeline = AsyncMock(
        return_value=mx.array(np.random.rand(4, 512, 768, 3).astype(np.float32))
    )
    mock.get_engine = MagicMock(return_value=engine)
    return mock


def _make_mock_model_monolith(model_type="image"):
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    mock = MagicMock(spec=FusionModelWrapper)
    mock.model_type = model_type
    mock.model_name = f"mono-{model_type}"
    engine = MagicMock()
    engine.ensure_started = AsyncMock()
    mock.get_engine = MagicMock(return_value=engine)
    return mock


class TestGenerateStaged:
    def test_video_staged_calls_pipeline(self):
        from nodes.samplers import _generate_staged
        model = _staged_mock_model("video")
        positive = {"prompt": "cat playing"}
        negative = {"prompt": "blurry"}
        latent = {
            "samples": np.zeros((1, 16, 5, 32, 32), dtype=np.float32),
            "num_frames": 41, "width": 768, "height": 512,
        }
        result = asyncio.run(
            _generate_staged(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41)
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 3 and result.shape[-1] == 3
        engine = model.get_engine.return_value
        engine._run_staged_pipeline.assert_awaited_once()
        call = engine._run_staged_pipeline.await_args
        assert call.kwargs["prompt"] == "cat playing"
        assert call.kwargs["neg_prompt"] == "blurry"
        assert call.kwargs["num_frames"] == 41

    def test_image_staged_calls_pipeline(self):
        from nodes.samplers import _generate_staged
        model = _staged_mock_model("image")
        model.get_engine.return_value._run_staged_pipeline = AsyncMock(
            return_value=mx.array(np.random.rand(1, 3, 512, 512).astype(np.float32))
        )
        positive = {"prompt": "a cat"}
        negative = {"prompt": "bad"}
        latent = {
            "samples": np.zeros((1, 4, 64, 64), dtype=np.float32),
            "width": 512, "height": 512,
        }
        result = asyncio.run(
            _generate_staged(model, positive, negative, latent, 20, 6.0, 42, 512, 512, 1)
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (512, 512, 3)

    def test_staged_negative_none(self):
        from nodes.samplers import _generate_staged
        model = _staged_mock_model("video")
        positive = {"prompt": "cat"}
        negative = None
        latent = {
            "samples": np.zeros((1, 16, 5, 32, 32), dtype=np.float32),
            "num_frames": 41, "width": 768, "height": 512,
        }
        asyncio.run(
            _generate_staged(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41)
        )
        call = model.get_engine.return_value._run_staged_pipeline.await_args
        assert call.kwargs["neg_prompt"] == ""


class TestSampleDispatch:
    def test_t2v_routes_to_staged(self):
        from nodes.samplers import KSampler
        model = _staged_mock_model("video")
        positive = {"prompt": "cat"}
        negative = {"prompt": "bad"}
        latent = {
            "samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32),
            "num_frames": 41, "width": 768, "height": 512,
        }
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        model.get_engine.return_value._run_staged_pipeline.assert_awaited_once()
        assert "_decoded_frames_key" in result[0]

    def test_i2v_routes_to_monolith(self):
        from nodes.samplers import KSampler
        model = _make_mock_model_monolith("video")
        positive = {"prompt": "cat"}
        negative = {"prompt": "bad"}
        latent = {
            "samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32),
            "_i2v_image_path": "/tmp/x.png",
        }
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((4, 512, 768, 3), dtype=np.float32)) as ra:
            node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert ra.called
        assert ra.call_args[0][0].__qualname__ == "_generate_monolithic"

    def test_cascade_routes_to_monolith(self):
        from nodes.samplers import KSampler
        model = _make_mock_model_monolith("image")
        prior = np.zeros((64, 64, 3), dtype=np.float32)
        positive = {"prompt": "p", "stable_cascade_prior": prior}
        negative = {"prompt": "n"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=prior) as ra:
            node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert ra.called
        assert ra.call_args[0][0].__qualname__ == "_generate_monolithic"
