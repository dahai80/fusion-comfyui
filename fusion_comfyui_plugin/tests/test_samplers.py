import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

import mlx.core as mx


def _make_mock_model(model_type="image"):
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    mock = MagicMock(spec=FusionModelWrapper)
    mock.model_type = model_type
    mock.model_name = f"test-{model_type}"
    engine_mock = MagicMock()
    engine_mock._engine.generate = AsyncMock(
        return_value=[b"\x89PNG" + b"\x00" * 100]
    )
    engine_mock.ensure_started = AsyncMock()
    mock.get_engine = MagicMock(return_value=engine_mock)
    return mock


def _sync_run(coro):
    coro.close()
    return np.zeros((1, 4, 64, 64), dtype=np.float32)


def _run_coro(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class TestRunAsync:
    def test_run_async_no_running_loop(self):
        from fusion_comfyui.core.async_utils import run_async

        async def coro():
            return 42

        result = run_async(coro())
        assert result == 42

    def test_run_async_with_running_loop(self):
        from fusion_comfyui.core.async_utils import run_async

        async def inner():
            async def coro():
                return 99
            return run_async(coro())

        result = asyncio.run(inner())
        assert result == 99

    def test_run_async_timeout_default(self):
        from fusion_comfyui.core.async_utils import run_async
        import inspect
        sig = inspect.signature(run_async)
        assert sig.parameters["timeout"].default == 600


class TestGenerateMonolithic:
    def _install_mock_av(self, container=None, import_error=False):
        import sys
        mock_av = MagicMock()
        if import_error:
            mock_av.open.side_effect = ImportError("no av")
        elif container:
            mock_av.open.return_value = container
        else:
            mock_av.open.side_effect = ImportError("no av")
        sys.modules["av"] = mock_av
        return mock_av

    def _remove_mock_av(self):
        import sys
        sys.modules.pop("av", None)

    def test_video_gen_no_i2v(self):
        from nodes.samplers import _generate_monolithic
        model = _make_mock_model("video")
        model.get_engine.return_value._engine.generate = AsyncMock(
            return_value=[b"\x00" * 100]
        )
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        self._install_mock_av(import_error=True)
        try:
            with patch("fusion_comfyui.core.bridge.to_mlx_array", return_value=mx.array(np.zeros((1, 4, 64, 64)))):
                result = asyncio.run(
                    _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41)
                )
            assert result is not None
        finally:
            self._remove_mock_av()

    def test_video_gen_with_i2v(self):
        from nodes.samplers import _generate_monolithic
        model = _make_mock_model("video")
        model.get_engine.return_value._engine.generate = AsyncMock(
            return_value=[b"\x00" * 100]
        )
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {
            "samples": np.zeros((1, 4, 64, 64), dtype=np.float32),
            "_i2v_image_path": "/tmp/test.png",
            "_i2v_image_strength": 0.8,
        }
        self._install_mock_av(import_error=True)
        try:
            with patch("fusion_comfyui.core.bridge.to_mlx_array", return_value=mx.array(np.zeros((1, 4, 64, 64)))):
                _result = asyncio.run(
                    _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41)
                )
            gen_call = model.get_engine.return_value._engine.generate.call_args
            assert gen_call.kwargs["image"] == "/tmp/test.png"
            assert gen_call.kwargs["image_strength"] == 0.8
        finally:
            self._remove_mock_av()

    def test_video_gen_no_negative(self):
        from nodes.samplers import _generate_monolithic
        model = _make_mock_model("video")
        model.get_engine.return_value._engine.generate = AsyncMock(
            return_value=[b"\x00" * 100]
        )
        positive = {"prompt": "test"}
        negative = None
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        self._install_mock_av(import_error=True)
        try:
            with patch("fusion_comfyui.core.bridge.to_mlx_array", return_value=mx.array(np.zeros((1, 4, 64, 64)))):
                _result = asyncio.run(
                    _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41)
                )
            gen_call = model.get_engine.return_value._engine.generate.call_args
            assert gen_call.kwargs["negative_prompt"] is None
        finally:
            self._remove_mock_av()

    def test_video_gen_empty_negative(self):
        from nodes.samplers import _generate_monolithic
        model = _make_mock_model("video")
        model.get_engine.return_value._engine.generate = AsyncMock(
            return_value=[b"\x00" * 100]
        )
        positive = {"prompt": "test"}
        negative = {"prompt": ""}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        self._install_mock_av(import_error=True)
        try:
            with patch("fusion_comfyui.core.bridge.to_mlx_array", return_value=mx.array(np.zeros((1, 4, 64, 64)))):
                _result = asyncio.run(
                    _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41)
                )
            gen_call = model.get_engine.return_value._engine.generate.call_args
            assert gen_call.kwargs["negative_prompt"] is None
        finally:
            self._remove_mock_av()

    def test_video_gen_av_decode(self):
        from nodes.samplers import _generate_monolithic
        model = _make_mock_model("video")
        fake_bytes = b"fake_mp4_data"
        model.get_engine.return_value._engine.generate = AsyncMock(
            return_value=[fake_bytes]
        )
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        mock_frame1 = MagicMock()
        mock_frame1.to_ndarray.return_value = np.zeros((512, 768, 3), dtype=np.uint8)
        mock_frame2 = MagicMock()
        mock_frame2.to_ndarray.return_value = np.zeros((512, 768, 3), dtype=np.uint8)
        mock_container = MagicMock()
        mock_container.decode.return_value = [mock_frame1, mock_frame2]
        mock_container.close = MagicMock()
        self._install_mock_av(container=mock_container)
        try:
            result = asyncio.run(
                _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41)
            )
            assert isinstance(result, np.ndarray)
            assert result.ndim == 4
            assert result.shape[0] == 2
        finally:
            self._remove_mock_av()

    def test_video_gen_av_exception(self):
        from nodes.samplers import _generate_monolithic
        model = _make_mock_model("video")
        model.get_engine.return_value._engine.generate = AsyncMock(
            return_value=[b"bad_data"]
        )
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        mock_av = MagicMock()
        mock_av.open.side_effect = Exception("decode error")
        import sys
        sys.modules["av"] = mock_av
        try:
            with patch("fusion_comfyui.core.bridge.to_mlx_array", return_value=mx.array(np.zeros((1, 4, 64, 64)))):
                result = asyncio.run(
                    _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41)
                )
            assert result is not None
        finally:
            sys.modules.pop("av", None)

    def test_video_gen_no_frames(self):
        from nodes.samplers import _generate_monolithic
        model = _make_mock_model("video")
        model.get_engine.return_value._engine.generate = AsyncMock(
            return_value=[b"fake"]
        )
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        mock_container = MagicMock()
        mock_container.decode.return_value = []
        mock_container.close = MagicMock()
        self._install_mock_av(container=mock_container)
        try:
            with patch("fusion_comfyui.core.bridge.to_mlx_array", return_value=mx.array(np.zeros((1, 4, 64, 64)))):
                result = asyncio.run(
                    _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41)
                )
            assert result is not None
        finally:
            self._remove_mock_av()

    def test_image_gen(self):
        from nodes.samplers import _generate_monolithic
        from PIL import Image
        model = _make_mock_model("image")
        img = Image.new("RGB", (64, 64), color="red")
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        model.get_engine.return_value._engine.generate = AsyncMock(
            return_value=[buf.getvalue()]
        )
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        result = asyncio.run(
            _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 64, 64, 1)
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3

    def test_image_gen_rgba(self):
        from nodes.samplers import _generate_monolithic
        from PIL import Image
        model = _make_mock_model("image")
        img = Image.new("RGBA", (64, 64), color="red")
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        model.get_engine.return_value._engine.generate = AsyncMock(
            return_value=[buf.getvalue()]
        )
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        result = asyncio.run(
            _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 64, 64, 1)
        )
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3

    def test_unknown_model_type(self):
        from nodes.samplers import _generate_monolithic
        model = _make_mock_model("unknown")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        samples = np.zeros((1, 4, 64, 64), dtype=np.float32)
        latent = {"samples": samples}
        result = asyncio.run(
            _generate_monolithic(model, positive, negative, latent, 20, 6.0, 42, 64, 64, 1)
        )
        np.testing.assert_array_equal(result, samples)


class TestKSampler:
    def test_input_types(self):
        from nodes.samplers import KSampler
        inputs = KSampler.INPUT_TYPES()
        assert "required" in inputs

    def test_sample_ndarray_4d(self):
        from nodes.samplers import KSampler
        model = _make_mock_model("image")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((1, 512, 512, 3), dtype=np.float32)):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert result is not None
        assert "_decoded_frames_key" in result[0]

    def test_sample_ndarray_5d(self):
        from nodes.samplers import KSampler
        model = _make_mock_model("video")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32)}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((4, 512, 768, 3), dtype=np.float32)):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert result is not None

    def test_sample_ndarray_3d(self):
        from nodes.samplers import KSampler
        model = _make_mock_model("image")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((512, 768, 3), dtype=np.float32)):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert result is not None

    def test_sample_mx_array_result(self):
        from nodes.samplers import KSampler
        model = _make_mock_model("image")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = KSampler()
        mx_result = mx.array(np.zeros((1, 4, 64, 64), dtype=np.float32))
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=mx_result):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert result is not None
        assert "samples" in result[0]

    def test_sample_unexpected_result_type(self):
        from nodes.samplers import KSampler
        model = _make_mock_model("image")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value="unexpected"):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert result is not None

    def test_sample_non_fusion_model(self):
        from nodes.samplers import KSampler
        model = MagicMock()
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            with pytest.raises(RuntimeError, match="FusionModelWrapper"):
                node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)

    def test_sample_mx_array_input(self):
        from nodes.samplers import KSampler
        model = _make_mock_model("image")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": mx.array(np.zeros((1, 4, 64, 64), dtype=np.float32))}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((1, 512, 512, 3), dtype=np.float32)):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert result is not None

    def test_sample_with_explicit_dims(self):
        from nodes.samplers import KSampler
        model = _make_mock_model("video")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {
            "samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32),
            "num_frames": 41,
            "width": 768,
            "height": 512,
        }
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((4, 512, 768, 3), dtype=np.float32)):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert result is not None

    def test_sample_i2v_temp_file_preserved(self):
        from nodes.samplers import KSampler
        import tempfile
        model = _make_mock_model("video")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(b"test")
        tmp.close()
        latent = {
            "samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32),
            "_i2v_image_path": tmp.name,
        }
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((4, 512, 768, 3), dtype=np.float32)):
            _result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        import os
        # Temp file is NOT deleted by KSampler — multi-KSampler workflows (e.g. wan22 14B i2v)
        # reuse the same latent dict across stages. OS handles cleanup.
        assert os.path.exists(tmp.name)
        os.unlink(tmp.name)

    def test_sample_i2v_cleanup_missing_file(self):
        from nodes.samplers import KSampler
        model = _make_mock_model("video")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {
            "samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32),
            "_i2v_image_path": "/tmp/nonexistent_file_xyz123.png",
        }
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((4, 512, 768, 3), dtype=np.float32)):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert result is not None

    def test_sample_unknown_shape(self):
        from nodes.samplers import KSampler
        model = _make_mock_model("image")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": "weird"}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((1, 512, 512, 3), dtype=np.float32)):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert result is not None


class TestKSamplerAdvanced:
    def test_input_types(self):
        from nodes.samplers import KSamplerAdvanced
        inputs = KSamplerAdvanced.INPUT_TYPES()
        assert "required" in inputs

    def test_sample_delegates(self):
        from nodes.samplers import KSamplerAdvanced
        model = _make_mock_model("image")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = KSamplerAdvanced()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((1, 512, 512, 3), dtype=np.float32)):
            result = node.sample(model, "enable", 42, 20, 6.0, "euler", "normal",
                                 positive, negative, latent)
        assert result is not None


class TestSamplerCustom:
    def test_input_types(self):
        from nodes.samplers import SamplerCustom
        inputs = SamplerCustom.INPUT_TYPES()
        assert "required" in inputs

    def test_sample(self):
        from nodes.samplers import SamplerCustom
        model = _make_mock_model("image")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        sigmas = np.array([1.0, 0.5, 0.0])
        node = SamplerCustom()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((1, 512, 512, 3), dtype=np.float32)):
            result = node.sample(model, True, 42, 6.0, positive, negative,
                                 "sampler", sigmas, latent)
        assert len(result) == 2

    def test_sample_no_len_sigmas(self):
        from nodes.samplers import SamplerCustom
        model = _make_mock_model("image")
        positive = {"prompt": "test"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = SamplerCustom()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((1, 512, 512, 3), dtype=np.float32)):
            result = node.sample(model, True, 42, 6.0, positive, negative,
                                 "sampler", 42, latent)
        assert len(result) == 2


class TestSamplerCustomAdvanced:
    def test_input_types(self):
        from nodes.samplers import SamplerCustomAdvanced
        inputs = SamplerCustomAdvanced.INPUT_TYPES()
        assert "required" in inputs

    def test_sample(self):
        from nodes.samplers import SamplerCustomAdvanced
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        sigmas = np.array([1.0, 0.5, 0.0])
        noise = {"noise_seed": 42}
        guider = {"model": MagicMock(), "conditioning": {"prompt": "test", "guidance": 6.0}}
        node = SamplerCustomAdvanced()
        result = node.sample(noise, guider, MagicMock(), sigmas, latent)
        assert len(result) == 2
        assert result[0]["samples"] is latent["samples"]


class TestFusionKSamplerNode:
    def test_input_types(self):
        from nodes.samplers import FusionKSamplerNode
        inputs = FusionKSamplerNode.INPUT_TYPES()
        assert "required" in inputs

    def test_sample(self):
        from nodes.samplers import FusionKSamplerNode
        pipeline = MagicMock()
        pipeline.get_memory_stats.return_value = {"active_mb": 100, "peak_mb": 200}
        positive = MagicMock()
        negative = MagicMock()
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = FusionKSamplerNode()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.bridge.to_mlx_array", return_value=mx.array(np.zeros((1, 4, 64, 64)))), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=mx.array(np.zeros((1, 4, 64, 64), dtype=np.float32))):
            result = node.sample(pipeline, positive, negative, latent, 20, 6.0, 42, 1024, 1024, 1)
        assert result is not None

    def test_sample_error(self):
        from nodes.samplers import FusionKSamplerNode
        pipeline = MagicMock()
        pipeline.get_memory_stats.return_value = {"active_mb": 100, "peak_mb": 200}
        positive = MagicMock()
        negative = MagicMock()
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = FusionKSamplerNode()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.bridge.to_mlx_array", return_value=mx.array(np.zeros((1, 4, 64, 64)))), \
             patch("fusion_comfyui.core.async_utils.run_async", side_effect=RuntimeError("denoise fail")):
            with pytest.raises(RuntimeError):
                node.sample(pipeline, positive, negative, latent, 20, 6.0, 42, 1024, 1024, 1)


class TestLatentUpscaleOverride:
    def test_rgb_path_upscales_in_pixel_space(self):
        from nodes.samplers import LatentUpscale
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        samples = {"samples": rgb[np.newaxis, np.newaxis, ...], "_image_init_path": "/tmp/x.png"}
        node = LatentUpscale()
        result = node.upscale(samples, "bicubic", 128, 128, "disabled")
        arr = result[0]["samples"]
        assert arr.ndim == 5 and arr.shape[-1] == 3
        assert arr.shape[-3] == 128 and arr.shape[-2] == 128
        assert result[0]["width"] == 128 and result[0]["height"] == 128
        assert result[0]["num_frames"] == 1

    def test_rgb_path_clamps_out_of_range(self):
        from nodes.samplers import LatentUpscale
        rgb = np.full((32, 32, 3), 2.0, dtype=np.float32)
        samples = {"samples": rgb[np.newaxis, np.newaxis, ...], "_image_init_path": "/tmp/x.png"}
        node = LatentUpscale()
        result = node.upscale(samples, "bilinear", 64, 64, "disabled")
        assert result[0]["samples"].max() <= 1.0

    def test_true_latent_path_uses_common_upscale(self):
        from nodes.samplers import LatentUpscale
        latent = np.zeros((1, 4, 64, 64), dtype=np.float32)
        samples = {"samples": latent}
        node = LatentUpscale()
        fake_common_upscale = MagicMock(return_value=latent)
        with patch("nodes._scaling.common_upscale", fake_common_upscale):
            node.upscale(samples, "bicubic", 512, 512, "disabled")
        assert fake_common_upscale.called
        args = fake_common_upscale.call_args[0]
        assert args[1] == 64 and args[2] == 64


def test_latent_upscale_true_latent_numpy():
    from nodes.samplers import LatentUpscale
    latent = {
        "samples": np.random.default_rng(3).random((1, 1, 4, 16, 16)).astype(np.float32),
    }
    out = LatentUpscale().upscale(latent, "bilinear", 128, 128, "disabled")
    assert out[0]["samples"].shape == (1, 1, 4, 16, 16)
    assert isinstance(out[0]["samples"], np.ndarray)
