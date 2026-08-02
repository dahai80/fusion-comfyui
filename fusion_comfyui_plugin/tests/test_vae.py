import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _run_coro(coro):
    import asyncio
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


class TestVAEDecode:
    def test_input_types(self):
        from nodes.vae import VAEDecode
        inputs = VAEDecode.INPUT_TYPES()
        assert "required" in inputs

    def test_decode_with_decoded_frames_4d(self):
        from nodes.vae import VAEDecode
        frames = np.zeros((1, 512, 512, 3), dtype=np.float32)
        samples = {"_decoded_frames": frames}
        mock_vae = MagicMock()
        node = VAEDecode()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            result = node.decode(mock_vae, samples)
        assert result is not None

    def test_decode_with_decoded_frames_3d(self):
        from nodes.vae import VAEDecode
        frames = np.zeros((512, 512, 3), dtype=np.float32)
        samples = {"_decoded_frames": frames}
        mock_vae = MagicMock()
        node = VAEDecode()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            result = node.decode(mock_vae, samples)
        assert result is not None

    def test_decode_with_decoded_frames_5d(self):
        from nodes.vae import VAEDecode
        frames = np.zeros((2, 4, 512, 512, 3), dtype=np.float32)
        samples = {"_decoded_frames": frames}
        mock_vae = MagicMock()
        node = VAEDecode()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            result = node.decode(mock_vae, samples)
        assert result is not None

    def test_decode_with_decoded_frames_mx_array(self):
        from nodes.vae import VAEDecode
        import sys
        MxArray = sys.modules["mlx.core"].array
        mx_frames = MxArray(np.zeros((1, 512, 512, 3), dtype=np.float32))
        samples = {"_decoded_frames": mx_frames}
        mock_vae = MagicMock()
        node = VAEDecode()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            result = node.decode(mock_vae, samples)
        assert result is not None

    def test_decode_requires_fusion_vae(self):
        from nodes.vae import VAEDecode
        mock_vae = MagicMock(spec=[])
        samples = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = VAEDecode()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            with pytest.raises(RuntimeError, match="FusionVAEWrapper"):
                node.decode(mock_vae, samples)

    def _make_fusion_vae_mock(self, **kw):
        from core.wrappers import FusionVAEWrapper
        mock_vae = MagicMock(spec=FusionVAEWrapper)
        mock_vae.get_engine = MagicMock(**kw)
        return mock_vae

    def test_decode_via_engine_mx_result(self):
        from nodes.vae import VAEDecode
        import sys
        MxArray = sys.modules["mlx.core"].array
        mx_result = MxArray(np.zeros((1, 3, 512, 512), dtype=np.float32))
        mock_vae = self._make_fusion_vae_mock()
        mock_vae.model_name = "test"
        samples = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = VAEDecode()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("core.bridge.to_mlx_array", return_value=mx_result), \
             patch("core.bridge.to_image_array", return_value=np.zeros((1, 512, 512, 3))), \
             patch("core.async_utils.run_async", return_value=mx_result):
            result = node.decode(mock_vae, samples)
        assert result is not None

    def test_decode_via_engine_np_result(self):
        from nodes.vae import VAEDecode
        mock_vae = self._make_fusion_vae_mock()
        mock_vae.model_name = "test"
        samples = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = VAEDecode()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("core.bridge.to_mlx_array", return_value=MagicMock()), \
             patch("core.async_utils.run_async", return_value=np.zeros((1, 512, 512, 3), dtype=np.float32)):
            result = node.decode(mock_vae, samples)
        assert result is not None

    def test_decode_fallback_not_implemented(self):
        from nodes.vae import VAEDecode
        import sys
        MxArray = sys.modules["mlx.core"].array
        mx_latent = MxArray(np.zeros((1, 4, 64, 64), dtype=np.float32))
        mock_vae = self._make_fusion_vae_mock()
        mock_vae.model_name = "test"
        samples = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = VAEDecode()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("core.bridge.to_mlx_array", return_value=mx_latent), \
             patch("core.async_utils.run_async", return_value=mx_latent), \
             patch("core.bridge.to_image_array", return_value=np.zeros((1, 512, 512, 3))):
            result = node.decode(mock_vae, samples)
        assert result is not None


class TestVAEDecodeTiled:
    def test_input_types(self):
        from nodes.vae import VAEDecodeTiled
        inputs = VAEDecodeTiled.INPUT_TYPES()
        assert "required" in inputs

    def test_delegates_to_vae_decode(self):
        from nodes.vae import VAEDecodeTiled
        frames = np.zeros((1, 512, 512, 3), dtype=np.float32)
        samples = {"_decoded_frames": frames}
        mock_vae = MagicMock()
        node = VAEDecodeTiled()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            result = node.decode(mock_vae, samples, tile_size=256)
        assert result is not None


class TestFusionVAEDecoderNode:
    def test_input_types(self):
        from nodes.vae import FusionVAEDecoderNode
        inputs = FusionVAEDecoderNode.INPUT_TYPES()
        assert "required" in inputs

    def test_decode(self):
        from nodes.vae import FusionVAEDecoderNode
        import sys
        MxArray = sys.modules["mlx.core"].array
        mx_result = MxArray(np.zeros((1, 3, 512, 512), dtype=np.float32))
        pipeline = MagicMock()
        pipeline.get_memory_stats.return_value = {"active_mb": 100, "peak_mb": 200}
        samples = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = FusionVAEDecoderNode()
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("core.bridge.to_mlx_array", return_value=MagicMock()), \
             patch("core.bridge.to_image_array", return_value=np.zeros((1, 512, 512, 3))), \
             patch("core.async_utils.run_async", return_value=mx_result):
            result = node.decode(pipeline, samples, tile_sample_min_size=256)
        assert result is not None
