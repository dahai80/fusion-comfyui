import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock


class TestInferModelType:
    def test_wan(self):
        from fusion_comfyui.core.engine_wrapper import _infer_model_type
        assert _infer_model_type("Wan2.2-5B") == "video"

    def test_flux(self):
        from fusion_comfyui.core.engine_wrapper import _infer_model_type
        assert _infer_model_type("FLUX.2-dev") == "image"

    def test_unknown(self):
        from fusion_comfyui.core.engine_wrapper import _infer_model_type
        assert _infer_model_type("something-else") == "image"


class TestFusionEngineWrapper:
    def test_init_defaults(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        assert e.model_name == "test-model"
        assert e.offload_strategy == "sequential"
        assert e.quant_bit == "fp8_e4m3"
        assert e.model_type == "image"
        assert e._started is False

    def test_init_video_model(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="Wan2.2-5B")
        assert e.model_type == "video"

    def test_set_progress_callback(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        cb = MagicMock()
        e.set_progress_callback(cb)
        assert e._on_step is cb

    def test_load_stage(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        result = e.load_stage("encode")
        assert result is e

    def test_unload_stage(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e.unload_stage("dit")

    def test_get_memory_stats(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        with patch("mlx.core.metal.get_active_memory", return_value=100 * 1024 * 1024), \
             patch("mlx.core.metal.get_peak_memory", return_value=200 * 1024 * 1024):
            stats = e.get_memory_stats()
            assert stats["active_mb"] == 100.0
            assert stats["peak_mb"] == 200.0
            assert stats["model_name"] == "test-model"
            assert stats["started"] is False

    def test_stop_not_started(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        result = e.stop()
        import inspect
        assert inspect.iscoroutine(result)
        result.close()

    def test_stop_started(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        e._engine.stop = AsyncMock()
        import asyncio
        asyncio.run(e.stop())
        assert e._started is False

    def test_ensure_started_image(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="flux2-model", quant_bit="fp8_e4m3")
        with patch("fusion_mlx.public_api.ImageGenEngine") as MockEngine:
            mock_inst = MagicMock()
            mock_inst.start = AsyncMock()
            MockEngine.return_value = mock_inst
            import asyncio
            asyncio.run(e.ensure_started())
            assert e._started is True
            MockEngine.assert_called_once()

    def test_ensure_started_video(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="Wan2.2-5B", quant_bit="fp8_e4m3")
        with patch("fusion_mlx.public_api.VideoGenEngine") as MockEngine:
            mock_inst = MagicMock()
            mock_inst.start = AsyncMock()
            MockEngine.return_value = mock_inst
            import asyncio
            asyncio.run(e.ensure_started())
            assert e._started is True

    def test_ensure_started_already(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        import asyncio
        asyncio.run(e.ensure_started())
        assert e._started is True

    def test_load_text_encoder(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        e._engine.load_text_encoder = AsyncMock()
        import asyncio
        asyncio.run(e.load_text_encoder())

    def test_encode_text(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        embed = MagicMock()
        embed.shape = (1, 77, 768)
        e._engine.encode_text = AsyncMock(return_value={"embed": embed})
        import asyncio
        result = asyncio.run(e.encode_text("hello", "bad"))
        assert result["negative_prompt"] == "bad"

    def test_unload_text_encoder(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        e._engine.unload_text_encoder = AsyncMock()
        import asyncio
        asyncio.run(e.unload_text_encoder())

    def test_load_dit(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        e._engine.load_dit = AsyncMock()
        import asyncio
        asyncio.run(e.load_dit())

    def test_denoise_with_embed(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        mock_result = MagicMock()
        mock_result.shape = (1, 4, 64, 64)
        e._engine.denoise = AsyncMock(return_value=mock_result)
        positive = {"embed": MagicMock()}
        negative = {"embed": MagicMock()}
        import asyncio
        result = asyncio.run(e.denoise(MagicMock(), positive, negative, steps=20, cfg=6.0, seed=42))
        assert result is not None

    def test_denoise_video(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="Wan2.2-5B")
        e._started = True
        e._engine = MagicMock()
        mock_result = MagicMock()
        mock_result.shape = (1, 16, 11, 64, 64)
        e._engine.denoise = AsyncMock(return_value=mock_result)
        positive = {"embed": MagicMock()}
        negative = {"embed": MagicMock()}
        import asyncio
        _result = asyncio.run(e.denoise(MagicMock(), positive, negative, steps=20, cfg=6.0, seed=42, num_frames=41))

    def test_denoise_fallback_no_embed(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        e._engine.generate = AsyncMock(return_value=[b"\x89PNG" + b"\x00" * 100])
        positive = {"prompt": "hello"}
        negative = {"negative_prompt": "bad", "embed": MagicMock()}
        import asyncio
        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img_arr = np.zeros((512, 512, 3), dtype=np.float32)
            mock_open.return_value = mock_img
            with patch("numpy.array", return_value=mock_img_arr):
                _result = asyncio.run(e.denoise(MagicMock(), positive, negative, steps=20, cfg=6.0, seed=42, width=512, height=512))

    def test_denoise_fallback_video(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="Wan2.2-5B")
        e._started = True
        e._engine = MagicMock()
        e._engine.generate = AsyncMock(return_value=[b"\x00" * 100])
        positive = {"prompt": "hello"}
        negative = {"negative_prompt": "bad", "embed": MagicMock()}
        import asyncio
        mock_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_reader = MagicMock()
        mock_reader.__iter__ = MagicMock(return_value=iter([mock_frame]))
        mock_reader.close = MagicMock()
        with patch("imageio.get_reader", return_value=mock_reader):
            _result = asyncio.run(e.denoise(MagicMock(), positive, negative, steps=20, cfg=6.0, seed=42, num_frames=4))

    def test_unload_dit(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        e._engine.unload_dit = AsyncMock()
        import asyncio
        asyncio.run(e.unload_dit())

    def test_load_vae(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        e._engine.load_vae = AsyncMock()
        import asyncio
        asyncio.run(e.load_vae())

    def test_decode(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        mock_result = MagicMock()
        mock_result.shape = (1, 3, 512, 512)
        e._engine.decode = AsyncMock(return_value=mock_result)
        import asyncio
        result = asyncio.run(e.decode(MagicMock()))
        assert result is mock_result

    def test_decode_tiled(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        mock_result = MagicMock()
        mock_result.shape = (1, 3, 512, 512)
        e._engine.decode_tiled = AsyncMock(return_value=mock_result)
        import asyncio
        _result = asyncio.run(e.decode_tiled(MagicMock(), tile_size=256))

    def test_decode_tiled_no_method(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock(spec=["decode", "start"])
        mock_result = MagicMock()
        mock_result.shape = (1, 3, 512, 512)
        e._engine.decode = AsyncMock(return_value=mock_result)
        import asyncio
        _result = asyncio.run(e.decode_tiled(MagicMock(), tile_size=256))

    def test_unload_vae(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        e._started = True
        e._engine = MagicMock()
        e._engine.unload_vae = AsyncMock()
        import asyncio
        asyncio.run(e.unload_vae())

    def test_stage_context(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        e = FusionEngineWrapper(model_name="test-model")
        ctx = e.stage("dit")
        assert ctx is not None
