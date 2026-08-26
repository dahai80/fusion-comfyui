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


class TestRunStagedPipeline:
    def _make_wrapper(self, model_type="video"):
        import mlx.core as mx
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        # spec dropped: _run_staged_pipeline is NEW, spec would block it
        w = MagicMock()
        w.model_type = model_type
        w.load_text_encoder = AsyncMock()
        w.encode_text = AsyncMock(side_effect=lambda p, neg="": {"embed": mx.array(np.zeros((1, 256), dtype=np.float32))})
        w.unload_text_encoder = AsyncMock()
        w.load_dit = AsyncMock()
        w.denoise = AsyncMock(return_value=mx.array(np.zeros((1, 16, 5, 32, 32), dtype=np.float32)))
        w.unload_dit = AsyncMock()
        w.load_vae = AsyncMock()
        w.decode = AsyncMock(return_value=mx.array(np.zeros((4, 512, 768, 3), dtype=np.float32)))
        w.unload_vae = AsyncMock()
        # bind the real method onto the mock so it calls the mocks above
        w._run_staged_pipeline = FusionEngineWrapper._run_staged_pipeline.__get__(w)
        return w

    def test_video_full_stage_order_and_purge(self):
        import asyncio
        import mlx.core as mx
        w = self._make_wrapper("video")
        with patch("fusion_comfyui.core.engine_wrapper.FusionMemoryGuardian.purge_memory") as purge:
            result = asyncio.run(w._run_staged_pipeline(
                mx.array(np.zeros((1, 16, 5, 32, 32))), "cat", "dog", 20, 6.0, 42, num_frames=41))
        assert isinstance(result, mx.array)
        # Stage call order
        w.load_text_encoder.assert_awaited_once()
        assert w.encode_text.await_count == 2  # pos + neg (cfg>1)
        w.unload_text_encoder.assert_awaited_once()
        w.load_dit.assert_awaited_once()
        w.denoise.assert_awaited_once()
        w.unload_dit.assert_awaited_once()
        w.load_vae.assert_awaited_once()
        w.decode.assert_awaited_once()
        w.unload_vae.assert_awaited_once()
        # purge between each of the 3 stages
        assert purge.call_count == 3

    def test_cfg_le_1_skips_negative_encode(self):
        import asyncio
        import mlx.core as mx
        w = self._make_wrapper("video")
        asyncio.run(w._run_staged_pipeline(
            mx.array(np.zeros((1, 16, 5, 32, 32))), "cat", "dog", 20, 1.0, 42, num_frames=41))
        assert w.encode_text.await_count == 1  # positive only
        # denoise called with neg_cond=None
        denoise_args = w.denoise.await_args
        assert denoise_args.args[2] is None or denoise_args.kwargs.get("negative") is None

    def test_image_pipeline_no_num_frames(self):
        import asyncio
        import mlx.core as mx
        w = self._make_wrapper("image")
        result = asyncio.run(w._run_staged_pipeline(
            mx.array(np.zeros((1, 16, 32, 32))), "cat", "dog", 20, 6.0, 42))
        w.denoise.assert_awaited_once()
        w.decode.assert_awaited_once()
        assert isinstance(result, mx.array)


class TestStageContext:
    def test_default_fields_none(self):
        from fusion_comfyui.core.stage_context import StageContext
        ctx = StageContext(model_wrapper=object())
        assert ctx.latent is None
        assert ctx.pos_cond is None
        assert ctx.neg_cond is None
        assert ctx.pixels is None
        assert ctx.model_type == "video"

    def test_construct_with_fields(self):
        import mlx.core as mx
        from fusion_comfyui.core.stage_context import StageContext
        latent = mx.array(np.zeros((1, 16, 5, 8, 8), dtype=np.float32))
        ctx = StageContext(
            model_wrapper=object(), latent=latent, model_type="image",
        )
        assert ctx.latent is latent
        assert ctx.model_type == "image"

    def test_staged_pipeline_populates_ctx_fields(self):
        # white-box: _run_staged_pipeline threads data through a StageContext.
        # Capture the ctx via a wrapper that records what denoise/decode receive.
        import asyncio
        import mlx.core as mx
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper

        w = MagicMock()
        w.model_type = "video"
        pos_embed = {"embed": mx.array(np.zeros((1, 256), dtype=np.float32))}
        neg_embed = {"embed": mx.array(np.zeros((1, 256), dtype=np.float32))}
        w.load_text_encoder = AsyncMock()
        w.encode_text = AsyncMock(side_effect=lambda p, neg="": pos_embed if p == "cat" else neg_embed)
        w.unload_text_encoder = AsyncMock()
        w.load_dit = AsyncMock()
        denoised = mx.array(np.zeros((1, 16, 5, 32, 32), dtype=np.float32))
        w.denoise = AsyncMock(return_value=denoised)
        w.unload_dit = AsyncMock()
        w.load_vae = AsyncMock()
        pixels = mx.array(np.zeros((4, 512, 768, 3), dtype=np.float32))
        w.decode = AsyncMock(return_value=pixels)
        w.unload_vae = AsyncMock()
        w._run_staged_pipeline = FusionEngineWrapper._run_staged_pipeline.__get__(w)

        in_latent = mx.array(np.zeros((1, 16, 5, 32, 32), dtype=np.float32))
        result = asyncio.run(w._run_staged_pipeline(in_latent, "cat", "dog", 20, 6.0, 42, num_frames=41))

        # denoise received pos_cond and neg_cond (dicts with embed), threaded via ctx
        dargs = w.denoise.await_args
        assert dargs.args[1] is pos_embed, "pos_cond not threaded to denoise via ctx"
        assert dargs.args[2] is neg_embed, "neg_cond not threaded to denoise via ctx"
        # decode received the denoised latent threaded via ctx
        cargs = w.decode.await_args
        assert cargs.args[0] is denoised, "denoised latent not threaded to decode via ctx"
        # result is the pixels threaded via ctx
        assert result is pixels, "pixels not returned from ctx"
