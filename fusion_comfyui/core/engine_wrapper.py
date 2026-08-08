import gc
import logging
from typing import Any, Callable, Awaitable

import mlx.core as mx

from fusion_comfyui.core.config import load_config, RadixCache
from fusion_comfyui.core.lifecycle import FusionMemoryGuardian
from fusion_comfyui.core.timer import NodeTimer

logger = logging.getLogger("fusion_comfyui.core.engine_wrapper")

StepCallback = Callable[[int, int], Awaitable[None]]

_MODEL_TYPES = {
    "flux2": "image",
    "flux": "image",
    "wan2": "video",
    "wan": "video",
    "skyreels": "video",
    "ltx": "video",
    "cosmos": "video",
    "hunyuan": "video",
    "svd": "video",
}

_LATENT_CHANNELS = {
    "flux2": 16,
    "flux": 4,
    "wan2": 16,
    "wan": 16,
    "skyreels": 16,
    "ltx": 16,
    "cosmos": 16,
    "hunyuan": 16,
    "svd": 4,
}


def _infer_model_type(model_name: str) -> str:
    name = model_name.lower()
    for key, mtype in _MODEL_TYPES.items():
        if key in name:
            return mtype
    return "image"


def _get_latent_channels(model_name: str) -> int:
    name = model_name.lower()
    for key, ch in _LATENT_CHANNELS.items():
        if key in name:
            return ch
    return 4


def _get_latent_shape(model_name: str, height: int, width: int, num_frames: int = 1) -> tuple:
    channels = _get_latent_channels(model_name)
    is_video = _infer_model_type(model_name) == "video"
    t = num_frames if is_video else 1
    if t > 1:
        return (1, t, channels, height // 8, width // 8)
    return (1, channels, height // 8, width // 8)


async def unload_all_fusion_engines():
    """Unload all fusion-mlx engine pool entries and clear Metal cache.

    Call this at pipeline startup to ensure a clean slate before any node runs.
    """
    logger.info("unload_all_fusion_engines: starting full unload")
    try:
        from fusion_mlx.pool.engine_pool import EnginePool
        pool = EnginePool.get_instance()
        if pool:
            await pool.shutdown()
            logger.info("unload_all_fusion_engines: EnginePool.shutdown() done")
    except Exception as e:
        logger.debug("EnginePool.shutdown() skipped: %s", e)
    gc.collect()
    mx.metal.clear_cache()
    active = mx.metal.get_active_memory() / 1024 / 1024
    logger.info("unload_all_fusion_engines: complete, active_mem=%.0fMB", active)


class FusionEngineWrapper:
    def __init__(self, model_name: str, offload_strategy: str = "sequential", quant_bit: str = "fp8_e4m3"):
        self.model_name = model_name
        self.offload_strategy = offload_strategy
        self.quant_bit = quant_bit
        self.model_type = _infer_model_type(model_name)
        self._engine = None
        self._started = False
        self._p3config = load_config()
        self._radix_cache = RadixCache(self._p3config.radix_cache_max_mb) if self._p3config.radix_cache_enabled else None
        self._on_step: StepCallback | None = None
        self._pulid_pipeline = None
        self._lipsync_pipeline = None
        self._musetalk_pipeline = None
        self._tts_engine = None
        self._vlm_engine = None
        logger.info(
            "FusionEngineWrapper: model=%s type=%s offload=%s quant=%s",
            model_name, self.model_type, offload_strategy, quant_bit,
        )

    def set_progress_callback(self, cb: StepCallback | None):
        self._on_step = cb

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_text_encoder_loaded(self) -> bool:
        if not self._started or not self._engine:
            return False
        return getattr(self._engine, "_text_encoder_loaded", False)

    @property
    def is_dit_loaded(self) -> bool:
        if not self._started or not self._engine:
            return False
        return getattr(self._engine, "_dit_loaded", False)

    @property
    def is_vae_loaded(self) -> bool:
        if not self._started or not self._engine:
            return False
        return getattr(self._engine, "_vae_loaded", False)

    def last_denoise_stats(self) -> dict:
        if not self._engine or not self._started:
            return {"available": False, "enabled": False}
        fn = getattr(self._engine, "last_denoise_stats", None)
        if fn is None:
            return {"available": False, "enabled": False}
        try:
            return fn()
        except Exception as e:
            logger.warning("last_denoise_stats failed: %s", e)
            return {"available": False, "enabled": False, "error": str(e)}

    async def ensure_started(self):
        if self._started:
            return
        async with NodeTimer.timed(self.model_name, "engine_start"):
            model_path = self._resolve_model_path()
            if self.model_type == "image":
                from fusion_mlx.engines.image_gen import ImageGenEngine
                quantize = None
                if self.quant_bit in ("4bit", "w4", "nf4"):
                    quantize = 4
                elif self.quant_bit in ("8bit", "w8a16", "fp8_e4m3"):
                    quantize = 8
                self._engine = ImageGenEngine(model_path, quantize=quantize)
            elif self.model_type == "video":
                from fusion_mlx.engines.video import VideoGenEngine
                self._engine = VideoGenEngine(model_path)
            await self._engine.start()
            self._started = True
            logger.info("Engine started: %s", self.model_name)

    def _resolve_model_path(self) -> str:
        import os
        if os.path.isdir(self.model_name):
            return self.model_name
        try:
            from fusion_mlx.model_registry import list_available_models
            model_type = "image" if self.model_type == "image" else "video"
            for m in list_available_models(model_type):
                if m["name"] == self.model_name:
                    return m["path"]
        except Exception:
            pass
        return self.model_name

    async def load_text_encoder(self):
        async with NodeTimer.timed(self.model_name, "load_text_encoder"):
            await self.ensure_started()
            await self._engine.load_text_encoder()
            logger.info("stage loaded: text_encoder for %s", self.model_name)

    async def encode_text(self, prompt: str, negative_prompt: str = "") -> dict:
        async with NodeTimer.timed(self.model_name, "encode_text", prompt_len=len(prompt)):
            await self.ensure_started()
            result = await self._engine.encode_text(prompt)
            logger.info("encode_text: prompt_len=%d", len(prompt))
            if not isinstance(result, dict):
                result = {"embed": result}
            result["negative_prompt"] = negative_prompt
            result["prompt"] = prompt
            return result

    async def unload_text_encoder(self):
        async with NodeTimer.timed(self.model_name, "unload_text_encoder"):
            if self._engine and self._started:
                await self._engine.unload_text_encoder()
                logger.info("stage unloaded: text_encoder for %s", self.model_name)
                FusionMemoryGuardian.purge_memory()

    async def load_dit(self):
        async with NodeTimer.timed(self.model_name, "load_dit"):
            await self.ensure_started()
            await self._engine.load_dit()
            logger.info("stage loaded: dit for %s", self.model_name)

    async def denoise(self, latent, positive, negative, steps=20, cfg=6.0, seed=0, **kwargs):
        async with NodeTimer.timed(self.model_name, "denoise", steps=steps, cfg=cfg, seed=seed):
            await self.ensure_started()
            pos_embed = positive.get("embed") if isinstance(positive, dict) else positive
            neg_embed = None
            if negative:
                neg_embed = negative.get("embed") if isinstance(negative, dict) else negative

            if pos_embed is None:
                logger.warning("no embed in positive conditioning, falling back to generate()")
                return await self._fallback_generate(latent, positive, negative, steps, cfg, seed, **kwargs)

            if self.model_type == "video":
                num_frames = kwargs.get("num_frames", 97)
                result = await self._engine.denoise(
                    latent, pos_embed, neg_embed, steps, cfg, seed, num_frames,
                )
            else:
                result = await self._engine.denoise(
                    latent, pos_embed, neg_embed, steps, cfg, seed,
                )
            logger.info("denoise: steps=%d cfg=%.1f seed=%d out_shape=%s", steps, cfg, seed, tuple(result.shape))
            return result

    async def unload_dit(self):
        async with NodeTimer.timed(self.model_name, "unload_dit"):
            if self._engine and self._started:
                await self._engine.unload_dit()
                logger.info("stage unloaded: dit for %s", self.model_name)
                FusionMemoryGuardian.purge_memory()

    async def load_vae(self):
        async with NodeTimer.timed(self.model_name, "load_vae"):
            await self.ensure_started()
            await self._engine.load_vae()
            logger.info("stage loaded: vae for %s", self.model_name)

    async def decode(self, latent, tile_size=256):
        async with NodeTimer.timed(self.model_name, "decode"):
            await self.ensure_started()
            result = await self._engine.decode(latent)
            logger.info("decode: out_shape=%s", tuple(result.shape))
            return result

    async def decode_tiled(self, latent, tile_size=256):
        async with NodeTimer.timed(self.model_name, "decode_tiled", tile_size=tile_size):
            await self.ensure_started()
            if hasattr(self._engine, "decode_tiled"):
                result = await self._engine.decode_tiled(latent, tile_size=tile_size)
            else:
                result = await self._engine.decode(latent)
            logger.info("decode_tiled: tile_size=%d out_shape=%s", tile_size, tuple(result.shape))
            return result

    async def unload_vae(self):
        async with NodeTimer.timed(self.model_name, "unload_vae"):
            if self._engine and self._started:
                await self._engine.unload_vae()
                logger.info("stage unloaded: vae for %s", self.model_name)
                FusionMemoryGuardian.purge_memory()

    async def generate_i2v(self, prompt: str, image_path: str, num_frames: int = 49, seed: int = 0, **kwargs):
        async with NodeTimer.timed(self.model_name, "generate_i2v", frames=num_frames, seed=seed):
            await self.ensure_started()
            logger.info("generate_i2v: prompt=%dchars image=%s frames=%d seed=%d", len(prompt), image_path, num_frames, seed)
            if self.model_type != "video":
                logger.warning("generate_i2v called on non-video model %s", self.model_name)
            result_bytes = await self._engine.generate(
                prompt=prompt,
                num_frames=num_frames,
                seed=seed,
                n=1,
                on_step=self._on_step,
                image=image_path,
            )
            import tempfile
            import numpy as np
            tmp_path = None
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(result_bytes[0])
                tmp_path = tmp.name
            try:
                import imageio
                reader = imageio.get_reader(tmp_path, "ffmpeg")
                frames = [np.array(f).astype(np.float32) / 255.0 for f in reader]
                reader.close()
            except ImportError:
                import os
                os.unlink(tmp_path)
                logger.warning("imageio not available for I2V decode")
                return None
            import os
            os.unlink(tmp_path)
            if not frames:
                return None
            logger.info("generate_i2v: decoded %d frames", len(frames))
            return mx.array(np.stack(frames, axis=0))

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 3.5,
        seed: int = 0,
    ) -> bytes:
        async with NodeTimer.timed(self.model_name, "generate_image", steps=steps, cfg=cfg, seed=seed, width=width, height=height):
            await self.ensure_started()
            result_bytes = await self._engine.generate(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                guidance=cfg,
                n_images=1,
                negative_prompt=negative_prompt or None,
                on_step=self._on_step,
            )
            logger.info("generate_image: %dx%d steps=%d seed=%d", width, height, steps, seed)
            return result_bytes[0]

    async def _fallback_generate(self, latent, positive, negative, steps, cfg, seed, **kwargs):
        logger.info("fallback: using monolithic generate() for %s", self.model_name)
        prompt = positive.get("prompt", "") if isinstance(positive, dict) else ""
        negative.get("negative_prompt", "") if isinstance(negative, dict) else ""
        if self.model_type == "image":
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 1024)
            result_bytes = await self._engine.generate(
                prompt=prompt, width=width, height=height,
                steps=steps, seed=seed, guidance=cfg, n_images=1,
                on_step=self._on_step,
            )
            import io
            import numpy as np
            from PIL import Image
            img = Image.open(io.BytesIO(result_bytes[0]))
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
            return mx.array(arr)
        elif self.model_type == "video":
            num_frames = kwargs.get("num_frames", 97)
            width = kwargs.get("width", 768)
            height = kwargs.get("height", 512)
            result_bytes = await self._engine.generate(
                prompt=prompt, num_frames=num_frames, width=width, height=height,
                seed=seed, n=1,
                on_step=self._on_step,
            )
            import tempfile
            import numpy as np
            tmp_path = None
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(result_bytes[0])
                tmp_path = tmp.name
            try:
                import imageio
                reader = imageio.get_reader(tmp_path, "ffmpeg")
                frames = [np.array(f).astype(np.float32) / 255.0 for f in reader]
                reader.close()
            except ImportError:
                import os
                os.unlink(tmp_path)
                logger.warning("imageio not available, returning raw latent")
                return latent
            import os
            os.unlink(tmp_path)
            if not frames:
                return latent
            return mx.array(np.stack(frames, axis=0))
        return latent

    def get_latent_shape(self, height: int, width: int, num_frames: int = 1) -> tuple:
        return _get_latent_shape(self.model_name, height, width, num_frames)

    def create_empty_latent(self, height: int, width: int, num_frames: int = 1) -> mx.array:
        shape = self.get_latent_shape(height, width, num_frames)
        logger.info("create_empty_latent: shape=%s model=%s", shape, self.model_name)
        return mx.zeros(shape, dtype=mx.float16)

    def load_stage(self, stage_name: str):
        logger.info("load_stage: %s (use async load_* methods)", stage_name)
        return self

    def unload_stage(self, stage_name: str):
        logger.info("unload_stage: %s", stage_name)
        gc.collect()
        mx.metal.clear_cache()

    def stage(self, stage_name: str):
        from .lifecycle import PipelineStageContext
        return PipelineStageContext(self, stage_name)

    # ── PuLID lifecycle ──────────────────────────────────────

    async def load_pulid(self, model_dir: str):
        async with NodeTimer.timed("PuLID", "load_pipeline", model_dir=model_dir):
            from fusion_mlx.video.pulid_mlx import PuLIDPipeline
            self._pulid_pipeline = PuLIDPipeline.from_pretrained(model_dir)
            logger.info("PuLID pipeline loaded from %s", model_dir)

    @property
    def is_pulid_loaded(self) -> bool:
        return self._pulid_pipeline is not None

    async def pulid_extract_id(self, bgr_image) -> Any:
        async with NodeTimer.timed("PuLID", "extract_id_embedding"):
            if not self._pulid_pipeline:
                raise RuntimeError("PuLID pipeline not loaded, call load_pulid() first")
            id_embed = self._pulid_pipeline.extract_id_embedding(bgr_image)
            logger.info("PuLID extract_id_embedding: shape=%s", tuple(id_embed.shape))
            return id_embed

    async def pulid_setup_attn(self, dit_model):
        async with NodeTimer.timed("PuLID", "setup_attn_processors"):
            if not self._pulid_pipeline:
                raise RuntimeError("PuLID pipeline not loaded")
            self._pulid_pipeline.setup_attn_processors(dit_model)
            logger.info("PuLID setup_attn_processors done")

    async def pulid_inject_id(self, id_embedding):
        async with NodeTimer.timed("PuLID", "inject_id"):
            if not self._pulid_pipeline:
                raise RuntimeError("PuLID pipeline not loaded")
            self._pulid_pipeline.inject_id(id_embedding)
            logger.info("PuLID inject_id done")

    async def pulid_clear_id(self):
        if self._pulid_pipeline:
            self._pulid_pipeline.clear_id()
            logger.info("PuLID clear_id done")

    async def unload_pulid(self):
        async with NodeTimer.timed("PuLID", "unload_pipeline"):
            self._pulid_pipeline = None
            gc.collect()
            mx.metal.clear_cache()
            logger.info("PuLID pipeline unloaded")

    # ── LatentSync lifecycle ─────────────────────────────────

    async def load_lipsync(self, model_dir: str):
        async with NodeTimer.timed("LatentSync", "load_pipeline", model_dir=model_dir):
            from fusion_mlx.video.latentsync_mlx import LipsyncPipelineMLX
            self._lipsync_pipeline = LipsyncPipelineMLX.from_pretrained(model_dir)
            logger.info("LatentSync pipeline loaded from %s", model_dir)

    @property
    def is_lipsync_loaded(self) -> bool:
        return self._lipsync_pipeline is not None

    async def lipsync_run(self, video_path: str, audio_path: str, output_path: str, **kwargs):
        async with NodeTimer.timed("LatentSync", "run", video=video_path, audio=audio_path):
            if not self._lipsync_pipeline:
                raise RuntimeError("LatentSync pipeline not loaded, call load_lipsync() first")
            self._lipsync_pipeline(
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=output_path,
                **kwargs,
            )
            logger.info("LatentSync run: output=%s", output_path)

    async def unload_lipsync(self):
        async with NodeTimer.timed("LatentSync", "unload_pipeline"):
            self._lipsync_pipeline = None
            gc.collect()
            mx.metal.clear_cache()
            logger.info("LatentSync pipeline unloaded")

    # ── MuseTalk lifecycle ───────────────────────────────────

    async def load_musetalk(self, weights_root: str = "", dist_dir: str = ""):
        async with NodeTimer.timed("MuseTalk", "load_pipeline"):
            from fusion_mlx.video.musetalk_mlx import MuseTalkPipeline
            if dist_dir:
                self._musetalk_pipeline = MuseTalkPipeline.from_pretrained_mlx(dist_dir)
            else:
                self._musetalk_pipeline = MuseTalkPipeline.from_pretrained(weights_root)
            logger.info("MuseTalk pipeline loaded")

    @property
    def is_musetalk_loaded(self) -> bool:
        return self._musetalk_pipeline is not None

    async def unload_musetalk(self):
        async with NodeTimer.timed("MuseTalk", "unload_pipeline"):
            self._musetalk_pipeline = None
            gc.collect()
            mx.metal.clear_cache()
            logger.info("MuseTalk pipeline unloaded")

    # ── TTS lifecycle ────────────────────────────────────────

    async def load_tts(self, model_name: str):
        async with NodeTimer.timed("TTS", "load_engine", model=model_name):
            from fusion_mlx.engines.tts import TTSEngine
            self._tts_engine = TTSEngine(model_name)
            await self._tts_engine.start()
            logger.info("TTS engine started: %s", model_name)

    @property
    def is_tts_loaded(self) -> bool:
        return self._tts_engine is not None

    async def tts_synthesize(self, text: str, voice: str = None, ref_audio: str = None, speed: float = 1.0) -> bytes:
        async with NodeTimer.timed("TTS", "synthesize", text_len=len(text)):
            if not self._tts_engine:
                raise RuntimeError("TTS engine not loaded, call load_tts() first")
            result = await self._tts_engine.synthesize(
                text=text, voice=voice, ref_audio=ref_audio, speed=speed,
            )
            logger.info("TTS synthesize: text=%dchars voice=%s", len(text), voice)
            return result

    async def unload_tts(self):
        async with NodeTimer.timed("TTS", "unload_engine"):
            if self._tts_engine:
                await self._tts_engine.stop()
                self._tts_engine = None
            gc.collect()
            mx.metal.clear_cache()
            logger.info("TTS engine unloaded")

    # ── VLM lifecycle ────────────────────────────────────────

    async def load_vlm(self, model_name: str):
        async with NodeTimer.timed("VLM", "load_engine", model=model_name):
            from fusion_mlx.engines.vlm import VLMBatchedEngine
            self._vlm_engine = VLMBatchedEngine(model_name)
            await self._vlm_engine.start()
            logger.info("VLM engine started: %s", model_name)

    @property
    def is_vlm_loaded(self) -> bool:
        return self._vlm_engine is not None

    async def vlm_chat(self, messages: list, **kwargs) -> Any:
        async with NodeTimer.timed("VLM", "chat", msg_count=len(messages)):
            if not self._vlm_engine:
                raise RuntimeError("VLM engine not loaded, call load_vlm() first")
            result = await self._vlm_engine.chat(messages=messages, **kwargs)
            logger.info("VLM chat: msg_count=%d", len(messages))
            return result

    async def unload_vlm(self):
        async with NodeTimer.timed("VLM", "unload_engine"):
            if self._vlm_engine:
                await self._vlm_engine.stop()
                self._vlm_engine = None
            gc.collect()
            mx.metal.clear_cache()
            logger.info("VLM engine unloaded")

    # ── Full cleanup ─────────────────────────────────────────

    async def stop(self):
        if self._engine and self._started:
            await self._engine.stop()
            self._started = False
            logger.info("Engine stopped: %s", self.model_name)
        if self._pulid_pipeline:
            await self.unload_pulid()
        if self._lipsync_pipeline:
            await self.unload_lipsync()
        if self._musetalk_pipeline:
            await self.unload_musetalk()
        if self._tts_engine:
            await self.unload_tts()
        if self._vlm_engine:
            await self.unload_vlm()

    def get_memory_stats(self):
        return {
            "active_mb": round(mx.metal.get_active_memory() / 1024 / 1024, 1),
            "peak_mb": round(mx.metal.get_peak_memory() / 1024 / 1024, 1),
            "model_name": self.model_name,
            "model_type": self.model_type,
            "started": self._started,
            "pulid_loaded": self.is_pulid_loaded,
            "lipsync_loaded": self.is_lipsync_loaded,
            "musetalk_loaded": self.is_musetalk_loaded,
            "tts_loaded": self.is_tts_loaded,
            "vlm_loaded": self.is_vlm_loaded,
        }
