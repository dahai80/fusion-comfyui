import io
import logging
from typing import Callable, Awaitable

import mlx.core as mx

logger = logging.getLogger("fusion_comfyui.engine_wrapper")

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


def _infer_model_type(model_name: str) -> str:
    name = model_name.lower()
    for key, mtype in _MODEL_TYPES.items():
        if key in name:
            return mtype
    return "image"


class FusionEngineWrapper:
    def __init__(self, model_name: str, offload_strategy: str = "sequential", quant_bit: str = "fp8_e4m3"):
        self.model_name = model_name
        self.offload_strategy = offload_strategy
        self.quant_bit = quant_bit
        self.model_type = _infer_model_type(model_name)
        self._engine = None
        self._started = False
        self._on_step: StepCallback | None = None
        logger.info(
            "FusionEngineWrapper: model=%s type=%s offload=%s quant=%s",
            model_name, self.model_type, offload_strategy, quant_bit,
        )

    def set_progress_callback(self, cb: StepCallback | None):
        self._on_step = cb

    async def ensure_started(self):
        if self._started:
            return
        logger.info("Starting fusion-mlx engine: %s", self.model_name)
        if self.model_type == "image":
            from fusion_mlx.public_api import ImageGenEngine
            quantize = None
            if self.quant_bit in ("4bit", "w4", "nf4"):
                quantize = 4
            elif self.quant_bit in ("8bit", "w8a16", "fp8_e4m3"):
                quantize = 8
            self._engine = ImageGenEngine(self.model_name, quantize=quantize)
        elif self.model_type == "video":
            from fusion_mlx.public_api import VideoGenEngine
            self._engine = VideoGenEngine(self.model_name)
        await self._engine.start()
        self._started = True
        logger.info("Engine started: %s", self.model_name)

    async def load_text_encoder(self):
        await self.ensure_started()
        await self._engine.load_text_encoder()
        logger.info("stage loaded: text_encoder for %s", self.model_name)

    async def encode_text(self, prompt: str, negative_prompt: str = "") -> dict:
        await self.ensure_started()
        result = await self._engine.encode_text(prompt)
        logger.info("encode_text: prompt_len=%d embed_shape=%s", len(prompt), tuple(result["embed"].shape))
        result["negative_prompt"] = negative_prompt
        return result

    async def unload_text_encoder(self):
        if self._engine and self._started:
            await self._engine.unload_text_encoder()
            logger.info("stage unloaded: text_encoder for %s", self.model_name)

    async def load_dit(self):
        await self.ensure_started()
        await self._engine.load_dit()
        logger.info("stage loaded: dit for %s", self.model_name)

    async def denoise(self, latent, positive, negative, steps=20, cfg=6.0, seed=0, **kwargs):
        await self.ensure_started()
        pos_embed = positive.get("embed")
        neg_embed = negative.get("embed") if negative else None
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
        mx.eval(result)
        logger.info("denoise: steps=%d cfg=%.1f seed=%d out_shape=%s", steps, cfg, seed, tuple(result.shape))
        return result

    async def unload_dit(self):
        if self._engine and self._started:
            await self._engine.unload_dit()
            logger.info("stage unloaded: dit for %s", self.model_name)

    async def load_vae(self):
        await self.ensure_started()
        await self._engine.load_vae()
        logger.info("stage loaded: vae for %s", self.model_name)

    async def decode(self, latent, tile_size=256):
        await self.ensure_started()
        result = await self._engine.decode(latent)
        mx.eval(result)
        logger.info("decode: out_shape=%s", tuple(result.shape))
        return result

    async def decode_tiled(self, latent, tile_size=256):
        await self.ensure_started()
        if hasattr(self._engine, "decode_tiled"):
            result = await self._engine.decode_tiled(latent, tile_size=tile_size)
        else:
            result = await self._engine.decode(latent)
        mx.eval(result)
        logger.info("decode_tiled: tile_size=%d out_shape=%s", tile_size, tuple(result.shape))
        return result

    async def unload_vae(self):
        if self._engine and self._started:
            await self._engine.unload_vae()
            logger.info("stage unloaded: vae for %s", self.model_name)

    async def _fallback_generate(self, latent, positive, negative, steps, cfg, seed, **kwargs):
        import numpy as np

        logger.info("fallback: using monolithic generate() for %s", self.model_name)
        prompt = positive.get("prompt", "")
        neg_prompt = negative.get("negative_prompt", "")
        if self.model_type == "image":
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 1024)
            result_raw = await self._engine.generate(
                prompt=prompt, width=width, height=height,
                steps=steps, seed=seed, guidance=cfg, n_images=1,
                on_step=self._on_step, output_format="raw",
            )
            raw_arr = result_raw[0]
            if isinstance(raw_arr, np.ndarray):
                arr = raw_arr.astype(np.float32) / 255.0
            else:
                from PIL import Image
                img = Image.open(io.BytesIO(raw_arr))
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
            try:
                result_raw = await self._engine.generate(
                    prompt=prompt, num_frames=num_frames, width=width, height=height,
                    seed=seed, n=1, num_inference_steps=steps, cfg_scale=cfg,
                    negative_prompt=neg_prompt or None, on_step=self._on_step,
                    output_format="raw",
                )
                if isinstance(result_raw[0], np.ndarray):
                    logger.info("_fallback_generate: raw video frames shape=%s", result_raw[0].shape)
                    video = result_raw[0].astype(np.float32) / 255.0
                    if video.ndim == 4 and video.shape[3] == 3:
                        video = video.transpose(3, 0, 1, 2)[np.newaxis, ...]
                    return mx.array(video)
            except (TypeError, AttributeError) as e:
                logger.info("_fallback_generate: raw output not supported, falling back to mp4: %s", e)
            result_bytes = await self._engine.generate(
                prompt=prompt, num_frames=num_frames, width=width, height=height,
                seed=seed, n=1, num_inference_steps=steps, cfg_scale=cfg,
                negative_prompt=neg_prompt or None, on_step=self._on_step,
            )
            import av
            container = av.open(io.BytesIO(result_bytes[0]))
            try:
                frames = []
                for frame in container.decode(video=0):
                    arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                    frames.append(arr)
            finally:
                container.close()
            if not frames:
                return latent
            return mx.array(np.stack(frames, axis=0))

    def load_stage(self, stage_name: str):
        logger.info("load_stage: %s (use async load_* methods)", stage_name)
        return self

    def unload_stage(self, stage_name: str):
        from core.lifecycle import FusionMemoryGuardian
        logger.info("unload_stage: %s", stage_name)
        FusionMemoryGuardian.maybe_purge()

    def stage(self, stage_name: str):
        from core.lifecycle import PipelineStageContext
        return PipelineStageContext(self, stage_name)

    async def stop(self):
        if self._engine and self._started:
            await self._engine.stop()
            self._started = False
            self._engine = None
            logger.info("Engine stopped: %s", self.model_name)

    def get_memory_stats(self) -> dict:
        active = mx.metal.get_active_memory() / 1024 / 1024
        peak = mx.metal.get_peak_memory() / 1024 / 1024
        return {
            "active_mb": round(active, 1),
            "peak_mb": round(peak, 1),
            "model_name": self.model_name,
            "model_type": self.model_type,
            "started": self._started,
        }
