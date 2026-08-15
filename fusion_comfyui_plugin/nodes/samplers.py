import io
import logging

import mlx.core as mx
import numpy as np

import core.async_utils
from ._sampler_constants import SAMPLER_NAMES, SCHEDULER_NAMES

logger = logging.getLogger("fusion_comfyui.nodes.samplers")

_decoded_frames_cache = {}


async def _generate_monolithic(model_wrapper, positive, negative, latent_image,
                               steps, cfg, seed, width, height, num_frames):
    engine = model_wrapper.get_engine()
    await engine.ensure_started()

    prompt = positive.get("prompt", "")
    neg_prompt = negative.get("prompt", "") if negative else ""
    i2v_image = latent_image.get("_i2v_image_path")
    i2v_strength = latent_image.get("_i2v_image_strength", 1.0)

    if model_wrapper.model_type == "video":
        gen_kwargs = {
            "prompt": prompt,
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "seed": seed,
            "n": 1,
            "num_inference_steps": steps,
            "cfg_scale": cfg,
            "negative_prompt": neg_prompt or None,
            "output_format": "raw",
        }
        if i2v_image:
            gen_kwargs["image"] = i2v_image
            gen_kwargs["image_strength"] = i2v_strength
            logger.info("_generate_monolithic: I2V with image=%s strength=%.2f", i2v_image, i2v_strength)

        vace_ctrl = latent_image.get("_vace_control_video")
        vace_mask = latent_image.get("_vace_control_mask")
        vace_ref = latent_image.get("_vace_reference_images")
        if vace_ctrl:
            gen_kwargs["control_video"] = vace_ctrl
            logger.info("_generate_monolithic: VACE control_video=%s", vace_ctrl)
        if vace_mask:
            gen_kwargs["control_mask"] = vace_mask
            logger.info("_generate_monolithic: VACE control_mask=%s", vace_mask)
        if vace_ref:
            gen_kwargs["reference_images"] = vace_ref
            logger.info("_generate_monolithic: VACE reference_images=%s", vace_ref)

        try:
            result_raw = await engine._engine.generate(**gen_kwargs)
            if isinstance(result_raw[0], np.ndarray):
                logger.info("_generate_monolithic: raw video frames shape=%s", result_raw[0].shape)
                video = result_raw[0].astype(np.float32) / 255.0
                if video.ndim == 4 and video.shape[3] == 3:
                    return video
                return np.stack([video], axis=0)
        except (TypeError, AttributeError) as e:
            logger.info("_generate_monolithic: raw output not supported, falling back to mp4: %s", e)
            gen_kwargs.pop("output_format", None)

        result_bytes = await engine._engine.generate(**gen_kwargs)
        logger.info("_generate_monolithic: got %d result bytes arrays, sizes=%s",
                     len(result_bytes), [len(b) for b in result_bytes])
        try:
            import av
            container = av.open(io.BytesIO(result_bytes[0]))
            try:
                frames = []
                for frame in container.decode(video=0):
                    arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                    frames.append(arr)
            finally:
                container.close()
            logger.info("_generate_monolithic: decoded %d frames from mp4 (av in-memory)", len(frames))
        except ImportError:
            logger.warning("av not available, returning empty latent")
            from core.bridge import to_mlx_array
            return to_mlx_array(latent_image["samples"])
        except Exception as e:
            logger.error("_generate_monolithic: failed to decode mp4: %s", e)
            from core.bridge import to_mlx_array
            return to_mlx_array(latent_image["samples"])

        if not frames:
            from core.bridge import to_mlx_array
            return to_mlx_array(latent_image["samples"])

        return np.stack(frames, axis=0)

    elif model_wrapper.model_type == "image":
        # Stable Cascade two-KSampler workflow: the prior KSampler (stage_c)
        # already ran the self-contained fusion-mlx pipeline (prior+decoder+
        # vqgan) and produced the final decoded RGB image. The decoder
        # KSampler (stage_b) only needs that image — re-running the whole
        # pipeline here with cfg=0.0 disables prior CFG and corrupts the
        # output (bimodal 0/255). Short-circuit: return the attached prior.
        prior_img = None
        for cond in (positive, negative):
            if isinstance(cond, dict):
                cand = cond.get("stable_cascade_prior")
                if cand is not None:
                    prior_img = cand
                    break
        if prior_img is not None:
            arr = np.asarray(prior_img)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
            # prior KSampler stored samples as 5D (1,1,H,W,3); collapse
            # leading singleton dims to the 3D (H,W,3) the caller expects.
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            logger.info(
                "_generate_monolithic image: cascade decoder stage, "
                "passing through prior KSampler output shape=%s",
                arr.shape,
            )
            return arr

        result_raw = await engine._engine.generate(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            guidance=cfg,
            n_images=1,
            output_format="raw",
        )
        logger.info("_generate_monolithic image: got %d raw arrays, shapes=%s",
                     len(result_raw), [getattr(r, "shape", "?") for r in result_raw])
        raw_arr = result_raw[0]
        if isinstance(raw_arr, np.ndarray):
            arr = raw_arr.astype(np.float32) / 255.0
        else:
            from PIL import Image
            try:
                img = Image.open(io.BytesIO(raw_arr))
            except Exception as pil_err:
                logger.error("_generate_monolithic image: PIL failed (%s), trying fallback", pil_err)
                result = await engine._fallback_generate(
                    latent_image["samples"], positive, negative, steps, cfg, seed,
                    width=width, height=height,
                )
                return result
            arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    return latent_image["samples"]


class KSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (list(SAMPLER_NAMES),),
                "scheduler": (list(SCHEDULER_NAMES),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "model/sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise=1.0):
        from core.wrappers import FusionModelWrapper
        from core.lifecycle import FusionMemoryGuardian

        seed = seed & 0xFFFFFFFF
        FusionMemoryGuardian.maybe_purge()

        if not isinstance(model, FusionModelWrapper):
            logger.error("KSampler: received non-fusion model %s", type(model))
            raise RuntimeError("KSampler override requires FusionModelWrapper")

        raw_samples = latent_image["samples"]
        if isinstance(raw_samples, mx.array):
            shape = tuple(raw_samples.shape)
        elif isinstance(raw_samples, np.ndarray):
            shape = raw_samples.shape
        else:
            shape = (1, 16, 1, 64, 64)

        # Prefer explicit num_frames/width/height from latent dict
        # (latent_t != num_frames; e.g. Wan22: length=41 -> latent_t=11, but engine needs num_frames=41)
        num_frames = latent_image.get("num_frames", None)
        height = latent_image.get("height", None)
        width = latent_image.get("width", None)

        if num_frames is None or height is None or width is None:
            if len(shape) == 5:
                _, c, t, h, w = shape
                num_frames = num_frames or max(t, 1)
                height = height or h * 8
                width = width or w * 8
            elif len(shape) == 4:
                _, c, h, w = shape
                num_frames = num_frames or 1
                height = height or h * 8
                width = width or w * 8
            else:
                num_frames = num_frames or 97
                height = height or 512
                width = width or 768

        logger.info(
            "KSampler override: model=%s steps=%d cfg=%.1f seed=%d frames=%d %dx%d",
            model.model_name, steps, cfg, seed, num_frames, width, height,
        )

        result = core.async_utils.run_async(
            _generate_monolithic(
                model, positive, negative, latent_image,
                steps, cfg, seed, width, height, num_frames,
            ),
            timeout=3600,
        )

        _i2v_tmp = latent_image.get("_i2v_image_path")
        # Don't delete the temp file here — multi-KSampler workflows (e.g. wan22 14B i2v)
        # reuse the same latent dict across stages.  Deleting it would break subsequent
        # KSampler nodes that also need the i2v image.  OS temp cleanup handles this.

        if isinstance(result, np.ndarray) and result.ndim >= 3:
            extra = {k: v for k, v in latent_image.items() if k != "samples"}
            # Store as numpy to avoid Metal allocation in format_value
            if result.ndim == 5:
                samples_np = result
            elif result.ndim == 4:
                samples_np = result[np.newaxis, ...]
            else:
                samples_np = result[np.newaxis, np.newaxis, ...]
            cache_key = hash(samples_np.tobytes()) & 0xFFFFFFFF
            _decoded_frames_cache[cache_key] = result
            mx.metal.clear_cache()
            output = {"samples": samples_np, **extra}
            output["_decoded_frames_key"] = cache_key
            logger.info("KSampler: monolithic generate done, frames shape=%s", result.shape)
            return (output,)

        if isinstance(result, mx.array):
            mx.eval(result)
            extra = {k: v for k, v in latent_image.items() if k != "samples"}
            result_np = np.array(result)
            mx.metal.clear_cache()
            output = {"samples": result_np, **extra}
            logger.info("KSampler: stage denoise done, latent shape=%s", tuple(result.shape))
            return (output,)

        extra = {k: v for k, v in latent_image.items() if k != "samples"}
        output = {"samples": latent_image["samples"], **extra}
        logger.warning("KSampler: unexpected result type %s", type(result))
        return (output,)


class KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"], {"advanced": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (list(SAMPLER_NAMES),),
                "scheduler": (list(SCHEDULER_NAMES),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "advanced": True}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "advanced": True}),
                "return_with_leftover_noise": (["disable", "enable"], {"advanced": True}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "model/sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, start_at_step=0, end_at_step=10000,
               return_with_leftover_noise="disable"):
        ksampler = KSampler()
        return ksampler.sample(
            model, noise_seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, denoise=1.0,
        )


class SamplerCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            }
        }
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "model/sampling"

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative,
               sampler, sigmas, latent_image):
        ksampler = KSampler()
        steps = len(sigmas) - 1 if hasattr(sigmas, '__len__') else 20
        result = ksampler.sample(
            model, noise_seed, steps, cfg, "euler", "normal",
            positive, negative, latent_image, denoise=1.0,
        )
        return (result[0], result[0])


class SamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            }
        }
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "model/sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        from core.wrappers import FusionModelWrapper
        model = guider.get("model") if isinstance(guider, dict) else getattr(guider, "model", None)
        conditioning = guider.get("conditioning") if isinstance(guider, dict) else getattr(guider, "conditioning", None)
        noise_seed = noise.get("noise_seed", 0) if isinstance(noise, dict) else 0

        if not isinstance(model, FusionModelWrapper):
            logger.error("SamplerCustomAdvanced: no FusionModelWrapper in guider, got %s", type(model))
            extra = {k: v for k, v in latent_image.items() if k != "samples"}
            return ({"samples": latent_image["samples"], **extra},) * 2

        positive = conditioning if isinstance(conditioning, dict) else {"prompt": ""}
        negative = {"prompt": ""}
        if isinstance(positive, dict) and "prompt" not in positive:
            positive = {"prompt": ""}

        cfg = positive.get("guidance", 6.0) if isinstance(positive, dict) else 6.0
        steps = len(sigmas) - 1 if hasattr(sigmas, "__len__") else 20

        ksampler = KSampler()
        output = ksampler.sample(
            model, noise_seed, steps, cfg, "euler", "normal",
            positive, negative, latent_image, denoise=1.0,
        )
        logger.info("SamplerCustomAdvanced: done, returning (output, denoised_output) 2-tuple")
        return (output[0], output[0])


class FusionKSamplerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
                "positive": ("FUSION_COND",),
                "negative": ("FUSION_COND",),
                "latent_image": ("LATENT",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 257}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Fusion-MLX/Sampling"

    def sample(self, pipeline, positive, negative, latent_image, steps, cfg, seed,
               width=1024, height=1024, num_frames=1):
        from core.bridge import to_mlx_array
        from core.lifecycle import FusionMemoryGuardian

        seed = seed & 0xFFFFFFFF
        FusionMemoryGuardian.maybe_purge()
        logger.info(
            "FusionKSampler: steps=%d cfg=%.1f seed=%d size=%dx%d frames=%d",
            steps, cfg, seed, width, height, num_frames,
        )

        raw_samples = latent_image["samples"]
        mlx_latent = to_mlx_array(raw_samples)

        try:
            output_mlx = core.async_utils.run_async(
                self._sample_staged(
                    pipeline, mlx_latent, positive, negative,
                    steps, cfg, seed, width, height, num_frames,
                ),
                timeout=600,
            )
        except Exception as e:
            logger.error("FusionKSampler: denoise failed: %s", e)
            raise

        mx.eval(output_mlx)

        extra = {k: v for k, v in latent_image.items() if k != "samples"}
        output_latent = {"samples": output_mlx, **extra}

        mem_stats = pipeline.get_memory_stats()
        logger.info(
            "FusionKSampler: done. memory: active=%.0fMB peak=%.0fMB out_shape=%s",
            mem_stats["active_mb"],
            mem_stats["peak_mb"],
            tuple(output_mlx.shape),
        )

        return (output_latent,)

    async def _sample_staged(self, pipeline, mlx_latent, positive, negative,
                             steps, cfg, seed, width, height, num_frames):
        await pipeline.load_dit()
        try:
            result = await pipeline.denoise(
                mlx_latent, positive, negative,
                steps=steps, cfg=cfg, seed=seed,
                width=width, height=height, num_frames=num_frames,
            )
        finally:
            await pipeline.unload_dit()
        return result
