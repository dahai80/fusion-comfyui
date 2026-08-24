import io
import logging
import os
import tempfile

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image

import core.async_utils

logger = logging.getLogger("fusion_comfyui.nodes.shortcuts")

_pulid_cache = {}


def _raw_shape(raw) -> tuple:
    if isinstance(raw, np.ndarray):
        return tuple(raw.shape)
    try:
        img = Image.open(io.BytesIO(raw))
        return img.size[::-1] + (len(img.getbands()),)
    except Exception:
        return ()


def _raw_width_collapsed(raw, width: int, height: int) -> bool:
    # Upstream raw output collapses width to a tiny value (e.g. 3px) while
    # height stays correct. Flag any spatial dim well below the request.
    threshold_w = min(width * 0.5, 64)
    threshold_h = min(height * 0.5, 64)
    try:
        if isinstance(raw, np.ndarray):
            if raw.ndim == 3:
                h, w, _c = raw.shape
            elif raw.ndim == 2:
                h, w = raw.shape
            else:
                return False
            return w < threshold_w or h < threshold_h
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        return w < threshold_w or h < threshold_h
    except Exception:
        return False


def _bytes_to_image_array(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    return arr


def _video_bytes_to_frame_array(video_bytes: bytes) -> np.ndarray:
    import av
    container = av.open(io.BytesIO(video_bytes))
    try:
        frames = []
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            frames.append(arr)
    finally:
        container.close()

    if not frames:
        raise RuntimeError("No frames decoded from video output")
    return np.stack(frames, axis=0)


def _video_result_to_frames(video_result, label: str) -> np.ndarray:
    if isinstance(video_result[0], np.ndarray):
        raw = video_result[0]
        frames_np = raw.astype(np.float32) / 255.0
        if frames_np.ndim == 3 and frames_np.shape[2] == 3:
            frames_np = frames_np[np.newaxis, ...]
        logger.info("%s: raw ndarray path, shape=%s", label, frames_np.shape)
    elif isinstance(video_result[0], (bytes, bytearray)):
        frames_np = _video_bytes_to_frame_array(video_result[0])
    elif isinstance(video_result, (bytes, bytearray)):
        frames_np = _video_bytes_to_frame_array(video_result)
    else:
        logger.warning("%s: unexpected result type=%s", label, type(video_result))
        frames_np = _video_bytes_to_frame_array(
            bytes(video_result[0]) if video_result else b""
        )
    return frames_np


class FusionImageGenNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/Shortcuts"

    def generate(self, pipeline, prompt, negative_prompt="", width=1024, height=1024,
                 steps=20, cfg=6.0, seed=42):
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()
        logger.info(
            "FusionImageGen: prompt_len=%d size=%dx%d steps=%d cfg=%.1f seed=%d",
            len(prompt), width, height, steps, cfg, seed,
        )

        # Upstream engine occasionally returns a width-collapsed raw array
        # (e.g. shape (H,3,3)) for image gen. Detect and retry with a fresh
        # seed instead of emitting a degenerate black image. See issue #37.
        max_retries = 3
        cur_seed = seed
        result_raw = None
        for attempt in range(max_retries + 1):
            try:
                result_raw = core.async_utils.run_async(
                    self._generate_image(pipeline, prompt, negative_prompt,
                                         width, height, steps, cfg, cur_seed),
                    timeout=600,
                )
            except Exception as e:
                logger.error("FusionImageGen: failed (attempt %d): %s", attempt + 1, e)
                raise
            if attempt < max_retries and _raw_width_collapsed(result_raw[0], width, height):
                logger.warning(
                    "FusionImageGen: width collapse detected attempt=%d shape=%s requested=%dx%d; retrying with seed %d",
                    attempt + 1, _raw_shape(result_raw[0]), width, height, cur_seed + 1,
                )
                cur_seed += 1
                continue
            break

        raw_arr = result_raw[0]
        if isinstance(raw_arr, np.ndarray):
            image_np = raw_arr.astype(np.float32) / 255.0
            if image_np.ndim == 3:
                image_np = image_np[np.newaxis, ...]
        else:
            image_np = _bytes_to_image_array(raw_arr)
        logger.info("FusionImageGen: output shape=%s", image_np.shape)
        return (image_np,)

    async def _generate_image(self, pipeline, prompt, negative_prompt,
                               width, height, steps, cfg, seed):
        await pipeline.ensure_started()
        neg = negative_prompt if negative_prompt else None
        result = await pipeline._engine.generate(
            prompt=prompt, width=width, height=height,
            steps=steps, seed=seed, guidance=cfg, n_images=1,
            negative_prompt=neg, output_format="raw",
        )
        return result


class FusionVideoGenNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 768, "min": 256, "max": 1280, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 720, "step": 64}),
                "num_frames": ("INT", {"default": 41, "min": 1, "max": 257}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/Shortcuts"

    def generate(self, pipeline, prompt, negative_prompt="", width=768, height=512,
                 num_frames=41, fps=24, steps=30, cfg=5.0, seed=42):
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()
        logger.info(
            "FusionVideoGen: prompt_len=%d size=%dx%d frames=%d steps=%d cfg=%.1f seed=%d",
            len(prompt), width, height, num_frames, steps, cfg, seed,
        )

        try:
            video_result = core.async_utils.run_async(
                self._generate_video(pipeline, prompt, negative_prompt,
                                     width, height, num_frames, fps, steps, cfg, seed),
                timeout=3600,
            )
        except Exception as e:
            logger.error("FusionVideoGen: failed: %s", e)
            raise

        frames_np = _video_result_to_frames(video_result, "FusionVideoGen")
        logger.info("FusionVideoGen: output shape=%s", frames_np.shape)
        return (frames_np,)

    async def _generate_video(self, pipeline, prompt, negative_prompt,
                               width, height, num_frames, fps, steps, cfg, seed):
        await pipeline.ensure_started()
        neg = negative_prompt if negative_prompt else None
        try:
            result_raw = await pipeline._engine.generate(
                prompt=prompt, num_frames=num_frames, width=width, height=height,
                fps=fps, seed=seed, n=1, num_inference_steps=steps,
                cfg_scale=cfg, negative_prompt=neg, output_format="raw",
            )
            if isinstance(result_raw[0], np.ndarray):
                logger.info("_generate_video: raw frames shape=%s", result_raw[0].shape)
                return result_raw
            logger.info(
                "_generate_video: raw not returned (got %s), using as mp4",
                type(result_raw[0]).__name__,
            )
            return result_raw
        except (TypeError, AttributeError) as e:
            logger.info("_generate_video: raw not supported, falling back to mp4: %s", e)
        result_bytes = await pipeline._engine.generate(
            prompt=prompt, num_frames=num_frames, width=width, height=height,
            fps=fps, seed=seed, n=1, num_inference_steps=steps,
            cfg_scale=cfg, negative_prompt=neg,
        )
        return result_bytes


class FusionImageToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 768, "min": 256, "max": 1280, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 720, "step": 64}),
                "num_frames": ("INT", {"default": 41, "min": 1, "max": 257}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/Shortcuts"

    def generate(self, pipeline, image, prompt, negative_prompt="", width=768,
                 height=512, num_frames=41, fps=24, steps=30, cfg=5.0, seed=42):
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()
        logger.info(
            "FusionI2V: prompt_len=%d size=%dx%d frames=%d steps=%d cfg=%.1f seed=%d",
            len(prompt), width, height, num_frames, steps, cfg, seed,
        )

        control_image = self._prepare_control_image(image)

        try:
            video_result = core.async_utils.run_async(
                self._generate_i2v(pipeline, control_image, prompt,
                                   negative_prompt, width, height,
                                   num_frames, fps, steps, cfg, seed),
                timeout=3600,
            )
        except Exception as e:
            logger.error("FusionI2V: failed: %s", e)
            raise

        frames_np = _video_result_to_frames(video_result, "FusionI2V")
        logger.info("FusionI2V: output shape=%s", frames_np.shape)
        return (frames_np,)

    def _prepare_control_image(self, image):
        from core.bridge import to_numpy

        arr = to_numpy(image)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            pass
        elif arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = arr.transpose(1, 2, 0)

        return Image.fromarray(arr[:, :, :3])

    async def _generate_i2v(self, pipeline, control_image, prompt,
                             negative_prompt, width, height, num_frames, fps,
                             steps, cfg, seed):
        await pipeline.ensure_started()
        neg = negative_prompt if negative_prompt else None
        image_input = control_image
        tmp_path = None
        if isinstance(control_image, Image.Image):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                control_image.save(tmp, format="PNG")
                tmp.close()
                tmp_path = tmp.name
                image_input = tmp_path
                logger.info("_generate_i2v: saved PIL Image to temp file %s", tmp_path)
            except Exception:
                tmp.close()
                os.unlink(tmp.name)
                raise
        try:
            result_raw = await pipeline._engine.generate(
                prompt=prompt, num_frames=num_frames, width=width, height=height,
                fps=fps, seed=seed, n=1, num_inference_steps=steps,
                cfg_scale=cfg, negative_prompt=neg,
                image=image_input, output_format="raw",
            )
            if isinstance(result_raw[0], np.ndarray):
                logger.info("_generate_i2v: raw frames shape=%s", result_raw[0].shape)
                return result_raw
            logger.info(
                "_generate_i2v: raw not returned (got %s), using as mp4",
                type(result_raw[0]).__name__,
            )
            return result_raw
        except (TypeError, AttributeError) as e:
            logger.info("_generate_i2v: raw not supported, falling back to mp4: %s", e)
        try:
            result_bytes = await pipeline._engine.generate(
                prompt=prompt, num_frames=num_frames, width=width, height=height,
                fps=fps, seed=seed, n=1, num_inference_steps=steps,
                cfg_scale=cfg, negative_prompt=neg,
                image=image_input,
            )
            return result_bytes
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _image_to_bgr(image_np: np.ndarray) -> np.ndarray:
    arr = image_np.copy()
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


class FusionIdentityPipelineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "identity_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/Shortcuts"

    def generate(self, pipeline, reference_image, prompt, negative_prompt="",
                 width=1024, height=1024, steps=20, cfg=6.0,
                 identity_weight=1.0, seed=42):
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()
        bgr = _image_to_bgr(reference_image)
        logger.info(
            "FusionIdentityPipeline: prompt_len=%d ref=%s size=%dx%d steps=%d "
            "cfg=%.1f weight=%.2f seed=%d",
            len(prompt), bgr.shape, width, height, steps, cfg, identity_weight, seed,
        )

        try:
            result_raw = core.async_utils.run_async(
                self._generate(pipeline, bgr, prompt, negative_prompt,
                               width, height, steps, cfg, identity_weight, seed),
                timeout=600,
            )
        except Exception as e:
            logger.error("FusionIdentityPipeline: failed: %s", e)
            raise

        raw_arr = result_raw[0]
        if isinstance(raw_arr, np.ndarray):
            image_np = raw_arr.astype(np.float32) / 255.0
        else:
            image_np = _bytes_to_image_array(raw_arr)
        logger.info("FusionIdentityPipeline: output shape=%s", image_np.shape)
        return (image_np,)

    async def _generate(self, pipeline, bgr_image, prompt, negative_prompt,
                         width, height, steps, cfg, identity_weight, seed):
        from fusion_mlx.public_api import PuLIDPipeline
        cache_key = "pulid_flux_v0.9.1"
        pulid = _pulid_cache.get(cache_key)
        if pulid is None:
            logger.info("FusionIdentityPipeline: loading PuLIDPipeline (first call, will be cached)")
            pulid = PuLIDPipeline.from_pretrained(
                os.path.expanduser("~/.cache/fusion-mlx/pulid/pulid_flux_v0.9.1"),
                dtype=mx.float16,
            )
            _pulid_cache[cache_key] = pulid
        else:
            logger.info("FusionIdentityPipeline: reusing cached PuLIDPipeline")
        try:
            id_embedding = pulid.extract_id_embedding(bgr_image)
            if id_embedding is None:
                raise RuntimeError("No face detected in reference image")
            pulid.setup_attn_processors(pipeline._engine._model)
            pulid.inject_id(id_embedding)

            await pipeline.ensure_started()
            neg = negative_prompt if negative_prompt else None
            result_raw = await pipeline._engine.generate(
                prompt=prompt, width=width, height=height,
                steps=steps, seed=seed, guidance=cfg, n_images=1,
                negative_prompt=neg, output_format="raw",
            )
        finally:
            pulid.clear_id()

        return result_raw
