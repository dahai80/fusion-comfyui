import io
import logging
import os

import cv2
import numpy as np
from PIL import Image

import core.async_utils

logger = logging.getLogger("fusion_comfyui.nodes.identity")

_PULID_MODEL_DIRS = []


def _get_pulid_models():
    if _PULID_MODEL_DIRS:
        return _PULID_MODEL_DIRS
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "pulid")
        if os.path.isdir(models_dir):
            for d in sorted(os.listdir(models_dir)):
                full = os.path.join(models_dir, d)
                if os.path.isdir(full):
                    _PULID_MODEL_DIRS.append(d)
    except Exception:
        pass
    fusion_dir = os.path.expanduser("~/.cache/fusion-mlx/pulid")
    if os.path.isdir(fusion_dir):
        for d in sorted(os.listdir(fusion_dir)):
            full = os.path.join(fusion_dir, d)
            if os.path.isdir(full) and d not in _PULID_MODEL_DIRS:
                _PULID_MODEL_DIRS.append(d)
    if not _PULID_MODEL_DIRS:
        _PULID_MODEL_DIRS.append("pulid_flux_v0.9.1")
    return _PULID_MODEL_DIRS


def _resolve_pulid_path(model_name: str) -> str:
    try:
        import folder_paths
        candidate = os.path.join(folder_paths.models_dir, "pulid", model_name)
        if os.path.isdir(candidate):
            return candidate
    except Exception:
        pass
    fusion_cache = os.path.expanduser(f"~/.cache/fusion-mlx/pulid/{model_name}")
    if os.path.isdir(fusion_cache):
        return fusion_cache
    return model_name


def _image_to_bgr(image_np: np.ndarray) -> np.ndarray:
    arr = image_np
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] > 4:
        arr = arr.transpose(1, 2, 0)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def _bytes_to_image_array(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    return arr


class FusionIdentityLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_pulid_models(),),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("FUSION_IDENTITY_MODEL",)
    RETURN_NAMES = ("identity_model",)
    FUNCTION = "load_identity"
    CATEGORY = "Fusion-MLX/Identity"

    def load_identity(self, model_name, dtype="float16"):
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()
        model_path = _resolve_pulid_path(model_name)
        logger.info("FusionIdentityLoader: model=%s path=%s dtype=%s", model_name, model_path, dtype)

        dtype_map = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}
        mx_dtype = dtype_map.get(dtype, "float16")

        try:
            pipeline = core.async_utils.run_async(
                self._load_pipeline(model_path, mx_dtype), timeout=120,
            )
        except Exception as e:
            logger.error("FusionIdentityLoader: failed to load PuLID: %s", e)
            raise

        FusionMemoryGuardian.maybe_purge()
        logger.info("FusionIdentityLoader: loaded PuLID pipeline from %s", model_name)
        return (pipeline,)

    async def _load_pipeline(self, model_path, dtype):
        import mlx.core as mx
        mx_dtype = getattr(mx, dtype, mx.float16)
        from fusion_mlx.video.pulid_mlx.pipeline import PuLIDPipeline
        pipeline = PuLIDPipeline.from_pretrained(model_path, dtype=mx_dtype)
        return pipeline


class FusionIdentityApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "identity_model": ("FUSION_IDENTITY_MODEL",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("FUSION_IDENTITY_EMBED",)
    RETURN_NAMES = ("identity_embed",)
    FUNCTION = "apply_identity"
    CATEGORY = "Fusion-MLX/Identity"

    def apply_identity(self, identity_model, image, weight=1.0, start_at=0.0, end_at=1.0):
        bgr = _image_to_bgr(image)
        logger.info("FusionIdentityApply: image_shape=%s weight=%.2f range=[%.2f,%.2f]", bgr.shape, weight, start_at, end_at)

        try:
            id_embedding = identity_model.extract_id_embedding(bgr)
        except Exception as e:
            logger.error("FusionIdentityApply: face extraction failed: %s", e)
            raise

        if id_embedding is None:
            logger.warning("FusionIdentityApply: no face detected")
            raise RuntimeError("No face detected in reference image")

        result = {
            "id_embedding": id_embedding,
            "weight": weight,
            "start_at": start_at,
            "end_at": end_at,
            "attn_processors": identity_model.attn_processors,
        }
        logger.info("FusionIdentityApply: embedding_shape=%s", id_embedding.shape)
        return (result,)


class FusionIdentityGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
                "identity_model": ("FUSION_IDENTITY_MODEL",),
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
    CATEGORY = "Fusion-MLX/Identity"

    def generate(self, pipeline, identity_model, reference_image, prompt,
                 negative_prompt="", width=1024, height=1024, steps=20,
                 cfg=6.0, identity_weight=1.0, seed=42):
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()
        bgr = _image_to_bgr(reference_image)
        logger.info(
            "FusionIdentityGenerate: prompt_len=%d ref_shape=%s size=%dx%d steps=%d cfg=%.1f weight=%.2f seed=%d",
            len(prompt), bgr.shape, width, height, steps, cfg, identity_weight, seed,
        )

        try:
            id_embedding = identity_model.extract_id_embedding(bgr)
            if id_embedding is None:
                raise RuntimeError("No face detected in reference image")
            logger.info("FusionIdentityGenerate: id_embedding shape=%s", id_embedding.shape)
            identity_model.inject_id(id_embedding)

            result_raw = core.async_utils.run_async(
                self._generate_with_identity(
                    pipeline, prompt, negative_prompt,
                    width, height, steps, cfg, seed,
                ),
                timeout=600,
            )

            identity_model.clear_id()

        except Exception as e:
            identity_model.clear_id()
            logger.error("FusionIdentityGenerate: failed: %s", e)
            raise

        raw_arr = result_raw[0]
        if isinstance(raw_arr, np.ndarray):
            image_np = raw_arr.astype(np.float32) / 255.0
        else:
            image_np = _bytes_to_image_array(raw_arr)
        FusionMemoryGuardian.maybe_purge()
        logger.info("FusionIdentityGenerate: output shape=%s", image_np.shape)
        return (image_np,)

    async def _generate_with_identity(self, pipeline, prompt, negative_prompt,
                                       width, height, steps, cfg, seed):
        await pipeline.ensure_started()
        neg = negative_prompt if negative_prompt else None
        result_raw = await pipeline._engine.generate(
            prompt=prompt, width=width, height=height,
            steps=steps, seed=seed, guidance=cfg, n_images=1,
            negative_prompt=neg, output_format="raw",
        )
        return result_raw
