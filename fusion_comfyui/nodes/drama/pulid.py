import logging
import os

import mlx.core as mx
import numpy as np

from fusion_comfyui.nodes.base import BaseNode
from fusion_comfyui.core.timer import NodeTimer
from fusion_comfyui.core.lifecycle import FusionMemoryGuardian

logger = logging.getLogger("fusion_comfyui.nodes.drama.pulid")


class PuLIDIdentityExtract(BaseNode):
    RETURN_TYPES = ("ID_EMBEDDING",)
    CATEGORY = "fusion-mlx/drama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "character_name": ("STRING", {"default": "character_1"}),
                "model_dir": ("STRING", {"default": ""}),
                "cache_dir": ("STRING", {"default": "identity_cache"}),
            }
        }

    async def execute(self, image, character_name="character_1", model_dir="", cache_dir="identity_cache"):
        async with NodeTimer.timed("PuLIDIdentityExtract", "full"):
            cache_path = os.path.join(cache_dir, f"{character_name}.safetensors")
            if os.path.exists(cache_path):
                logger.info("PuLIDIdentityExtract: loading cached embedding for %s", character_name)
                id_embed = mx.load(cache_path)
                return ({"id_embed": id_embed, "character_name": character_name},)

            if isinstance(image, mx.array):
                arr = np.array(image)
            else:
                arr = np.array(image)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            rgb_uint8 = arr.astype(np.uint8)
            bgr_uint8 = rgb_uint8[:, :, ::-1].copy()

            from fusion_mlx.video.pulid_mlx import PuLIDPipeline
            async with NodeTimer.timed("PuLIDIdentityExtract", "load_pipeline"):
                pipeline = PuLIDPipeline.from_pretrained(model_dir)

            async with NodeTimer.timed("PuLIDIdentityExtract", "extract_embedding"):
                id_embed = pipeline.extract_id_embedding(bgr_uint8)

            logger.info(
                "PuLIDIdentityExtract: %s embed_shape=%s",
                character_name, tuple(id_embed.shape),
            )

            os.makedirs(cache_dir, exist_ok=True)
            mx.save_safetensors(cache_path, {"id_embed": id_embed})
            logger.info("PuLIDIdentityExtract: cached to %s", cache_path)

            del pipeline
            FusionMemoryGuardian.purge_memory()

            return ({"id_embed": id_embed, "character_name": character_name},)


class PuLIDConditioningApply(BaseNode):
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "fusion-mlx/drama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "id_embeddings": ("ID_EMBEDDING",),
                "model": ("MODEL",),
            }
        }

    async def execute(self, conditioning, id_embeddings, model):
        async with NodeTimer.timed("PuLIDConditioningApply", "full"):
            if isinstance(conditioning, dict):
                cond = dict(conditioning)
            else:
                cond = {"embeds": conditioning}

            if not model.is_pulid_loaded:
                logger.info("PuLIDConditioningApply: loading PuLID pipeline")
                await model.load_pulid("")

            async with NodeTimer.timed("PuLIDConditioningApply", "setup_attn"):
                dit = getattr(model._engine, "_dit", None) if model._engine else None
                if dit:
                    await model.pulid_setup_attn(dit)

            if isinstance(id_embeddings, list):
                for item in id_embeddings:
                    embed = item["id_embed"] if isinstance(item, dict) else item
                    await model.pulid_inject_id(embed)
            elif isinstance(id_embeddings, dict):
                embed = id_embeddings["id_embed"]
                await model.pulid_inject_id(embed)
            else:
                await model.pulid_inject_id(id_embeddings)

            cond["pulid_active"] = True
            logger.info("PuLIDConditioningApply: ID injected into conditioning")

            return (cond,)


class PuLIDClearID(BaseNode):
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "fusion-mlx/drama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    async def execute(self, model):
        async with NodeTimer.timed("PuLIDClearID", "full"):
            await model.pulid_clear_id()
            logger.info("PuLIDClearID: cleared PuLID ID from model")
            return (model,)
