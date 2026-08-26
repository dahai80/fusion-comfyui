import logging

import fusion_comfyui.core.async_utils

logger = logging.getLogger("fusion_comfyui.nodes.conditioning")


class CLIPTextEncode:
    """Override native CLIPTextEncode — encodes text via fusion-mlx.
    Accepts FusionCLIPWrapper, stores prompt text for monolithic generate().
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP",),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "model/conditioning"

    def encode(self, clip, text):
        from fusion_comfyui.core.wrappers import FusionCLIPWrapper

        if isinstance(clip, FusionCLIPWrapper):
            result = {
                "prompt": text,
                "clip": clip,
                "embed": None,
                "model_name": clip.model_name,
            }
            logger.info("CLIPTextEncode override: prompt_len=%d model=%s", len(text), clip.model_name)
            return (result,)

        result = {"prompt": text, "clip": clip, "embed": None}
        logger.warning("CLIPTextEncode: non-fusion CLIP received, storing prompt text only")
        return (result,)


class FusionTextEncoderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("FUSION_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Fusion-MLX/Conditioning"

    def encode(self, pipeline, prompt, negative_prompt=""):
        from fusion_comfyui.core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()
        logger.info("FusionTextEncoder: encoding prompt_len=%d", len(prompt))

        try:
            result = fusion_comfyui.core.async_utils.run_async(
                self._encode_staged(pipeline, prompt, negative_prompt),
                timeout=300,
            )
        except Exception:
            result = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "embed": None,
            }
            logger.warning("FusionTextEncoder: async encode failed, storing prompt text only")

        return (result,)

    async def _encode_staged(self, pipeline, prompt, negative_prompt):
        await pipeline.load_text_encoder()
        try:
            result = await pipeline.encode_text(prompt, negative_prompt)
        finally:
            await pipeline.unload_text_encoder()
        return result
