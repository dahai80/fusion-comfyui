import logging

logger = logging.getLogger("fusion_comfyui.nodes.loaders")


_KNOWN_DIFFUSION_MODELS = [
    "wan2.2_ti2v_5B_fp16.safetensors",
    "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    "wan2.1_t2v_1.3B_fp16.safetensors",
    "wan2.1_i2v_480p_14B_fp16.safetensors",
    "wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors",
    "wan2.1_vace_14B_fp16.safetensors",
    "Cosmos-1_0-Diffusion-7B-Text2World.safetensors",
    "Cosmos-1_0-Diffusion-7B-Video2World.safetensors",
    "cosmos_predict2_2B_video2world_480p_16fps.safetensors",
    "hunyuan_video_t2v_720p_bf16.safetensors",
    "hunyuan_video_image_to_video_720p_bf16.safetensors",
    "hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors",
    "skyreels_v3_a2v_19b_mlx.safetensors",
    "skyreels_v3_r2v_14b_mlx.safetensors",
    "skyreels_v3_v2v_14b_mlx.safetensors",
]

_KNOWN_TEXT_ENCODERS = [
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "t5xxl_fp16.safetensors",
    "oldt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "clip_l.safetensors",
    "llava_llama3_fp8_scaled.safetensors",
    "t5_encoder.safetensors",
]

_KNOWN_VAE_MODELS = [
    "wan2.2_vae.safetensors",
    "wan_2.1_vae.safetensors",
    "cosmos_cv8x8x8_1.0.safetensors",
    "hunyuan_video_vae_bf16.safetensors",
    "vae.safetensors",
]

_KNOWN_CLIP_VISION = [
    "clip_vision_h.safetensors",
    "llava_llama3_vision.safetensors",
]

_KNOWN_CHECKPOINTS = [
    "ltx-video-2b-v0.9.5.safetensors",
    "ltx-video-2b-v0.9.safetensors",
    "ltx-video-2.3-mlx-q8",
    "svd.safetensors",
    "svd_xt.safetensors",
    "sd_xl_1.0.safetensors",
]


def _get_diffusion_models():
    try:
        import folder_paths
        files = folder_paths.get_filename_list("diffusion_models")
        if files:
            return sorted(set(files + _KNOWN_DIFFUSION_MODELS))
    except Exception:
        pass
    return _KNOWN_DIFFUSION_MODELS


def _get_text_encoders():
    try:
        import folder_paths
        files = folder_paths.get_filename_list("text_encoders")
        if files:
            return sorted(set(files + _KNOWN_TEXT_ENCODERS))
    except Exception:
        pass
    return _KNOWN_TEXT_ENCODERS


def _get_vae_models():
    try:
        import folder_paths
        files = folder_paths.get_filename_list("vae")
        if files:
            return sorted(set(files + _KNOWN_VAE_MODELS))
    except Exception:
        pass
    return _KNOWN_VAE_MODELS


def _get_clip_vision_models():
    try:
        import folder_paths
        files = folder_paths.get_filename_list("clip_vision")
        if files:
            return sorted(set(files + _KNOWN_CLIP_VISION))
    except Exception:
        pass
    return _KNOWN_CLIP_VISION


def _get_checkpoints():
    try:
        import folder_paths
        files = folder_paths.get_filename_list("checkpoints")
        if files:
            return sorted(set(files + _KNOWN_CHECKPOINTS))
    except Exception:
        pass
    return _KNOWN_CHECKPOINTS


class UNETLoader:
    """Override native UNETLoader — loads DiT via fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_get_diffusion_models(),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"advanced": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "model/loaders"

    def load_unet(self, unet_name, weight_dtype="default"):
        from core.wrappers import FusionModelWrapper, _map_unet_name_to_model_name, _infer_model_type, _resolve_model_path
        model_name = _map_unet_name_to_model_name(unet_name)
        model_type = _infer_model_type(model_name)
        model_path = _resolve_model_path(model_name)
        wrapper = FusionModelWrapper(
            model_path=model_path,
            model_name=model_name,
            model_type=model_type,
        )
        logger.info("UNETLoader override: unet=%s -> model=%s", unet_name, model_name)
        return (wrapper,)


class CLIPLoader:
    """Override native CLIPLoader — creates CLIP wrapper for fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (_get_text_encoders(),),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", "boogu", "krea2", "joyimage"],),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            }
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "model/loaders"

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        from core.wrappers import FusionCLIPWrapper, _map_clip_type_to_model_name, _resolve_model_path
        model_name = _map_clip_type_to_model_name(type, clip_name)
        model_path = _resolve_model_path(model_name)
        wrapper = FusionCLIPWrapper(
            model_path=model_path,
            model_name=model_name,
            clip_type=type,
        )
        logger.info("CLIPLoader override: clip=%s type=%s -> model=%s", clip_name, type, model_name)
        return (wrapper,)


class DualCLIPLoader:
    """Override native DualCLIPLoader — creates dual CLIP wrapper for fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name1": (_get_text_encoders(),),
                "clip_name2": (_get_text_encoders(),),
                "type": (["sdxl", "sd3", "flux", "hunyuan_video", "wan"],),
            }
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "model/loaders"

    def load_clip(self, clip_name1, clip_name2, type="sdxl"):
        from core.wrappers import FusionCLIPWrapper, _map_clip_type_to_model_name, _resolve_model_path
        model_name = _map_clip_type_to_model_name(type, clip_name1)
        model_path = _resolve_model_path(model_name)
        wrapper = FusionCLIPWrapper(
            model_path=model_path,
            model_name=model_name,
            clip_type=type,
        )
        logger.info("DualCLIPLoader override: clip1=%s clip2=%s type=%s -> model=%s", clip_name1, clip_name2, type, model_name)
        return (wrapper,)


class VAELoader:
    """Override native VAELoader — creates VAE wrapper for fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_name": (_get_vae_models(),)}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "model/loaders"

    def load_vae(self, vae_name):
        from core.wrappers import FusionVAEWrapper, _resolve_model_path, _map_vae_name_to_model_name
        model_name = _map_vae_name_to_model_name(vae_name)
        model_path = _resolve_model_path(model_name)
        wrapper = FusionVAEWrapper(
            model_path=model_path,
            model_name=vae_name,
        )
        logger.info("VAELoader override: vae=%s", vae_name)
        return (wrapper,)


class CheckpointLoaderSimple:
    """Override native CheckpointLoaderSimple — loads all components via fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (_get_checkpoints(),)}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "model/loaders"

    def load_checkpoint(self, ckpt_name):
        from core.wrappers import (
            FusionModelWrapper, FusionCLIPWrapper, FusionVAEWrapper,
            _infer_model_type, _resolve_model_path, _map_checkpoint_to_model_name,
        )
        model_name = _map_checkpoint_to_model_name(ckpt_name)
        model_type = _infer_model_type(model_name)
        model_path = _resolve_model_path(model_name)
        model = FusionModelWrapper(model_path=model_path, model_name=model_name, model_type=model_type)
        clip = FusionCLIPWrapper(model_path=model_path, model_name=model_name, clip_type=model_type, model_wrapper=model)
        vae = FusionVAEWrapper(model_path=model_path, model_name=model_name, model_wrapper=model)
        logger.info("CheckpointLoaderSimple override: ckpt=%s -> model=%s path=%s", ckpt_name, model_name, model_path)
        return (model, clip, vae)


class ImageOnlyCheckpointLoader:
    """Override native ImageOnlyCheckpointLoader — same as CheckpointLoaderSimple."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (_get_checkpoints(),)}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "model/loaders"

    def load_checkpoint(self, ckpt_name):
        loader = CheckpointLoaderSimple()
        return loader.load_checkpoint(ckpt_name)


class FusionModelLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = _get_checkpoints() + _get_diffusion_models()
        return {
            "required": {
                "model_name": (models,),
                "offload_strategy": (["sequential", "none"], {"default": "sequential"}),
                "quant_bit": (["fp8_e4m3", "4bit", "fp16"], {"default": "fp8_e4m3"}),
            }
        }

    RETURN_TYPES = ("FUSION_PIPELINE",)
    RETURN_NAMES = ("fusion_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Fusion-MLX/Loaders"

    def load_pipeline(self, model_name, offload_strategy, quant_bit):
        from core.engine_wrapper import FusionEngineWrapper
        pipeline = FusionEngineWrapper(
            model_name=model_name,
            offload_strategy=offload_strategy,
            quant_bit=quant_bit,
        )
        logger.info("FusionModelLoader: model=%s offload=%s quant=%s", model_name, offload_strategy, quant_bit)
        return (pipeline,)
