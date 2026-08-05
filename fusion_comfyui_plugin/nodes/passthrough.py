import logging
import math

import mlx.core as mx
import numpy as np

from ._sampler_constants import SAMPLER_NAMES

logger = logging.getLogger("fusion_comfyui.nodes.passthrough")


class ModelSamplingSD3:
    """Passthrough — no-op for fusion-mlx (sampling config handled by backend)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/advanced"

    def patch(self, model, shift):
        logger.info("ModelSamplingSD3 passthrough: shift=%.2f", shift)
        return (model,)


class ModelSamplingContinuousEDM:
    """Passthrough — no-op for fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sampling": (["v_prediction", "edm", "edm_playground_v2.5", "eps", "cosmos_rflow"],),
                "sigma_max": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sigma_min": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1000.0, "step": 0.001}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/advanced"

    def patch(self, model, sampling, sigma_max, sigma_min):
        logger.info("ModelSamplingContinuousEDM passthrough: %s sigma=%.3f-%.1f", sampling, sigma_min, sigma_max)
        return (model,)


class ModelSamplingFlux:
    """Passthrough — no-op for fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/advanced"

    def patch(self, model, width, height, max_shift, base_shift, shift):
        logger.info("ModelSamplingFlux passthrough: %dx%d shift=%.2f", width, height, shift)
        return (model,)


class BasicGuider:
    """Passthrough — guider selection is irrelevant for fusion-mlx generate()."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
            }
        }
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "model/advanced"

    def get_guider(self, model, conditioning):
        logger.info("BasicGuider passthrough")
        return ({"model": model, "conditioning": conditioning},)


class BasicScheduler:
    """Passthrough — scheduler config is handled by fusion-mlx backend."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "model/advanced"

    def get_sigmas(self, model, scheduler, steps, denoise):
        logger.info("BasicScheduler passthrough: %s steps=%d", scheduler, steps)
        sigmas = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)
        return (sigmas,)


class KSamplerSelect:
    """Passthrough — returns a stub sampler object."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (list(SAMPLER_NAMES),),
            }
        }
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "model/sampling"

    def get_sampler(self, sampler_name):
        logger.info("KSamplerSelect passthrough: %s", sampler_name)
        return ({"sampler_name": sampler_name},)


class RandomNoise:
    """Passthrough — noise generation handled by fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "model/noise"

    def get_noise(self, noise_seed):
        logger.info("RandomNoise passthrough: seed=%d", noise_seed)
        return ({"noise_seed": noise_seed},)


class FluxGuidance:
    """Passthrough — guidance config handled by fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"
    CATEGORY = "model/advanced"

    def append(self, conditioning, guidance):
        logger.info("FluxGuidance passthrough: guidance=%.1f", guidance)
        if isinstance(conditioning, dict):
            conditioning = conditioning.copy()
            conditioning["guidance"] = guidance
        return (conditioning,)


class CLIPVisionLoader:
    """Passthrough — CLIP vision not used in fusion-mlx pipeline."""
    @classmethod
    def INPUT_TYPES(cls):
        from .loaders import _get_clip_vision_models
        return {
            "required": {
                "clip_name": (_get_clip_vision_models(),),
            }
        }
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"
    CATEGORY = "model/advanced"

    def load_clip(self, clip_name):
        logger.info("CLIPVisionLoader passthrough: %s", clip_name)
        return ({"clip_name": clip_name},)


class CLIPVisionEncode:
    """Passthrough — CLIP vision encoding not used in fusion-mlx pipeline."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "encode"
    CATEGORY = "model/advanced"

    def encode(self, clip_vision, image):
        logger.info("CLIPVisionEncode passthrough")
        return ({"image": image},)


class LTXVConditioning:
    """Passthrough — conditioning adjustments handled by fusion-mlx backend."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "append"
    CATEGORY = "model/advanced"

    def append(self, positive, negative, frame_rate):
        logger.info("LTXVConditioning passthrough: frame_rate=%.1f", frame_rate)
        if isinstance(positive, dict):
            positive = positive.copy()
            positive["frame_rate"] = frame_rate
        if isinstance(negative, dict):
            negative = negative.copy()
            negative["frame_rate"] = frame_rate
        return (positive, negative)


class LTXVScheduler:
    """Passthrough — scheduler config handled by fusion-mlx backend."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
                "stretch": (["enable", "disable"], {"advanced": True, "default": True}),
                "terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01, "advanced": True}),
            },
            "optional": {
                "latent": ("LATENT",),
            }
        }
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "model/advanced"

    def get_sigmas(self, steps, max_shift, base_shift, stretch, terminal, latent=None):
        if isinstance(stretch, bool):
            stretch = "enable" if stretch else "disable"
        stretch_enabled = stretch == "enable"

        if latent is not None and isinstance(latent, dict):
            samples = latent.get("samples")
            if samples is not None:
                tokens = int(np.prod(samples.shape[2:]))
            else:
                tokens = 4096
        else:
            tokens = 4096

        sigmas = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)

        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b

        power = 1
        sigmas = np.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1.0 / sigmas - 1.0) ** power),
            0,
        )

        if stretch_enabled:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        logger.info("LTXVScheduler: steps=%d tokens=%d stretch=%s sigma_range=[%.4f,%.4f]", steps, tokens, stretch, sigmas[0], sigmas[-1])
        return (sigmas,)


class CosmosImageToVideoLatent:
    """Passthrough — creates empty latent for Cosmos i2v."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "width": ("INT", {"default": 1280, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 704, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 121, "min": 1, "max": 16384, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, vae, width=1280, height=704, length=121, batch_size=1, start_image=None, end_image=None):
        latent = mx.zeros((batch_size, 16, (length - 1) // 8 + 1, height // 8, width // 8), dtype=mx.float32)
        result = {"samples": latent, "num_frames": length, "width": width, "height": height}
        if start_image is not None:
            import tempfile
            from PIL import Image as PILImage
            img_arr = start_image
            if isinstance(img_arr, mx.array):
                img_arr = np.array(img_arr)
            if img_arr.ndim == 4:
                img_arr = img_arr[0]
            pil_img = PILImage.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="fusion_cosmos_i2v_")
            pil_img.save(tmp.name)
            tmp.close()
            result["_i2v_image_path"] = tmp.name
            logger.info("CosmosImageToVideoLatent: saved start_image to %s for i2v", tmp.name)
        logger.info("CosmosImageToVideoLatent: shape=%s start=%s end=%s", latent.shape, start_image is not None, end_image is not None)
        return (result,)


class CosmosPredict2ImageToVideoLatent:
    """Passthrough — creates empty latent for Cosmos Predict2 i2v."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "width": ("INT", {"default": 848, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 93, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, vae, width=848, height=480, length=93, batch_size=1, start_image=None, end_image=None):
        t_latent = max(2, (length // 4 // 2) * 2)
        h_latent = (height // 8 // 2) * 2
        w_latent = (width // 8 // 2) * 2
        latent = mx.zeros((1, 16, t_latent, h_latent, w_latent), dtype=mx.float32)
        result = {"samples": latent, "num_frames": length, "width": width, "height": height}
        if start_image is not None:
            import tempfile
            from PIL import Image as PILImage
            img_arr = start_image
            if isinstance(img_arr, mx.array):
                img_arr = np.array(img_arr)
            if img_arr.ndim == 4:
                img_arr = img_arr[0]
            pil_img = PILImage.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="fusion_cosmos_p2_i2v_")
            pil_img.save(tmp.name)
            tmp.close()
            result["_i2v_image_path"] = tmp.name
            logger.info("CosmosPredict2ImageToVideoLatent: saved start_image to %s for i2v", tmp.name)
        logger.info("CosmosPredict2ImageToVideoLatent: shape=%s start=%s end=%s", latent.shape, start_image is not None, end_image is not None)
        return (result,)


class EmptyLTXVLatentVideo:
    """Creates empty latent for LTXV text-to-video."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                "length": ("INT", {"default": 97, "min": 1, "max": 257, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, width=768, height=512, length=97, batch_size=1):
        latent = mx.zeros((batch_size, 128, (length - 1) // 8 + 1, height // 32, width // 32), dtype=mx.float32)
        logger.info("EmptyLTXVLatentVideo: shape=%s (%dx%d frames=%d)", latent.shape, width, height, length)
        return ({"samples": latent, "num_frames": length, "width": width, "height": height, "downscale_ratio_spacial": 32},)


class HunyuanImageToVideo:
    """Passthrough — creates empty latent for Hunyuan i2v."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 848, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 53, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "guidance_type": (["v1 (concat)", "v2 (replace)", "custom"], {"advanced": True}),
            },
            "optional": {
                "start_image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, positive, vae, width=848, height=480, length=53, batch_size=1, guidance_type="v1 (concat)", start_image=None):
        latent = mx.zeros((batch_size, 16, (length - 1) // 4 + 1, height // 8, width // 8), dtype=mx.float32)
        logger.info("HunyuanImageToVideo: shape=%s guidance=%s start=%s", latent.shape, guidance_type, start_image is not None)
        return (
            positive,
            {"samples": latent, "num_frames": length, "width": width, "height": height},
        )


class LTXVAddGuide:
    """Passthrough — guide addition handled by fusion-mlx backend."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "append"
    CATEGORY = "model/advanced"

    def append(self, positive, negative, vae, latent, image, frame_idx=0, strength=1.0):
        logger.info("LTXVAddGuide passthrough frame_idx=%d strength=%.2f", frame_idx, strength)
        return (positive, negative, latent)


class LTXVCropGuides:
    """Passthrough — guide cropping handled by fusion-mlx backend."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "crop"
    CATEGORY = "model/advanced"

    def crop(self, positive, negative, latent):
        logger.info("LTXVCropGuides passthrough")
        return (positive, negative, latent)


class LTXVPreprocess:
    """Passthrough — preprocessing handled by fusion-mlx backend."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "img_compression": ("INT", {"default": 35, "min": 0, "max": 100}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"
    CATEGORY = "model/advanced"

    def preprocess(self, image, img_compression=35):
        logger.info("LTXVPreprocess passthrough img_compression=%d", img_compression)
        return (image,)


class SVD_img2vid_Conditioning:
    """Passthrough — SVD conditioning not supported by fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "init_image": ("IMAGE",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 576, "min": 16, "max": 16384, "step": 8}),
                "video_frames": ("INT", {"default": 14, "min": 1, "max": 4096}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023, "advanced": True}),
                "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
                "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "model/advanced"

    def encode(self, clip_vision, init_image, vae, width, height, video_frames,
               motion_bucket_id, fps, augmentation_level):
        logger.info("SVD_img2vid_Conditioning passthrough (SVD not supported)")
        latent = mx.zeros((1, 4, video_frames, height // 8, width // 8), dtype=mx.float32)
        positive = {"prompt": ""}
        negative = {"prompt": ""}
        return (positive, negative, {"samples": latent, "num_frames": video_frames, "width": width, "height": height})


class TextEncodeHunyuanVideo_ImageToVideo:
    """Passthrough — Hunyuan i2v text encoding handled by fusion-mlx."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "image_interleave": ("INT", {"default": 2, "min": 1, "max": 512, "advanced": True}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "model/advanced"

    def encode(self, clip, clip_vision_output, prompt, image_interleave=2):
        logger.info("TextEncodeHunyuanVideo_ImageToVideo passthrough: prompt=%s interleave=%d", prompt[:50], image_interleave)
        return ({"prompt": prompt},)


class TrimVideoLatent:
    """Passthrough — trims latent frames."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 4096}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "trim"
    CATEGORY = "model/latent"

    def trim(self, samples, start_frame=0, end_frame=-1):
        logger.info("TrimVideoLatent passthrough: start=%d end=%d", start_frame, end_frame)
        return (samples,)


class VideoLinearCFGGuidance:
    """Passthrough — CFG guidance handled by fusion-mlx backend."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "min_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/advanced"

    def patch(self, model, min_cfg):
        logger.info("VideoLinearCFGGuidance passthrough: min_cfg=%.1f", min_cfg)
        return (model,)


class WanCameraEmbedding:
    """Passthrough — camera embedding handled by fusion-mlx backend.
    Interface matches upstream comfy_extras/nodes_camera_trajectory.py."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose": ([
                    "Static", "Pan Up", "Pan Down", "Pan Left", "Pan Right",
                    "Zoom In", "Zoom Out", "Anti Clockwise (ACW)", "ClockWise (CW)",
                ],),
                "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "fx": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.000000001}),
                "fy": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.000000001}),
                "cx": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cy": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("WAN_CAMERA_EMBEDDING", "INT", "INT", "INT")
    RETURN_NAMES = ("camera_embedding", "width", "height", "length")
    FUNCTION = "append"
    CATEGORY = "model/conditioning/wan/camera"

    def append(self, camera_pose, width=832, height=480, length=81, speed=1.0, fx=0.5, fy=0.5, cx=0.5, cy=0.5):
        import numpy as np
        logger.info("WanCameraEmbedding passthrough: pose=%s %dx%d len=%d speed=%.1f", camera_pose, width, height, length, speed)
        cam_embedding = np.zeros((1, 16, (length - 1) // 4 + 1, height // 8, width // 8), dtype=np.float32)
        return (cam_embedding, width, height, length)


class WanCameraImageToVideo:
    """Passthrough — Wan camera i2v handled by fusion-mlx backend.
    Interface matches upstream comfy_extras/nodes_wan.py."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "start_image": ("IMAGE",),
                "camera_conditions": ("WAN_CAMERA_EMBEDDING",),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "model/conditioning/wan/camera"

    def generate(self, positive, negative, vae, width=832, height=480, length=81, batch_size=1, clip_vision_output=None, start_image=None, camera_conditions=None):
        latent = mx.zeros((batch_size, 16, (length - 1) // 4 + 1, height // 8, width // 8), dtype=mx.float32)
        result = {"samples": latent, "num_frames": length, "width": width, "height": height}
        if start_image is not None:
            import tempfile
            from PIL import Image as PILImage
            img_arr = start_image
            if isinstance(img_arr, mx.array):
                img_arr = np.array(img_arr)
            if img_arr.ndim == 4:
                img_arr = img_arr[0]
            pil_img = PILImage.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="fusion_i2v_")
            pil_img.save(tmp.name)
            tmp.close()
            result["_i2v_image_path"] = tmp.name
            logger.info("WanCameraImageToVideo: saved start_image to %s for i2v", tmp.name)
        else:
            logger.info("WanCameraImageToVideo: t2v mode (no start_image)")
        if camera_conditions is not None:
            logger.warning("WanCameraImageToVideo: camera_conditions ignored - pose conditioning not yet supported by fusion-mlx backend (upstream gap)")
        logger.info("WanCameraImageToVideo: shape=%s start=%s cam=%s", latent.shape, start_image is not None, camera_conditions is not None)
        return (positive, negative, result)


class WanVaceToVideo:
    """Passthrough — Wan VACE handled by fusion-mlx backend."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "control_masks": ("MASK",),
                "reference_image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, positive, negative, vae, width=832, height=480, length=81, batch_size=1, strength=1.0, control_video=None, control_masks=None, reference_image=None):
        import tempfile
        from PIL import Image as PILImage
        latent = mx.zeros((batch_size, 16, (length - 1) // 4 + 1, height // 8, width // 8), dtype=mx.float32)
        result = {"samples": latent, "num_frames": length, "width": width, "height": height}
        result["_vace_strength"] = strength

        try:
            if control_video is not None:
                arr = np.array(control_video) if not isinstance(control_video, np.ndarray) else control_video
                if arr.size > 0:
                    if arr.max() > 1.0:
                        arr = arr.astype(np.float32) / 255.0
                    if arr.ndim == 4:
                        frames = [PILImage.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)) for f in arr[:length]]
                    else:
                        frames = [PILImage.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))]
                    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="fusion_vace_ctrl_")
                    tmp.close()
                    try:
                        import av
                        container = av.open(tmp.name, mode="w")
                        stream = container.add_stream("mpeg4", rate=24)
                        stream.width = width
                        stream.height = height
                        stream.pix_fmt = "yuv420p"
                        for frame in frames:
                            frame = frame.resize((width, height))
                            av_frame = av.VideoFrame.from_image(frame)
                            for packet in stream.encode(av_frame):
                                container.mux(packet)
                        for packet in stream.encode():
                            container.mux(packet)
                        container.close()
                        result["_vace_control_video"] = tmp.name
                        logger.info("WanVaceToVideo: saved control_video (%d frames) to %s", len(frames), tmp.name)
                    except Exception as e:
                        logger.error("WanVaceToVideo: failed to save control video: %s", e)
        except Exception as e:
            logger.warning("WanVaceToVideo: control_video processing skipped: %s", e)

        try:
            if control_masks is not None:
                mask_arr = np.array(control_masks) if not isinstance(control_masks, np.ndarray) else control_masks
                if mask_arr.size > 0:
                    if mask_arr.max() > 1.0:
                        mask_arr = mask_arr.astype(np.float32) / 255.0
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr[:, None, :, :]
                    elif mask_arr.ndim == 2:
                        mask_arr = mask_arr[None, None, :, :]
                    tmp_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="fusion_vace_mask_")
                    first_mask = mask_arr[0, 0]
                    PILImage.fromarray((np.clip(first_mask, 0, 1) * 255).astype(np.uint8)).save(tmp_mask.name)
                    tmp_mask.close()
                    result["_vace_control_mask"] = tmp_mask.name
                    logger.info("WanVaceToVideo: saved control_mask to %s", tmp_mask.name)
        except Exception as e:
            logger.warning("WanVaceToVideo: control_masks processing skipped: %s", e)

        try:
            if reference_image is not None:
                arr = np.array(reference_image) if not isinstance(reference_image, np.ndarray) else reference_image
                if arr.size > 0:
                    if arr.max() > 1.0:
                        arr = arr.astype(np.float32) / 255.0
                    if arr.ndim == 4:
                        arr = arr[0]
                    pil_img = PILImage.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                    tmp_ref = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="fusion_vace_ref_")
                    pil_img.save(tmp_ref.name)
                    tmp_ref.close()
                    result["_vace_reference_images"] = [tmp_ref.name]
                    logger.info("WanVaceToVideo: saved reference_image to %s", tmp_ref.name)
        except Exception as e:
            logger.warning("WanVaceToVideo: reference_image processing skipped: %s", e)

        trim_latent_val = 0
        logger.info("WanVaceToVideo: shape=%s strength=%.2f ctrl=%s mask=%s ref=%s",
                     latent.shape, strength,
                     "yes" if "_vace_control_video" in result else "no",
                     "yes" if "_vace_control_mask" in result else "no",
                     "yes" if "_vace_reference_images" in result else "no")
        return (positive, negative, result, trim_latent_val)


class Note:
    """Passthrough — UI annotation node, no execution effect."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "hidden": {}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "annotate"
    CATEGORY = "utils"

    def annotate(self, **kwargs):
        logger.info("Note passthrough")
        return {}
