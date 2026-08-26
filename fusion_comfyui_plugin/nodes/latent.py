import logging

import mlx.core as mx
import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.latent")


class EmptyLatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": 16384, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, width=512, height=512, batch_size=1):
        latent = mx.zeros((batch_size, 4, height // 8, width // 8), dtype=mx.float32)
        logger.info("EmptyLatentImage: shape=%s (%dx%d)", latent.shape, width, height)
        return ({"samples": latent, "width": width, "height": height},)


class EmptyHunyuanLatentVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 848, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 33, "min": 1, "max": 4096, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, width=848, height=480, length=33, batch_size=1):
        latent = mx.zeros((batch_size, 16, (length - 1) // 4 + 1, height // 8, width // 8), dtype=mx.float32)
        logger.info("EmptyHunyuanLatentVideo: shape=%s (%dx%d frames=%d)", latent.shape, width, height, length)
        return ({"samples": latent, "num_frames": length, "width": width, "height": height, "downscale_ratio_spacial": 8},)


class EmptyCosmosLatentVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1280, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 704, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 121, "min": 1, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, width=1280, height=704, length=121, batch_size=1):
        t_latent = max(2, (length // 4 // 2) * 2)
        h_latent = (height // 8 // 2) * 2
        w_latent = (width // 8 // 2) * 2
        latent = mx.zeros((batch_size, 16, t_latent, h_latent, w_latent), dtype=mx.float32)
        logger.info("EmptyCosmosLatentVideo: shape=%s (%dx%d frames=%d)", latent.shape, width, height, length)
        return ({"samples": latent, "num_frames": length, "width": width, "height": height},)


class Wan22ImageToVideoLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "width": ("INT", {"default": 1280, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 704, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 49, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "start_image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, vae, width=1280, height=704, length=49, batch_size=1, start_image=None):
        num_frames = length
        latent_t = (num_frames - 1) // 4 + 1
        latent = mx.zeros((1, 48, latent_t, height // 16, width // 16), dtype=mx.float32)

        result = {"samples": latent, "num_frames": num_frames, "width": width, "height": height}

        if start_image is not None:
            import tempfile
            from PIL import Image as PILImage
            from fusion_comfyui.core.bridge import to_numpy
            img_arr = to_numpy(start_image)
            if img_arr.ndim == 4:
                img_arr = img_arr[0]
            pil_img = PILImage.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="fusion_i2v_")
            pil_img.save(tmp.name)
            tmp.close()
            result["_i2v_image_path"] = tmp.name
            logger.info("Wan22ImageToVideoLatent: saved start_image to %s for i2v", tmp.name)
        else:
            logger.info("Wan22ImageToVideoLatent: t2v mode")

        logger.info("Wan22ImageToVideoLatent: shape=%s (%dx%d frames=%d)", latent.shape, width, height, num_frames)
        return (result,)

class WanImageToVideo:
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
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, positive, negative, vae, width=832, height=480, length=81, batch_size=1, clip_vision_output=None, start_image=None):
        num_frames = length
        latent_t = (num_frames - 1) // 4 + 1
        latent = mx.zeros((batch_size, 16, latent_t, height // 8, width // 8), dtype=mx.float32)

        result = {"samples": latent, "num_frames": num_frames, "width": width, "height": height}

        if start_image is not None:
            import tempfile
            from PIL import Image as PILImage
            from fusion_comfyui.core.bridge import to_numpy
            img_arr = to_numpy(start_image)
            if img_arr.ndim == 4:
                img_arr = img_arr[0]
            pil_img = PILImage.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="fusion_i2v_")
            pil_img.save(tmp.name)
            tmp.close()
            result["_i2v_image_path"] = tmp.name
            logger.info("WanImageToVideo: saved start_image to %s for i2v", tmp.name)

        logger.info("WanImageToVideo: shape=%s (%dx%d frames=%d)", latent.shape, width, height, num_frames)
        return (positive, negative, result)

class LTXVImgToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "width": ("INT", {"default": 768, "min": 64, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 97, "min": 9, "max": 16384, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {},
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "model/latent"

    def generate(self, positive, negative, vae, image, width=768, height=512, length=97, batch_size=1, strength=1.0):
        num_frames = length
        latent_t = (num_frames - 1) // 8 + 1
        latent = mx.zeros((batch_size, 128, latent_t, height // 32, width // 32), dtype=mx.float32)

        result = {"samples": latent, "num_frames": num_frames, "width": width, "height": height}

        if image is not None:
            import tempfile
            from PIL import Image as PILImage
            from fusion_comfyui.core.bridge import to_numpy
            img_arr = to_numpy(image)
            if img_arr.ndim == 4:
                img_arr = img_arr[0]
            pil_img = PILImage.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="fusion_i2v_")
            pil_img.save(tmp.name)
            tmp.close()
            result["_i2v_image_path"] = tmp.name
            result["_i2v_image_strength"] = strength
            logger.info("LTXVImgToVideo: saved image to %s for i2v strength=%.2f", tmp.name, strength)

        logger.info("LTXVImgToVideo: shape=%s (%dx%d frames=%d strength=%.2f)", latent.shape, width, height, num_frames, strength)
        return (positive, negative, result)

class FusionEmptyLatentNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 257}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/Latent"

    def generate(self, width=1024, height=1024, batch_size=1, num_frames=1):
        latent_channels = 16
        latent_width = width // 2
        latent_height = height // 2

        if num_frames > 1:
            latent_time = (num_frames - 1) // 4 + 1
            shape = (batch_size, latent_channels, latent_time, latent_height, latent_width)
        else:
            shape = (batch_size, latent_channels, latent_height, latent_width)

        latent = mx.zeros(shape, dtype=mx.float32)

        logger.info(
            "FusionEmptyLatent: shape=%s (%dx%d, frames=%d)",
            shape, width, height, num_frames,
        )

        return ({"samples": latent, "num_frames": num_frames, "width": width, "height": height},)
