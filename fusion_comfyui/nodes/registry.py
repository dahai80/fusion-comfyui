import logging
from pathlib import Path

import mlx.core as mx

from fusion_comfyui.nodes.base import BaseNode
from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper, StepCallback
from fusion_comfyui.core.lifecycle import FusionMemoryGuardian

logger = logging.getLogger("fusion_comfyui.nodes.registry")

_step_callback: StepCallback | None = None


def set_global_step_callback(cb: StepCallback | None):
    global _step_callback
    _step_callback = cb
    logger.debug("Global step callback %s", "set" if cb else "cleared")


def get_available_models() -> list[str]:
    try:
        from fusion_mlx.model_registry import list_available_models as _list_models
        image_models = [m["name"] for m in _list_models("image")]
        video_models = [m["name"] for m in _list_models("video")]
        models = image_models + video_models
        if models:
            logger.info("Discovered %d models from registry: %s", len(models), models)
            return models
    except Exception as e:
        logger.debug("list_available_models failed: %s, using fallback", e)
    return [
        "flux2-dev",
        "flux-dev",
        "flux-schnell",
        "wan2.1-t2v-14b",
        "wan2.1-t2v-1.3b",
        "skyreels-v2",
        "ltx-video",
    ]


AVAILABLE_MODELS = get_available_models()


class FusionModelLoader(BaseNode):
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "fusion-mlx/loaders"

    @classmethod
    def INPUT_TYPES(cls):
        models = get_available_models()
        return {
            "required": {
                "model_name": (models, {"default": models[0] if models else "flux2-dev"}),
                "offload_strategy": (["sequential", "aggressive", "none"], {"default": "sequential"}),
                "quant_bit": (["fp8_e4m3", "4bit", "none"], {"default": "fp8_e4m3"}),
            }
        }

    async def execute(self, model_name, offload_strategy="sequential", quant_bit="fp8_e4m3"):
        logger.info("FusionModelLoader: %s offload=%s quant=%s", model_name, offload_strategy, quant_bit)
        wrapper = FusionEngineWrapper(model_name, offload_strategy, quant_bit)
        if _step_callback:
            wrapper.set_progress_callback(_step_callback)
        return (wrapper,)


class FusionTextEncoder(BaseNode):
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "fusion-mlx/conditioning"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    async def execute(self, model, prompt="", negative_prompt=""):
        logger.info("FusionTextEncoder: prompt=%dchars neg=%dchars", len(prompt), len(negative_prompt))
        await model.load_text_encoder()
        cond = await model.encode_text(prompt, negative_prompt)
        await model.unload_text_encoder()
        return (cond,)


class FusionKSampler(BaseNode):
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "fusion-mlx/sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 257}),
            }
        }

    async def execute(self, model, positive, negative, steps=20, cfg=6.0, seed=0, width=1024, height=1024, num_frames=1):
        logger.info("FusionKSampler: steps=%d cfg=%.1f seed=%d %dx%d frames=%d", steps, cfg, seed, width, height, num_frames)
        if _step_callback:
            model.set_progress_callback(_step_callback)
        await model.load_dit()
        latent = model.create_empty_latent(height, width, num_frames)
        result = await model.denoise(
            latent, positive, negative,
            steps=steps, cfg=cfg, seed=seed,
            width=width, height=height, num_frames=num_frames,
        )
        await model.unload_dit()
        return (result,)


class FusionVAEDecoder(BaseNode):
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "fusion-mlx/vae"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 512}),
            }
        }

    async def execute(self, model, latent, tile_size=256):
        logger.info("FusionVAEDecoder: tile_size=%d", tile_size)
        await model.load_vae()
        image = await model.decode_tiled(latent, tile_size=tile_size)
        await model.unload_vae()
        if isinstance(image, mx.array):
            if image.ndim == 4 and image.shape[-1] != 3:
                image = mx.transpose(image, (0, 2, 3, 1))
            mx.eval(image)
        return (image,)


class SaveImage(BaseNode):
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    CATEGORY = "fusion-mlx/output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "fusion"}),
            }
        }

    async def execute(self, images, filename_prefix="fusion"):
        import os
        import time
        import numpy as np
        from PIL import Image as PILImage

        output_dir = os.environ.get("FUSION_OUTPUT_DIR", "output")
        os.makedirs(output_dir, exist_ok=True)

        saved = []
        arr = np.array(images) if isinstance(images, mx.array) else images
        if arr.ndim == 4:
            for i in range(arr.shape[0]):
                frame = (arr[i] * 255).clip(0, 255).astype(np.uint8)
                if frame.shape[-1] != 3:
                    frame = frame.transpose(1, 2, 0) if frame.ndim == 3 else frame
                img = PILImage.fromarray(frame)
                ts = int(time.time() * 1000)
                fname = f"{filename_prefix}_{ts}_{i}.png"
                fpath = os.path.join(output_dir, fname)
                img.save(fpath)
                saved.append({"filename": fname, "subfolder": "", "type": "output"})
                logger.info("SaveImage: saved %s", fpath)
        elif arr.ndim == 3:
            frame = (arr * 255).clip(0, 255).astype(np.uint8)
            if frame.shape[-1] != 3:
                frame = frame.transpose(1, 2, 0)
            img = PILImage.fromarray(frame)
            ts = int(time.time() * 1000)
            fname = f"{filename_prefix}_{ts}.png"
            fpath = os.path.join(output_dir, fname)
            img.save(fpath)
            saved.append({"filename": fname, "subfolder": "", "type": "output"})
            logger.info("SaveImage: saved %s", fpath)

        FusionMemoryGuardian.purge_memory()
        return (images,)


class PreviewVideo(BaseNode):
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    CATEGORY = "fusion-mlx/output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "fusion_video"}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60}),
            }
        }

    async def execute(self, images, filename_prefix="fusion_video", fps=16):
        import os
        import time
        import numpy as np

        output_dir = os.environ.get("FUSION_OUTPUT_DIR", "output")
        os.makedirs(output_dir, exist_ok=True)

        arr = np.array(images) if isinstance(images, mx.array) else images
        if arr.ndim < 4:
            return (images,)

        frames = []
        for i in range(arr.shape[0]):
            f = (arr[i] * 255).clip(0, 255).astype(np.uint8)
            if f.shape[-1] != 3:
                f = f.transpose(1, 2, 0) if f.ndim == 3 else f
            frames.append(f)

        ts = int(time.time() * 1000)
        fname = f"{filename_prefix}_{ts}.mp4"
        fpath = os.path.join(output_dir, fname)

        try:
            import imageio
            writer = imageio.get_writer(fpath, fps=fps, codec="libx264", quality=8)
            for f in frames:
                writer.append_data(f)
            writer.close()
            logger.info("PreviewVideo: saved %s (%d frames)", fpath, len(frames))
        except ImportError:
            logger.warning("imageio not available, skipping video save")

        FusionMemoryGuardian.purge_memory()
        return (images,)


def _get_video_models() -> list[str]:
    models = get_available_models()
    video_kw = ("wan", "skyreels", "ltx", "video", "cogvideo", "hunyuan")
    video = [m for m in models if any(k in m.lower() for k in video_kw)]
    return video if video else ["wan2.1-t2v-14b", "wan2.1-t2v-1.3b", "skyreels-v2", "ltx-video"]


class FusionImageToVideo(BaseNode):
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "fusion-mlx/video"

    @classmethod
    def INPUT_TYPES(cls):
        _get_video_models()
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 257}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60}),
            }
        }

    async def execute(self, model, image, positive, negative, steps=30, cfg=6.0, seed=0, num_frames=49, fps=16):
        logger.info("FusionI2V: steps=%d cfg=%.1f seed=%d frames=%d fps=%d", steps, cfg, seed, num_frames, fps)
        if _step_callback:
            model.set_progress_callback(_step_callback)
        # I2V via monolithic generate() — fusion-mlx VideoGenEngine.generate()
        # accepts image kwarg for I2V-capable backends (wan2.1, skyreels).
        # No standalone encode_image() exists on the engine.
        prompt = positive.get("prompt", "") if isinstance(positive, dict) else ""
        import tempfile
        import numpy as np
        from PIL import Image as PILImage
        # Convert image tensor -> temp file for generate() image param
        if isinstance(image, mx.array):
            arr = np.array(image)
        else:
            arr = np.array(image)
        if arr.ndim == 4:
            arr = arr[0]
        frame = (arr * 255).clip(0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[-1] != 3:
            frame = frame.transpose(1, 2, 0)
        pil_img = PILImage.fromarray(frame)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_img.save(tmp, format="PNG")
            tmp_path = tmp.name
        try:
            result = await model.generate_i2v(
                prompt=prompt,
                image_path=tmp_path,
                num_frames=num_frames,
                seed=seed,
            )
        finally:
            import os
            os.unlink(tmp_path)
        if result is None:
            logger.warning("generate_i2v returned None, returning empty latent")
            return (model.create_empty_latent(frame.shape[0], frame.shape[1], num_frames),)
        return (result,)


class FusionVideoToVideo(BaseNode):
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "fusion-mlx/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "denoise_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    async def execute(self, model, latent, positive, negative, steps=20, cfg=6.0, seed=0, denoise_strength=0.7):
        logger.info("FusionV2V: steps=%d cfg=%.1f denoise=%.2f", steps, cfg, denoise_strength)
        if _step_callback:
            model.set_progress_callback(_step_callback)
        effective_steps = max(1, int(steps * denoise_strength))
        await model.load_dit()
        result = await model.denoise(
            latent, positive, negative,
            steps=effective_steps, cfg=cfg, seed=seed,
        )
        await model.unload_dit()
        return (result,)


class FusionControlNet(BaseNode):
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "fusion-mlx/conditioning"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "control_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_pct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_pct": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    async def execute(self, conditioning, control_image, strength=1.0, start_pct=0.0, end_pct=1.0):
        logger.info("FusionControlNet: strength=%.2f range=[%.2f,%.2f]", strength, start_pct, end_pct)
        if isinstance(conditioning, dict):
            cond = dict(conditioning)
        else:
            cond = {"embeds": conditioning}
        if isinstance(control_image, mx.array):
            cond["control_hint"] = control_image
        else:
            import numpy as np
            cond["control_hint"] = mx.array(np.array(control_image), dtype=mx.float32)
        cond["control_strength"] = strength
        cond["control_start_pct"] = start_pct
        cond["control_end_pct"] = end_pct
        return (cond,)


class FusionInpaint(BaseNode):
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "fusion-mlx/sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "mask": ("MASK",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    async def execute(self, model, positive, negative, latent, mask, steps=20, cfg=6.0, seed=0, denoise_strength=1.0):
        logger.info("FusionInpaint: steps=%d cfg=%.1f denoise=%.2f", steps, cfg, denoise_strength)
        if _step_callback:
            model.set_progress_callback(_step_callback)
        if isinstance(mask, mx.array):
            if mask.ndim == 2:
                mask = mask.reshape(1, 1, *mask.shape)
            elif mask.ndim == 3:
                mask = mask.reshape(1, *mask.shape)
        else:
            import numpy as np
            mask = mx.array(np.array(mask), dtype=mx.float32)
            if mask.ndim == 2:
                mask = mask.reshape(1, 1, *mask.shape)
            elif mask.ndim == 3:
                mask = mask.reshape(1, *mask.shape)
        if isinstance(latent, mx.array) and mask.shape[-2:] != latent.shape[-2:]:
            import numpy as np
            from PIL import Image as PILImage
            mask_np = np.array(mask).squeeze()
            target_h, target_w = latent.shape[-2], latent.shape[-1]
            mask_resized = np.array(
                PILImage.fromarray((mask_np * 255).astype(np.uint8)).resize(
                    (target_w, target_h), PILImage.Resampling.BILINEAR
                )
            ).astype(np.float32) / 255.0
            mask = mx.array(mask_resized.reshape(1, 1, target_h, target_w))
        noisy = latent * (1.0 - mask)
        effective_steps = max(1, int(steps * denoise_strength))
        await model.load_dit()
        result = await model.denoise(
            noisy, positive, negative,
            steps=effective_steps, cfg=cfg, seed=seed,
            mask=mask,
        )
        await model.unload_dit()
        return (result,)


class FusionLoadImage(BaseNode):
    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "fusion-mlx/loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""}),
            }
        }

    async def execute(self, image_path=""):
        import numpy as np
        from PIL import Image as PILImage
        logger.info("FusionLoadImage: %s", image_path)
        if not image_path or not Path(image_path).exists():
            blank = mx.zeros((512, 512, 3), dtype=mx.float32)
            return (blank, mx.ones((512, 512), dtype=mx.float32))
        img = PILImage.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        image = mx.array(arr)
        mask = mx.ones(arr.shape[:2], dtype=mx.float32)
        return (image, mask)


class FusionLatentFromImage(BaseNode):
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "fusion-mlx/latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
            }
        }

    async def execute(self, model, image):
        logger.info("FusionLatentFromImage: encoding image to latent")
        # fusion-mlx has no standalone VAE encode API.
        # The pipeline stage API only exposes decode, not encode.
        # For img2img workflows, use the monolithic generate() with
        # edit_image / image_strength params on ImageGenEngine.
        # This node creates a latent of matching spatial dims as placeholder.
        import numpy as np
        if isinstance(image, mx.array):
            arr = np.array(image)
        else:
            arr = np.array(image)
        if arr.ndim == 4:
            arr = arr[0]
        h, w = arr.shape[:2]
        latent = model.create_empty_latent(h, w, num_frames=1)
        logger.info("FusionLatentFromImage: placeholder latent shape=%s (no VAE encode API)", tuple(latent.shape))
        return (latent,)


from fusion_comfyui.nodes.xiyouji.registry import (
    NODE_CLASS_MAPPINGS as _XIYOUJI_CLASS,
    NODE_DISPLAY_NAME_MAPPINGS as _XIYOUJI_DISPLAY,
)

NODE_CLASS_MAPPINGS = {
    "FusionModelLoader": FusionModelLoader,
    "FusionTextEncoder": FusionTextEncoder,
    "FusionKSampler": FusionKSampler,
    "FusionVAEDecoder": FusionVAEDecoder,
    "SaveImage": SaveImage,
    "PreviewVideo": PreviewVideo,
    "FusionImageToVideo": FusionImageToVideo,
    "FusionVideoToVideo": FusionVideoToVideo,
    "FusionControlNet": FusionControlNet,
    "FusionInpaint": FusionInpaint,
    "FusionLoadImage": FusionLoadImage,
    "FusionLatentFromImage": FusionLatentFromImage,
}
NODE_CLASS_MAPPINGS.update(_XIYOUJI_CLASS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "FusionModelLoader": "⚡ Fusion-MLX Model Loader",
    "FusionTextEncoder": "⚡ Fusion-MLX Text Encoder",
    "FusionKSampler": "⚡ Fusion-MLX Sampler",
    "FusionVAEDecoder": "⚡ Fusion-MLX VAE Decoder",
    "SaveImage": "⚡ Save Image",
    "PreviewVideo": "⚡ Preview Video",
    "FusionImageToVideo": "⚡ Fusion-MLX Image→Video",
    "FusionVideoToVideo": "⚡ Fusion-MLX Video→Video",
    "FusionControlNet": "⚡ Fusion-MLX ControlNet Apply",
    "FusionInpaint": "⚡ Fusion-MLX Inpaint",
    "FusionLoadImage": "⚡ Fusion-MLX Load Image",
    "FusionLatentFromImage": "⚡ Fusion-MLX Image→Latent",
}
NODE_DISPLAY_NAME_MAPPINGS.update(_XIYOUJI_DISPLAY)


def build_node_info() -> dict:
    info = {}
    for name, cls in NODE_CLASS_MAPPINGS.items():
        input_types = cls.INPUT_TYPES()
        info[name] = {
            "input": input_types,
            "output": list(cls.RETURN_TYPES),
            "output_is_list": [False] * len(cls.RETURN_TYPES),
            "output_name": [f"output_{i}" for i in range(len(cls.RETURN_TYPES))],
            "category": cls.CATEGORY,
        }
    return info
