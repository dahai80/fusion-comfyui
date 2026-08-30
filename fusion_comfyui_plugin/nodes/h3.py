import logging
import os
import tempfile

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.h3")


def _save_temp_image(image, prefix):
    from PIL import Image as PILImage
    from fusion_comfyui.core.bridge import to_numpy
    arr = to_numpy(image)
    if arr.ndim == 4:
        arr = arr[0]
    rgb = (np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, :3]
    pil = PILImage.fromarray(rgb)
    # Write under /tmp explicitly: minimax_h3 _ALLOWED_READ_DIRS only permits
    # /tmp, NOT macOS $TMPDIR (/var/folders/.../T) which NamedTemporaryFile uses.
    fd, path = tempfile.mkstemp(suffix=".png", prefix=f"fusion_{prefix}_", dir="/tmp")
    with os.fdopen(fd, "wb") as fh:
        pil.save(fh, format="PNG")
    logger.info("_save_temp_image: saved %dx%d -> %s", rgb.shape[1], rgb.shape[0], path)
    return path


def _apply_768p_override(width, height):
    # AICF hardcodes 16:9 -> 960x544 (540p) in provider.ts; that code is off-limits.
    # FUSION_H3_VIDEO_768P=1 raises the video resolution to 768p inside this node,
    # preserving aspect and rounding to mult of 32. H3 VAE spatial /16 then patchify
    # /2 (patch_size=2) -> latent dims must be even -> width/height mult of 32.
    # Rounding to 16 alone yields odd latent dims (e.g. 1360/16=85) and trips
    # patchify_video_latents "not divisible by patch (1,2,2)". Only raises, never
    # lowers. 768p ~1.76x pixels of 540p -> shots must be <= ~3s to stay under
    # AICF's 30min poll deadline (see AICF provider.ts timeout 1_800_000).
    if os.environ.get("FUSION_H3_VIDEO_768P", "0") != "1":
        return width, height
    target_short = 768
    short = min(width, height)
    if short >= target_short:
        return width, height
    scale = target_short / short
    new_w = int(round(width * scale / 32)) * 32
    new_h = int(round(height * scale / 32)) * 32
    logger.info("_apply_768p_override: %dx%d -> %dx%d (FUSION_H3_VIDEO_768P=1)",
                width, height, new_w, new_h)
    return new_w, new_h



class MiniMaxH3SigmaShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "shift"
    CATEGORY = "Fusion-MLX/H3"

    def shift(self, model, shift_video=12.0, shift_audio=3.0):
        logger.info("MiniMaxH3SigmaShift: shift_video=%.1f shift_audio=%.1f (passthrough, MLX uses defaults)", shift_video, shift_audio)
        return (model,)


class EmptyMiniMaxH3LatentAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 960, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 73, "min": 5, "max": 3600, "step": 17}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/H3"

    def generate(self, width=960, height=544, length=73, batch_size=1):
        # H3 video VAE: z_channels=24, spatial /16 (vae_ratio), temporal /4 causal
        # (vae_ratio_t). Verified in fusion_mlx/video/minimax_h3/config.py
        # H3VAEConfig + generate.py _latents_shape. t_latent=(length-1)//4+1.
        import mlx.core as mx
        width, height = _apply_768p_override(width, height)
        t_latent = (length - 1) // 4 + 1
        latent = mx.zeros((batch_size, 24, t_latent, height // 16, width // 16), dtype=mx.float32)
        logger.info("EmptyMiniMaxH3LatentAV: shape=%s %dx%d frames=%d", latent.shape, width, height, length)
        return ({"samples": latent, "num_frames": length, "width": width, "height": height, "_h3_audio": True},)


class MiniMaxH3ImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 960, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 73, "min": 5, "max": 3600, "step": 17}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "quantize": (["dit8_te4", "dit8", "te4", "none"], {"default": "dit8_te4"}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/H3"

    def generate(self, clip, vae, prompt="", width=960, height=544, length=73,
                 first_frame=None, last_frame=None, quantize="dit8_te4"):
        # H3 latent: z_channels=24, spatial /16, temporal /4. See EmptyMiniMaxH3LatentAV.
        # audio forced False: fl2va (image/last_frame) is video-only, audio+image
        # mutually exclusive (generate_video raises). Only t2va (no image) may set audio.
        import mlx.core as mx
        width, height = _apply_768p_override(width, height)
        t_latent = (length - 1) // 4 + 1
        latent = mx.zeros((1, 24, t_latent, height // 16, width // 16), dtype=mx.float32)
        result = {"samples": latent, "num_frames": length, "width": width, "height": height,
                  "_h3_audio": False, "_h3_quantize": quantize}
        if first_frame is not None:
            result["_h3_first_frame_path"] = _save_temp_image(first_frame, "h3_i2v_first")
        if last_frame is not None:
            result["_h3_last_frame_path"] = _save_temp_image(last_frame, "h3_i2v_last")
        logger.info("MiniMaxH3ImageToVideo: %dx%d frames=%d quantize=%s first=%s last=%s",
                     width, height, length, quantize,
                     first_frame is not None, last_frame is not None)
        return ({"prompt": prompt}, result)

    @staticmethod
    def _save_temp_image(image, prefix):
        return _save_temp_image(image, prefix)


class MiniMaxH3ReferenceToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 960, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 73, "min": 5, "max": 3600, "step": 17}),
                "ref_image_size": (["match", "512", "768", "960"], {"default": "match"}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "ref_images": ("IMAGE",),
                "quantize": (["dit8_te4", "dit8", "te4", "none"], {"default": "dit8_te4"}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "generate"
    CATEGORY = "Fusion-MLX/H3"

    def generate(self, clip, vae, prompt="", width=960, height=544, length=73,
                 ref_image_size="match", audio_vae=None, ref_images=None, quantize="dit8_te4"):
        # H3 latent: z_channels=24, spatial /16, temporal /4. See EmptyMiniMaxH3LatentAV.
        # UPSTREAM GAP: MiniMaxH3Backend.generate() does NOT forward reference_images to
        # generate_video() (no ref2va branch, only image/last_frame fl2va path).
        # _h3_ref_images is staged here but dropped at the engine layer until fusion-mlx
        # adds a ref2va branch (issue -> PR -> dep bump). h3-r2v e2e is BLOCKED until then.
        import mlx.core as mx
        width, height = _apply_768p_override(width, height)
        t_latent = (length - 1) // 4 + 1
        latent = mx.zeros((1, 24, t_latent, height // 16, width // 16), dtype=mx.float32)
        result = {"samples": latent, "num_frames": length, "width": width, "height": height,
                  "_h3_audio": False, "_h3_quantize": quantize}
        if ref_images is not None:
            # ref_images may be a dict {ref_image_N: IMAGE} (r2v.json) or a plain IMAGE
            # tensor (possibly a batch on axis 0). Normalize to a list of /tmp png paths.
            refs = []
            if isinstance(ref_images, dict):
                for k in sorted(ref_images.keys()):
                    refs.append(_save_temp_image(ref_images[k], "h3_r2v_ref"))
            else:
                from fusion_comfyui.core.bridge import to_numpy
                arr = to_numpy(ref_images)
                if arr.ndim == 4:
                    for i in range(arr.shape[0]):
                        refs.append(_save_temp_image(arr[i:i + 1], "h3_r2v_ref"))
                else:
                    refs.append(_save_temp_image(ref_images, "h3_r2v_ref"))
            result["_h3_ref_images"] = refs
        logger.info("MiniMaxH3ReferenceToVideo: %dx%d frames=%d quantize=%s refs=%d",
                     width, height, length, quantize, len(result.get("_h3_ref_images", [])))
        return ({"prompt": prompt}, result)


class VAEDecodeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "Fusion-MLX/H3"

    def decode(self, samples, vae):
        # audio already muxed into the MLX-generated mp4; return a silent dummy so
        # CreateVideo's AUDIO input contract holds. No real decode happens here.
        logger.info("VAEDecodeAudio: passthrough (audio baked in muxed mp4), waveform=1x2 silent")
        return ({"waveform": np.zeros((1, 2), dtype=np.float32), "sample_rate": 24000},)


class CreateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "create"
    CATEGORY = "Fusion-MLX/H3"

    def create(self, images, audio, fps=24.0):
        # frames already decoded from the muxed mp4; forward as-is (audio already in mp4).
        logger.info("CreateVideo: passthrough frames=%s fps=%.1f (audio already in mp4)",
                     getattr(images, "shape", "?"), fps)
        return ({"images": images, "fps": fps, "audio": audio},)


class SaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "h3_t2v"}),
                "format": (["auto", "h264", "h265"], {"default": "auto"}),
                "codec": (["auto", "libx264", "libx265"], {"default": "auto"}),
            }
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Fusion-MLX/H3"

    def save(self, video, filename_prefix="h3_t2v", format="auto", codec="auto"):
        # Re-encode the passthrough IMAGE frames to mp4 via the proven PyAV helper.
        # The MLX engine already produced the final muxed mp4; VAEDecode passed the
        # decoded frames through, so we encode them here into output/ so /history serves
        # the file AICF downloads. Returns {"ui": {"videos": [...]}} (ComfyUI contract).
        import folder_paths
        from fusion_comfyui_plugin.nodes.video_io import FusionSaveVideoNode
        images = video["images"] if isinstance(video, dict) else video
        fps = video.get("fps", 24.0) if isinstance(video, dict) else 24.0
        out_codec = "libx265" if codec == "libx265" else "libx264"
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _prefix = \
            folder_paths.get_save_image_path(filename_prefix, output_dir, images.shape[2], images.shape[1])
        file = f"{filename}_{counter:05d}_.mp4"
        filepath = os.path.join(full_output_folder, file)
        helper = FusionSaveVideoNode()
        helper._encode_video_av(images, filepath, int(fps), out_codec, 18)
        logger.info("SaveVideo(H3): saved %s frames=%s fps=%.1f", filepath, images.shape, fps)
        return {"ui": {"videos": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}


class ImageScaleToTotalPixels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (["lanczos", "nearest-exact", "bilinear", "area"], {"default": "lanczos"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resolution_steps": ("INT", {"default": 32, "min": 1, "max": 1024}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "Fusion-MLX/H3"

    def upscale(self, image, upscale_method="lanczos", megapixels=1.0, resolution_steps=32):
        from PIL import Image as PILImage
        from fusion_comfyui.core.bridge import to_numpy
        arr = to_numpy(image)
        if arr.ndim == 4:
            arr = arr[0]
        h, w = arr.shape[0], arr.shape[1]
        target = int(megapixels * 1024 * 1024)
        scale = (target / (h * w)) ** 0.5
        new_w = max(resolution_steps, int(round(w * scale / resolution_steps)) * resolution_steps)
        new_h = max(resolution_steps, int(round(h * scale / resolution_steps)) * resolution_steps)
        pil = PILImage.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)[:, :, :3])
        method = getattr(PILImage, upscale_method.upper().replace("-", ""), PILImage.LANCZOS)
        pil = pil.resize((new_w, new_h), method)
        out = np.array(pil).astype(np.float32) / 255.0
        logger.info("ImageScaleToTotalPixels: %dx%d -> %dx%d (%.2fMP)", w, h, new_w, new_h, megapixels)
        return (out[np.newaxis, ...],)


class PrimitiveFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"float": ("FLOAT", {"default": 5.0, "min": -1e9, "max": 1e9, "step": 0.01})}}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "emit"
    CATEGORY = "Fusion-MLX/H3"

    def emit(self, float=5.0):
        return (float,)


class ComfyMathExpression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"expression": ("STRING", {"default": "a", "multiline": False})},
            "optional": {"a": ("FLOAT",), "b": ("FLOAT",), "c": ("FLOAT",)},
        }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "eval_expr"
    CATEGORY = "Fusion-MLX/H3"

    def eval_expr(self, expression="a", a=None, b=None, c=None):
        # Sandboxed eval mirroring ComfyUI's native ComfyMath node: __builtins__
        # stripped so __import__/open/exec are unreachable (verified by
        # test_comfy_math_expression_sandbox_no_builtins). Expression input is
        # operator-authored workflow JSON (AICF r2v length calc), not untrusted
        # user text. Only pure math helpers (max/min/round/abs/math) are exposed.
        import math
        env = {"a": a or 0.0, "b": b or 0.0, "c": c or 0.0,
               "max": max, "min": min, "round": round, "abs": abs, "math": math}
        val = float(eval(expression, {"__builtins__": {}}, env))  # noqa: S307 - sandboxed, mirrors ComfyUI ComfyMath; builtins stripped (test_comfy_math_expression_sandbox_no_builtins)
        logger.info("ComfyMathExpression: '%s' = %.4f", expression, val)
        return (val,)
