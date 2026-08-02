import json
import logging
import os
import subprocess
from fractions import Fraction

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.video_io")


class SaveWEBM:
    """Override native SaveWEBM — uses PyAV instead of torch for zero-PyTorch mode."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "codec": ("COMBO", {"options": ["vp9", "av1"]}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "crf": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 63.0, "step": 1.0}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_video"
    CATEGORY = "video"

    def save_video(self, images, filename_prefix="ComfyUI", codec="vp9",
                   fps=24.0, crf=32.0, prompt=None, extra_pnginfo=None):
        import av
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix, output_dir, images.shape[2], images.shape[1]
            )

        file = f"{filename}_{counter:05d}_.webm"
        filepath = os.path.join(full_output_folder, file)

        container = av.open(filepath, mode="w")
        if prompt is not None:
            container.metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                container.metadata[x] = json.dumps(extra_pnginfo[x])

        save_alpha = images.shape[-1] == 4 and codec == "vp9"
        codec_map = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}
        stream = container.add_stream(
            codec_map[codec], rate=Fraction(round(fps * 1000), 1000)
        )
        stream.width = images.shape[2]
        stream.height = images.shape[1]
        stream.pix_fmt = "yuva420p" if save_alpha else ("yuv420p10le" if codec == "av1" else "yuv420p")
        stream.bit_rate = 0
        stream.options = {"crf": str(int(crf))}
        if codec == "av1":
            stream.options["svt"] = "1"
            stream.options["preset"] = "6"

        img_np = np.array(images)
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        for frame_data in img_np:
            frame = av.VideoFrame.from_ndarray(frame_data[:, :, :3], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()

        logger.info("SaveWEBM: saved %s codec=%s fps=%.1f crf=%d", filepath, codec, fps, int(crf))
        return {"ui": {"videos": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}


class SaveAnimatedWEBP:
    """Override native SaveAnimatedWEBP — uses PIL instead of torch."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "fps": ("FLOAT", {"default": 6.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "lossless": ("BOOLEAN", {"default": True}),
                "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                "method": ("COMBO", {"options": ["default", "fastest", "slowest"]}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_webp"
    CATEGORY = "image/animation"

    def save_webp(self, images, filename_prefix="ComfyUI", fps=6.0,
                  lossless=True, quality=80, method="default",
                  prompt=None, extra_pnginfo=None):
        from PIL import Image as PILImage
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix, output_dir, images.shape[2], images.shape[1]
            )

        file = f"{filename}_{counter:05d}_.webp"
        filepath = os.path.join(full_output_folder, file)

        img_np = np.array(images)
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        pil_images = []
        for frame_data in img_np:
            pil = PILImage.fromarray(frame_data[:, :, :3])
            pil_images.append(pil)

        if pil_images:
            fps_val = float(fps) if not isinstance(fps, (int, float)) else fps
            duration_ms = int(1000.0 / fps_val)
            method_val = method
            if isinstance(method_val, str):
                method_map = {"default": 0, "fast": 1, "slow": 6}
                method_val = method_map.get(method_val, 0)
            pil_images[0].save(
                filepath,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration_ms,
                loop=0,
                lossless=bool(lossless),
                quality=int(quality),
                method=method_val,
            )

        logger.info("SaveAnimatedWEBP: saved %s fps=%.1f frames=%d", filepath, fps, len(pil_images))
        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}


class FusionSaveVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "FusionVideo"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "codec": (["libx264", "libvpx-vp9", "libx265"], {"default": "libx264"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
            },
            "optional": {
                "audio_file": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_video"
    CATEGORY = "Fusion-MLX/Video"

    def save_video(self, images, filename_prefix="FusionVideo", fps=24,
                   codec="libx264", crf=18, audio_file=""):

        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        ext = "mp4" if codec in ("libx264", "libx265") else "webm"
        output_path = self._unique_path(output_dir, filename_prefix, ext)
        self._encode_video_av(images, output_path, fps, codec, crf)

        if audio_file and os.path.exists(audio_file):
            self._mux_audio(output_path, audio_file)

        logger.info("FusionSaveVideo: saved %s codec=%s fps=%d", output_path, codec, fps)
        return {"ui": {"videos": [{"filename": os.path.basename(output_path), "subfolder": ""}]}}

    def _get_output_dir(self):
        try:
            import folder_paths
            return folder_paths.get_output_directory()
        except Exception:
            return os.path.join(os.getcwd(), "output")

    def _unique_path(self, output_dir, prefix, ext):
        base = os.path.join(output_dir, prefix)
        path = f"{base}.{ext}"
        counter = 1
        while os.path.exists(path):
            path = f"{base}_{counter:04d}.{ext}"
            counter += 1
        return path

    def _write_frames(self, images, frames_dir):
        from core.bridge import to_numpy
        from PIL import Image as PILImage

        arr = to_numpy(images)

        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        if arr.ndim == 5:
            B, T, H, W, C = arr.shape
            arr = arr.reshape(B * T, H, W, C)
        elif arr.ndim == 3:
            arr = arr[np.newaxis, ...]

        for i, frame in enumerate(arr):
            img = PILImage.fromarray(frame[:, :, :3])
            img.save(os.path.join(frames_dir, f"frame_{i:06d}.png"))

        logger.info("FusionSaveVideo: wrote %d frames", len(arr))

    def _encode_video_av(self, images, output_path, fps, codec, crf):
        import av
        from core.bridge import to_numpy

        arr = to_numpy(images)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        if arr.ndim == 5:
            B, T, H, W, C = arr.shape
            arr = arr.reshape(B * T, H, W, C)
        elif arr.ndim == 3:
            arr = arr[np.newaxis, ...]

        h, w = arr.shape[1], arr.shape[2]
        codec_name = "libx264" if codec in ("libx264", "libx265") else "libvpx-vp9"
        if codec == "libx265":
            codec_name = "libx265"

        container = av.open(output_path, "w")
        try:
            stream = container.add_stream(codec_name, rate=fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf)}
            if codec_name == "libx264":
                stream.options["movflags"] = "+faststart"

            for frame_arr in arr:
                frame = av.VideoFrame.from_ndarray(frame_arr[:, :, :3], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

        logger.info("FusionSaveVideo: encoded %d frames via av to %s", len(arr), output_path)

    def _mux_audio(self, video_path, audio_file):
        tmp_path = video_path + ".tmp.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac",
            "-movflags", "+faststart",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            os.replace(tmp_path, video_path)
        else:
            logger.warning("FusionSaveVideo: audio mux failed: %s", result.stderr[-200:])
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _encode_video(self, frames_dir, output_path, fps, codec, crf, audio_file):
        input_pattern = os.path.join(frames_dir, "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
        ]
        if audio_file and os.path.exists(audio_file):
            cmd.extend(["-i", audio_file])

        cmd.extend([
            "-c:v", codec,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ])
        if codec == "libvpx-vp9":
            cmd.extend(["-b:v", "0", "-row-mt", "1"])

        cmd.append(output_path)

        logger.info("FusionSaveVideo: ffmpeg cmd=%s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error("FusionSaveVideo: ffmpeg stderr=%s", result.stderr[-500:])
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-200:]}")


class FusionVideoConcatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("IMAGE",),
                "video_b": ("IMAGE",),
            },
            "optional": {
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    FUNCTION = "concat"
    CATEGORY = "Fusion-MLX/Video"

    def concat(self, video_a, video_b, fps=24):
        from core.bridge import to_numpy

        arr_a = to_numpy(video_a).astype(np.float32)
        arr_b = to_numpy(video_b).astype(np.float32)

        if arr_a.max() > 1.0:
            arr_a = arr_a / 255.0
        if arr_b.max() > 1.0:
            arr_b = arr_b / 255.0

        if arr_a.ndim == 5:
            B, T, H, W, C = arr_a.shape
            arr_a = arr_a.reshape(B * T, H, W, C)
        if arr_b.ndim == 5:
            B, T, H, W, C = arr_b.shape
            arr_b = arr_b.reshape(B * T, H, W, C)

        if arr_a.shape[1:] != arr_b.shape[1:]:
            logger.warning(
                "FusionVideoConcat: shape mismatch a=%s b=%s, resizing b to match a",
                arr_a.shape, arr_b.shape,
            )
            from PIL import Image as PILImage
            target_h, target_w = arr_a.shape[1], arr_a.shape[2]
            resized = []
            for frame in arr_b:
                pil = PILImage.fromarray((frame * 255).astype(np.uint8)[:, :, :3])
                pil = pil.resize((target_w, target_h), PILImage.LANCZOS)
                resized.append(np.array(pil).astype(np.float32) / 255.0)
            arr_b = np.stack(resized, axis=0)

        merged = np.concatenate([arr_a, arr_b], axis=0)

        logger.info("FusionVideoConcat: a=%s + b=%s -> merged=%s",
                     arr_a.shape, arr_b.shape, merged.shape)
        return (merged,)
