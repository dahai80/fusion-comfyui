import logging
import os

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.talking_head")

_LATENTSYNC_MODEL_DIRS = []


def _get_latentsync_models():
    if _LATENTSYNC_MODEL_DIRS:
        return _LATENTSYNC_MODEL_DIRS
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "latentsync")
        if os.path.isdir(models_dir):
            for d in sorted(os.listdir(models_dir)):
                full = os.path.join(models_dir, d)
                if os.path.isdir(full):
                    _LATENTSYNC_MODEL_DIRS.append(d)
    except Exception:
        pass
    fusion_dir = os.path.expanduser("~/.cache/fusion-mlx/latentsync")
    if os.path.isdir(fusion_dir):
        for d in sorted(os.listdir(fusion_dir)):
            full = os.path.join(fusion_dir, d)
            if os.path.isdir(full) and d not in _LATENTSYNC_MODEL_DIRS:
                _LATENTSYNC_MODEL_DIRS.append(d)
    if not _LATENTSYNC_MODEL_DIRS:
        _LATENTSYNC_MODEL_DIRS.append("latentsync_unet")
    return _LATENTSYNC_MODEL_DIRS


def _resolve_latentsync_path(model_name: str) -> str:
    try:
        import folder_paths
        candidate = os.path.join(folder_paths.models_dir, "latentsync", model_name)
        if os.path.isdir(candidate):
            return candidate
    except Exception:
        pass
    fusion_cache = os.path.expanduser(f"~/.cache/fusion-mlx/latentsync/{model_name}")
    if os.path.isdir(fusion_cache):
        return fusion_cache
    return model_name


def _resolve_audio_path(audio_input) -> str:
    if isinstance(audio_input, str) and audio_input:
        if os.path.isfile(audio_input):
            return audio_input
        try:
            import folder_paths
            annotated = folder_paths.get_annotated_filepath(audio_input)
            if os.path.isfile(annotated):
                return annotated
        except Exception:
            pass
    return audio_input


def _video_frames_to_image_array(video_path: str) -> np.ndarray:
    import av
    container = av.open(video_path)
    try:
        frames = []
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            frames.append(arr)
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return np.stack(frames, axis=0)


class FusionLipsyncLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_latentsync_models(),),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("FUSION_LIPSYNC_MODEL",)
    RETURN_NAMES = ("lipsync_model",)
    FUNCTION = "load_lipsync"
    CATEGORY = "Fusion-MLX/Talking-Head"

    def load_lipsync(self, model_name, dtype="float16"):
        import core.async_utils
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()
        model_path = _resolve_latentsync_path(model_name)
        logger.info("FusionLipsyncLoader: model=%s path=%s dtype=%s", model_name, model_path, dtype)

        dtype_map = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}
        mx_dtype = dtype_map.get(dtype, "float16")

        try:
            pipeline = core.async_utils.run_async(
                self._load_pipeline(model_path, mx_dtype),
                timeout=120,
            )
        except Exception as e:
            logger.error("FusionLipsyncLoader: failed to load LatentSync: %s", e)
            raise

        FusionMemoryGuardian.maybe_purge()
        logger.info("FusionLipsyncLoader: loaded LatentSync pipeline from %s", model_name)
        return (pipeline,)

    async def _load_pipeline(self, model_path, dtype):
        import mlx.core as mx
        mx_dtype = getattr(mx, dtype, mx.float16)
        from fusion_mlx.video.latentsync_mlx.pipeline import LipsyncPipelineMLX
        pipeline = LipsyncPipelineMLX.from_pretrained(model_path, dtype=mx_dtype)
        return pipeline


class FusionLipsyncApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lipsync_model": ("FUSION_LIPSYNC_MODEL",),
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "audio_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "output_fps": ("INT", {"default": 25}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_NODE = True
    FUNCTION = "apply_lipsync"
    CATEGORY = "Fusion-MLX/Talking-Head"

    def apply_lipsync(self, lipsync_model, video_path, audio_path,
                      audio=None, output_fps=25, num_inference_steps=20,
                      guidance_scale=1.0, seed=42):
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.maybe_purge()

        if audio is not None:
            audio_path = self._save_audio_to_temp(audio)

        if not video_path:
            raise ValueError("video_path is required")
        if not audio_path:
            raise ValueError("audio_path or audio input is required")

        audio_path = _resolve_audio_path(audio_path)
        logger.info(
            "FusionLipsyncApply: video=%s audio=%s fps=%d steps=%d seed=%d",
            video_path, audio_path, output_fps, num_inference_steps, seed,
        )

        import core.async_utils

        output_path = None
        try:
            output_path = core.async_utils.run_async(
                self._run_lipsync(
                    lipsync_model, video_path, audio_path,
                    output_fps, num_inference_steps, guidance_scale, seed,
                ),
                timeout=600,
            )
        except Exception as e:
            logger.error("FusionLipsyncApply: lipsync failed: %s", e)
            raise
        finally:
            if audio is not None and audio_path:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

        try:
            frames = _video_frames_to_image_array(output_path)
        finally:
            if output_path:
                try:
                    os.unlink(output_path)
                except OSError:
                    pass

        FusionMemoryGuardian.maybe_purge()
        logger.info("FusionLipsyncApply: output frames shape=%s", frames.shape)
        return (frames,)

    async def _run_lipsync(self, pipeline, video_path, audio_path,
                           output_fps, num_inference_steps, guidance_scale, seed):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        try:
            if hasattr(pipeline, '__call__'):
                result = pipeline(
                    video_path=video_path,
                    audio_path=audio_path,
                    video_out_path=output_path,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
                if hasattr(result, '__await__'):
                    await result
            else:
                raise RuntimeError("LipsyncPipelineMLX has no __call__ method")
        except Exception:
            try:
                os.unlink(output_path)
            except OSError:
                pass
            raise

        return output_path

    @staticmethod
    def _save_audio_to_temp(audio_tuple) -> str:
        import tempfile
        import wave
        audio_np, sample_rate = audio_tuple
        if audio_np.dtype != np.int16:
            audio_np = (audio_np * 32767).astype(np.int16)
        if audio_np.ndim == 2:
            audio_np = audio_np[0]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_np.tobytes())
        return tmp_path
