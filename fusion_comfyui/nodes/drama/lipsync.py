import logging
import os

from fusion_comfyui.nodes.base import BaseNode
from fusion_comfyui.core.timer import NodeTimer
from fusion_comfyui.core.lifecycle import FusionMemoryGuardian

logger = logging.getLogger("fusion_comfyui.nodes.drama.lipsync")


class FusionLipSync(BaseNode):
    RETURN_TYPES = ("VIDEO_PATH",)
    CATEGORY = "fusion-mlx/drama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "audio_path": ("STRING", {"default": ""}),
                "method": (["latentsync", "musetalk"], {"default": "latentsync"}),
                "model_dir": ("STRING", {"default": ""}),
                "num_frames": ("INT", {"default": 16, "min": 1, "max": 64}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 1247}),
            }
        }

    async def execute(
        self, video_path, audio_path, method="latentsync", model_dir="",
        num_frames=16, num_inference_steps=20, guidance_scale=1.5, seed=1247,
    ):
        async with NodeTimer.timed("FusionLipSync", "full", method=method):
            output_dir = os.environ.get("FUSION_OUTPUT_DIR", "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"lipsync_{seed}.mp4")

            if method == "latentsync":
                output_path = await self._run_latentsync(
                    video_path, audio_path, output_path, model_dir,
                    num_frames, num_inference_steps, guidance_scale, seed,
                )
            else:
                output_path = await self._run_musetalk(
                    video_path, audio_path, output_path, model_dir,
                )

            logger.info("FusionLipSync: output=%s", output_path)
            return (output_path,)

    async def _run_latentsync(
        self, video_path, audio_path, output_path, model_dir,
        num_frames, num_inference_steps, guidance_scale, seed,
    ):
        from fusion_mlx.video.latentsync_mlx import LipsyncPipelineMLX

        async with NodeTimer.timed("FusionLipSync", "load_latentsync"):
            pipeline = LipsyncPipelineMLX.from_pretrained(model_dir)
            logger.info("FusionLipSync: LatentSync loaded from %s", model_dir)

        async with NodeTimer.timed("FusionLipSync", "latentsync_run"):
            pipeline(
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=output_path,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

        async with NodeTimer.timed("FusionLipSync", "unload_latentsync"):
            del pipeline
            FusionMemoryGuardian.purge_memory()

        return output_path

    async def _run_musetalk(self, video_path, audio_path, output_path, model_dir):
        import subprocess

        from fusion_mlx.video.musetalk_mlx import MuseTalkPipeline

        async with NodeTimer.timed("FusionLipSync", "load_musetalk"):
            pipeline = MuseTalkPipeline.from_pretrained_mlx(model_dir) if model_dir else MuseTalkPipeline.from_pretrained("")
            logger.info("FusionLipSync: MuseTalk loaded")

        async with NodeTimer.timed("FusionLipSync", "musetalk_encode_audio"):
            audio_chunks = pipeline.encode_audio_from_wav(audio_path)

        async with NodeTimer.timed("FusionLipSync", "musetalk_extract_faces"):
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames_bgr = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_bgr.append(frame)
            cap.release()

            latent_stack = []
            for bgr in frames_bgr:
                latent = pipeline.get_latents_for_unet(bgr)
                latent_stack.append(latent)

        async with NodeTimer.timed("FusionLipSync", "musetalk_run"):
            faces = pipeline.run_batched(latent_stack, audio_chunks)

        async with NodeTimer.timed("FusionLipSync", "musetalk_compose"):
            fps = 25
            tmp_raw = output_path.replace(".mp4", "_raw.mp4")
            h, w = frames_bgr[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_raw, fourcc, fps, (w, h))
            for i, face in enumerate(faces):
                if i < len(frames_bgr):
                    frame = frames_bgr[i].copy()
                    fh, fw = face.shape[:2]
                    frame[:fh, :fw] = face
                    writer.write(frame)
            writer.release()
            cmd = [
                "ffmpeg", "-y", "-i", tmp_raw, "-i", audio_path,
                "-c:v", "libx264", "-c:a", "aac", "-shortest", output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            if os.path.exists(tmp_raw):
                os.unlink(tmp_raw)

        async with NodeTimer.timed("FusionLipSync", "unload_musetalk"):
            del pipeline
            FusionMemoryGuardian.purge_memory()

        return output_path
