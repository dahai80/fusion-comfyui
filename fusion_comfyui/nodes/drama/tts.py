import logging
import os

from fusion_comfyui.nodes.base import BaseNode
from fusion_comfyui.core.timer import NodeTimer
from fusion_comfyui.core.lifecycle import FusionMemoryGuardian

logger = logging.getLogger("fusion_comfyui.nodes.drama.tts")


class FusionTTS(BaseNode):
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "fusion-mlx/drama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "model_name": ("STRING", {"default": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"}),
                "voice": ("STRING", {"default": ""}),
                "ref_audio": ("STRING", {"default": ""}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            }
        }

    async def execute(self, text, model_name="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit", voice="", ref_audio="", speed=1.0):
        async with NodeTimer.timed("FusionTTS", "full", text_len=len(text)):
            from fusion_mlx.public_api import TTSEngine

            async with NodeTimer.timed("FusionTTS", "load_engine"):
                engine = TTSEngine(model_name)
                await engine.start()
                logger.info("FusionTTS: engine started %s", model_name)

            kwargs = {"text": text, "speed": speed}
            if voice:
                kwargs["voice"] = voice
            if ref_audio and os.path.exists(ref_audio):
                kwargs["ref_audio"] = ref_audio

            async with NodeTimer.timed("FusionTTS", "synthesize"):
                result = await engine.synthesize(**kwargs)

            output_dir = os.environ.get("FUSION_OUTPUT_DIR", "output")
            os.makedirs(output_dir, exist_ok=True)
            audio_path = os.path.join(output_dir, f"tts_{id(text) % 100000}.wav")
            if isinstance(result, bytes):
                with open(audio_path, "wb") as f:
                    f.write(result)
            elif hasattr(result, "audio"):
                import numpy as np
                audio_data = np.array(result.audio)
                import soundfile as sf
                sf.write(audio_path, audio_data, samplerate=24000)
            else:
                audio_path = ""

            logger.info("FusionTTS: saved %s", audio_path)

            async with NodeTimer.timed("FusionTTS", "unload_engine"):
                await engine.stop()
                FusionMemoryGuardian.purge_memory()

            return (audio_path,)
