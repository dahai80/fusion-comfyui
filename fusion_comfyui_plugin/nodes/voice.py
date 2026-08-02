import asyncio
import logging
import os
import tempfile
import wave

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.voice")

_KNOWN_TTS_MODELS = [
    "mlx-community/kokoro-82m",
    "lucasnewman/f5-tts-mlx",
    "Prince1/kokoro-82m",
    "f5-tts-mlx",
]

_DEFAULT_TTS_MODEL = "mlx-community/kokoro-82m"


def _list_tts_models() -> list:
    models = list(_KNOWN_TTS_MODELS)
    try:
        from fusion_mlx.model_registry import get_registry
        reg = get_registry()
        for name, info in reg.items():
            if info.get("type") == "tts" and name not in models:
                models.append(name)
    except Exception:
        pass
    return models


class FusionVoiceLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": _DEFAULT_TTS_MODEL}),
            },
        }

    RETURN_TYPES = ("FUSION_TTS",)
    RETURN_NAMES = ("tts_engine",)
    FUNCTION = "load"
    CATEGORY = "Fusion-MLX/Voice"

    def load(self, model_name):
        from core.lifecycle import FusionMemoryGuardian

        FusionMemoryGuardian.purge_memory()
        logger.info("FusionVoiceLoader: loading %s", model_name)

        try:
            from fusion_mlx import TTSEngine
            engine = TTSEngine(model_name=model_name)
        except ImportError:
            logger.warning("fusion_mlx.TTSEngine not available, trying mlx_audio direct")
            engine = _MLXAudioTTSEngine(model_name)

        logger.info("FusionVoiceLoader: engine created for %s", model_name)
        return (engine,)


class FusionVoiceSynthesizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_engine": ("FUSION_TTS",),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "voice": ("STRING", {"default": "af_heart"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "ref_audio": ("STRING", {"default": ""}),
                "ref_text": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_path")
    FUNCTION = "synthesize"
    CATEGORY = "Fusion-MLX/Voice"
    OUTPUT_NODE = True

    def synthesize(self, tts_engine, text, voice="af_heart", speed=1.0,
                   ref_audio="", ref_text="", temperature=0.7):
        if not text.strip():
            logger.warning("FusionVoiceSynthesize: empty text, returning silence")
            silence = np.zeros((1, 24000), dtype=np.float32)
            return (silence, "")

        logger.info(
            "FusionVoiceSynthesize: text_len=%d voice=%s speed=%.1f ref_audio=%s",
            len(text), voice, speed, bool(ref_audio),
        )

        ref_audio_val = ref_audio if ref_audio.strip() else None
        ref_text_val = ref_text if ref_text.strip() else None

        try:
            try:
                loop = asyncio.get_event_loop()
                running = loop.is_running()
            except RuntimeError:
                running = False

            if running:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._synthesize_async(
                            tts_engine, text, voice, speed,
                            ref_audio_val, ref_text_val, temperature,
                        ),
                    )
                    pcm_bytes, sample_rate = future.result(timeout=300)
            else:
                pcm_bytes, sample_rate = asyncio.run(
                    self._synthesize_async(
                        tts_engine, text, voice, speed,
                        ref_audio_val, ref_text_val, temperature,
                    )
                )
        except Exception as e:
            logger.error("FusionVoiceSynthesize: failed: %s", e)
            raise

        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        audio_np = audio_np[np.newaxis, :]

        output_path = self._save_wav(pcm_bytes, sample_rate)

        logger.info(
            "FusionVoiceSynthesize: output shape=%s duration=%.1fs path=%s",
            audio_np.shape, audio_np.shape[1] / sample_rate, output_path,
        )
        return (audio_np, output_path)

    async def _synthesize_async(self, tts_engine, text, voice, speed,
                                 ref_audio, ref_text, temperature):
        if hasattr(tts_engine, 'start'):
            await tts_engine.start()

        synthesize_kwargs = {
            "text": text,
            "voice": voice,
            "speed": speed,
        }
        if ref_audio is not None:
            synthesize_kwargs["ref_audio"] = ref_audio
        if ref_text is not None:
            synthesize_kwargs["ref_text"] = ref_text
        if temperature is not None:
            synthesize_kwargs["temperature"] = temperature

        pcm_bytes = await tts_engine.synthesize(**synthesize_kwargs)

        sample_rate = getattr(tts_engine, '_sample_rate', 24000)
        if hasattr(tts_engine, '_model') and tts_engine._model is not None:
            sample_rate = getattr(tts_engine._model, 'sample_rate', sample_rate)

        return pcm_bytes, sample_rate

    def _save_wav(self, pcm_bytes, sample_rate) -> str:
        output_dir = os.path.join(tempfile.gettempdir(), "fusion_comfyui_audio")
        os.makedirs(output_dir, exist_ok=True)

        idx = 0
        while True:
            path = os.path.join(output_dir, f"tts_output_{idx:04d}.wav")
            if not os.path.exists(path):
                break
            idx += 1

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

        return path


class _MLXAudioTTSEngine:
    def __init__(self, model_name):
        self._model_name = model_name
        self._model = None
        self._sample_rate = 24000
        self.started = False

    async def start(self):
        if self._model is not None:
            return
        logger.info("_MLXAudioTTSEngine: loading %s", self._model_name)
        loop = asyncio.get_event_loop()

        def _load():
            from mlx_audio.tts.utils import load_model
            return load_model(self._model_name, strict=False)

        self._model = await loop.run_in_executor(None, _load)
        self._sample_rate = getattr(self._model, "sample_rate", 24000)
        self.started = True
        logger.info("_MLXAudioTTSEngine: loaded %s (sr=%d)", self._model_name, self._sample_rate)

    async def synthesize(self, text, voice="af_heart", speed=1.0,
                          ref_audio=None, ref_text=None, temperature=0.7, **kwargs):
        if self._model is None:
            await self.start()

        loop = asyncio.get_event_loop()

        def _gen():

            gen_kwargs = {
                "text": text,
                "model": self._model,
                "voice": voice,
                "speed": speed,
                "verbose": False,
                "play": False,
                "save": False,
            }
            if ref_audio:
                gen_kwargs["ref_audio"] = ref_audio
            if ref_text:
                gen_kwargs["ref_text"] = ref_text
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            gen_kwargs.update(kwargs)

            import numpy as np

            chunks = []
            for result in self._model.generate(**{k: v for k, v in gen_kwargs.items()
                                                    if k != "text" and k != "model"}):
                a = result.audio
                if hasattr(a, 'astype'):
                    a = np.array(a)
                    if a.dtype != np.float32:
                        a = a.astype(np.float32)
                chunks.append(a)

            if not chunks:
                raise RuntimeError("TTS model produced no audio output")

            audio_np = np.concatenate(chunks, axis=0)
            audio_np = np.clip(audio_np, -1.0, 1.0)
            return (audio_np * 32767).astype(np.int16).tobytes()

        return await loop.run_in_executor(None, _gen)


class FusionVoiceCloneNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_engine": ("FUSION_TTS",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "ref_audio": ("STRING", {"default": ""}),
            },
            "optional": {
                "ref_text": ("STRING", {"default": "", "multiline": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_path")
    FUNCTION = "clone"
    CATEGORY = "Fusion-MLX/Voice"
    OUTPUT_NODE = True

    def clone(self, tts_engine, text, ref_audio, ref_text="", speed=1.0, temperature=0.7):
        if not ref_audio.strip():
            raise ValueError("FusionVoiceClone: ref_audio path is required for voice cloning")
        synth = FusionVoiceSynthesizeNode()
        return synth.synthesize(
            tts_engine, text, voice="af_heart", speed=speed,
            ref_audio=ref_audio, ref_text=ref_text, temperature=temperature,
        )


class FusionSaveAudioNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "FusionAudio"}),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "save"
    CATEGORY = "Fusion-MLX/Voice"
    OUTPUT_NODE = True

    def save(self, audio, filename_prefix="FusionAudio", sample_rate=24000):
        output_dir = os.path.join(tempfile.gettempdir(), "fusion_comfyui_audio")
        os.makedirs(output_dir, exist_ok=True)

        idx = 0
        while True:
            path = os.path.join(output_dir, f"{filename_prefix}_{idx:04d}.wav")
            if not os.path.exists(path):
                break
            idx += 1

        arr = np.array(audio)
        if arr.dtype != np.int16:
            if arr.max() <= 1.0:
                arr = (arr * 32767).astype(np.int16)
            else:
                arr = arr.astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(arr.tobytes())

        logger.info("FusionSaveAudio: saved %s (%d samples)", path, len(arr))
        return (path,)
