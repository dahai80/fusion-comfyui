import asyncio
import logging
import os
import tempfile
import wave

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.voice")

_KNOWN_TTS_MODELS = [
    "mlx-community/Kokoro-82M-bf16",
    "mlx-community/kokoro-82m",
    "lucasnewman/f5-tts-mlx",
    "Prince1/kokoro-82m",
    "f5-tts-mlx",
]

# 默认 TTS 模型: mlx-community/kokoro-82m 在 hf-mirror 404 (repo 不存在),
# 实际可用 repo 是 mlx-community/Kokoro-82M-bf16 (大写 K + -bf16 后缀).
_DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
# 旧默认 id 的别名映射, 老 graph 仍传 kokoro-82m 时自动重定向到可用 repo.
_TTS_MODEL_ALIASES = {
    "mlx-community/kokoro-82m": "mlx-community/Kokoro-82M-bf16",
}


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

        # 旧默认 id (kokoro-82m) 在 hf-mirror 404, 自动重定向到可用 repo (Kokoro-82M-bf16).
        resolved = _TTS_MODEL_ALIASES.get(model_name, model_name)
        if resolved != model_name:
            logger.warning("FusionVoiceLoader: alias %s -> %s", model_name, resolved)

        FusionMemoryGuardian.purge_memory()
        logger.info("FusionVoiceLoader: loading %s", resolved)

        try:
            from fusion_mlx import TTSEngine
            engine = TTSEngine(model_name=resolved)
        except ImportError:
            logger.warning("fusion_mlx.TTSEngine not available, trying mlx_audio direct")
            engine = _MLXAudioTTSEngine(resolved)

        logger.info("FusionVoiceLoader: engine created for %s", resolved)
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

            # kokoro 流式 generate 偶发 [broadcast_shapes] (1,N,1) vs (1,M,9) 对齐错误,
            # 非长度相关 (18 字符成功 / 21 字符失败 / 39 字符有时成功有时失败).
            # 整体重试一次: 失败时重新跑全量生成, 避免单次 flaky 让整条造片链断.
            gen_call_kwargs = {k: v for k, v in gen_kwargs.items()
                               if k != "text" and k != "model"}

            def _run_generate():
                chunks = []
                for result in self._model.generate(**gen_call_kwargs):
                    a = result.audio
                    if hasattr(a, 'astype'):
                        a = np.array(a)
                        if a.dtype != np.float32:
                            a = a.astype(np.float32)
                    chunks.append(a)
                return chunks

            last_err = None
            chunks = []
            for attempt in range(2):
                try:
                    chunks = _run_generate()
                    if chunks:
                        break
                    last_err = RuntimeError("TTS model produced no audio output")
                except Exception as e:
                    last_err = e
                    logger.warning("FusionVoiceSynthesize: generate attempt %d failed: %s", attempt + 1, e)
                    chunks = []

            if not chunks:
                if last_err:
                    raise last_err
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
        # 旧实现存到 tempdir, 不进 /history outputs, 业务层无法通过标准 /history+/view 取回音频.
        # 改为存到 ComfyUI output 目录并返回 ui.audio 结果, 让音频与 images/videos 一样进 /history.
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except Exception:
            output_dir = os.path.join(tempfile.gettempdir(), "fusion_comfyui_audio")
        os.makedirs(output_dir, exist_ok=True)

        idx = 0
        while True:
            filename = f"{filename_prefix}_{idx:04d}.wav"
            path = os.path.join(output_dir, filename)
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
        # ui.audio 让 ComfyUI 把音频文件元数据写入 /history outputs, /view 可下载.
        ui_result = {
            "audio": [{
                "filename": filename,
                "subfolder": "",
                "type": "output",
            }]
        }
        return {"ui": ui_result, "result": (path,)}
