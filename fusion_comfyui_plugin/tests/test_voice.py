import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestFusionVoiceLoaderNode:
    def test_input_types(self):
        from nodes.voice import FusionVoiceLoaderNode
        inputs = FusionVoiceLoaderNode.INPUT_TYPES()
        assert "required" in inputs
        assert "model_name" in inputs["required"]

    def test_return_types(self):
        from nodes.voice import FusionVoiceLoaderNode
        assert FusionVoiceLoaderNode.RETURN_TYPES == ("FUSION_TTS",)
        assert FusionVoiceLoaderNode.FUNCTION == "load"

    def test_load_creates_engine(self):
        from nodes.voice import FusionVoiceLoaderNode
        mock_engine = MagicMock()
        with patch("nodes.voice.TTSEngine", return_value=mock_engine, create=True), \
             patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.purge_memory"):
            node = FusionVoiceLoaderNode()
            result = node.load("mlx-community/kokoro-82m")
            assert len(result) == 1


class TestFusionVoiceSynthesizeNode:
    def test_input_types(self):
        from nodes.voice import FusionVoiceSynthesizeNode
        inputs = FusionVoiceSynthesizeNode.INPUT_TYPES()
        assert "required" in inputs
        assert "tts_engine" in inputs["required"]
        assert "text" in inputs["required"]
        assert "optional" in inputs
        assert "voice" in inputs["optional"]
        assert "ref_audio" in inputs["optional"]

    def test_return_types(self):
        from nodes.voice import FusionVoiceSynthesizeNode
        assert "AUDIO" in FusionVoiceSynthesizeNode.RETURN_TYPES
        assert FusionVoiceSynthesizeNode.OUTPUT_NODE is True

    def test_empty_text(self):
        from nodes.voice import FusionVoiceSynthesizeNode
        node = FusionVoiceSynthesizeNode()
        result = node.synthesize(MagicMock(), "   ")
        assert len(result) == 2
        audio_np, path = result
        assert audio_np.shape[0] == 1

    def test_synthesize_success(self):
        from nodes.voice import FusionVoiceSynthesizeNode
        pcm_data = np.zeros(24000, dtype=np.int16).tobytes()

        node = FusionVoiceSynthesizeNode()
        with patch.object(node, '_synthesize_async', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = (pcm_data, 24000)
            result = node.synthesize(MagicMock(), "Hello world")

        assert len(result) == 2
        audio_np, path = result
        assert audio_np.shape[0] == 1
        assert audio_np.shape[1] == 24000

    def test_save_wav(self):
        from nodes.voice import FusionVoiceSynthesizeNode
        node = FusionVoiceSynthesizeNode()
        pcm_data = np.zeros(48000, dtype=np.int16).tobytes()
        path = node._save_wav(pcm_data, 24000)
        assert path.endswith(".wav")
        import os
        assert os.path.exists(path)
        os.unlink(path)

    def test_synthesize_with_ref_audio(self):
        from nodes.voice import FusionVoiceSynthesizeNode
        pcm_data = np.zeros(24000, dtype=np.int16).tobytes()
        node = FusionVoiceSynthesizeNode()
        with patch.object(node, '_synthesize_async', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = (pcm_data, 24000)
            _result = node.synthesize(
                MagicMock(), "Hello", ref_audio="/tmp/ref.wav", ref_text="ref text",
            )
        mock_async.assert_called_once()


class TestFusionVoiceCloneNode:
    def test_input_types(self):
        from nodes.voice import FusionVoiceCloneNode
        inputs = FusionVoiceCloneNode.INPUT_TYPES()
        assert "tts_engine" in inputs["required"]
        assert "ref_audio" in inputs["required"]

    def test_clone_no_ref_audio_raises(self):
        from nodes.voice import FusionVoiceCloneNode
        node = FusionVoiceCloneNode()
        with pytest.raises(ValueError, match="ref_audio"):
            node.clone(MagicMock(), "Hello", "")

    def test_clone_delegates_to_synthesize(self):
        from nodes.voice import FusionVoiceCloneNode
        node = FusionVoiceCloneNode()
        with patch("nodes.voice.FusionVoiceSynthesizeNode.synthesize") as mock_synth:
            mock_synth.return_value = (np.zeros((1, 24000), dtype=np.float32), "/tmp/test.wav")
            node.clone(MagicMock(), "Hello", "/tmp/ref.wav", ref_text="ref")
            mock_synth.assert_called_once()


class TestFusionSaveAudioNode:
    def test_input_types(self):
        from nodes.voice import FusionSaveAudioNode
        inputs = FusionSaveAudioNode.INPUT_TYPES()
        assert "audio" in inputs["required"]
        assert "filename_prefix" in inputs["required"]

    def test_save_float_audio(self):
        from nodes.voice import FusionSaveAudioNode
        audio = np.random.randn(1, 24000).astype(np.float32) * 0.5
        node = FusionSaveAudioNode()
        result = node.save(audio, "test_audio", 24000)
        path = result["result"][0]
        assert isinstance(result["ui"]["audio"], list)
        assert result["ui"]["audio"][0]["filename"].endswith(".wav")
        import os
        assert os.path.exists(path)
        os.unlink(path)

    def test_save_int16_audio(self):
        from nodes.voice import FusionSaveAudioNode
        audio = np.zeros((1, 24000), dtype=np.int16)
        node = FusionSaveAudioNode()
        result = node.save(audio, "test_int16", 24000)
        path = result["result"][0]
        assert isinstance(result["ui"]["audio"], list)
        import os
        assert os.path.exists(path)
        os.unlink(path)


class TestMLXAudioTTSEngine:
    def test_init(self):
        from nodes.voice import _MLXAudioTTSEngine
        engine = _MLXAudioTTSEngine("test-model")
        assert engine._model_name == "test-model"
        assert engine._model is None
        assert engine.started is False

    @pytest.mark.asyncio
    async def test_start(self):
        from nodes.voice import _MLXAudioTTSEngine
        mock_model = MagicMock()
        mock_model.sample_rate = 22050
        with patch("nodes.voice._MLXAudioTTSEngine.start") as _mock_start:
            engine = _MLXAudioTTSEngine("test-model")
            engine._model = mock_model
            engine._sample_rate = 22050
            engine.started = True
            assert engine._model is mock_model
            assert engine._sample_rate == 22050

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        from nodes.voice import _MLXAudioTTSEngine
        engine = _MLXAudioTTSEngine("test-model")
        engine._model = MagicMock()
        engine.started = True
        await engine.start()
        assert engine.started is True


class TestListTTSModels:
    def test_includes_known(self):
        from nodes.voice import _list_tts_models
        models = _list_tts_models()
        assert "mlx-community/Kokoro-82M-bf16" in models
        assert "mlx-community/kokoro-82m" in models
        assert "lucasnewman/f5-tts-mlx" in models

    def test_default_is_working_repo(self):
        from nodes.voice import _DEFAULT_TTS_MODEL
        assert _DEFAULT_TTS_MODEL == "mlx-community/Kokoro-82M-bf16"

    def test_alias_redirects_old_default(self):
        from nodes.voice import _TTS_MODEL_ALIASES
        assert _TTS_MODEL_ALIASES["mlx-community/kokoro-82m"] == "mlx-community/Kokoro-82M-bf16"


class TestFusionVoiceSynthesizeHttpBackend:
    def test_input_types_has_http_backend(self):
        from nodes.voice import FusionVoiceSynthesizeNode
        inputs = FusionVoiceSynthesizeNode.INPUT_TYPES()
        assert "backend" in inputs["optional"]
        assert "http" in inputs["optional"]["backend"][0]
        assert "local" in inputs["optional"]["backend"][0]
        assert "model_name" in inputs["optional"]

    def test_http_success(self, monkeypatch, tmp_path):
        import io
        import wave
        from nodes.voice import FusionVoiceSynthesizeNode

        monkeypatch.setenv("FUSION_MLX_API_KEY", "test-key")
        monkeypatch.setenv("FUSION_MLX_BASE", "http://127.0.0.1:11434")

        pcm = np.zeros(24000, dtype=np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)
        wav_bytes = buf.getvalue()

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.content = wav_bytes
        fake_resp.text = ""

        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.post = MagicMock(return_value=fake_resp)

        node = FusionVoiceSynthesizeNode()
        with patch("httpx.Client", return_value=fake_client) as mock_ctor:
            result = node.synthesize(
                MagicMock(), "如果把太阳压缩成豌豆",
                voice="Cherry", backend="http",
                model_name="Qwen3-TTS-12Hz-1.7B-Base-8bit",
            )

        assert mock_ctor.called
        call_kwargs = fake_client.post.call_args
        assert call_kwargs[0][0].endswith("/v1/audio/speech")
        body = call_kwargs[1]["json"]
        assert body["model"] == "Qwen3-TTS-12Hz-1.7B-Base-8bit"
        assert body["voice"] == "Cherry"
        assert body["response_format"] == "wav"

        assert len(result) == 2
        audio_np, path = result
        assert audio_np.shape[0] == 1
        assert path.endswith(".wav")
        import os
        assert os.path.exists(path)
        os.unlink(path)

    def test_http_error_raises(self, monkeypatch):
        from nodes.voice import FusionVoiceSynthesizeNode

        monkeypatch.setenv("FUSION_MLX_API_KEY", "test-key")
        monkeypatch.setenv("FUSION_MLX_BASE", "http://127.0.0.1:11434")

        fake_resp = MagicMock()
        fake_resp.status_code = 404
        fake_resp.text = "Model not found"

        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.post = MagicMock(return_value=fake_resp)

        node = FusionVoiceSynthesizeNode()
        with patch("httpx.Client", return_value=fake_client):
            with pytest.raises(RuntimeError, match="404"):
                node.synthesize(
                    MagicMock(), "测试文本",
                    backend="http", model_name="bad-model",
                )

    def test_http_retry_500_then_success(self, monkeypatch):
        import io
        import wave
        from nodes.voice import FusionVoiceSynthesizeNode

        monkeypatch.setenv("FUSION_MLX_API_KEY", "test-key")
        monkeypatch.setenv("FUSION_MLX_BASE", "http://127.0.0.1:11434")

        pcm = np.zeros(24000, dtype=np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)
        wav_bytes = buf.getvalue()

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Internal server error"
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.content = wav_bytes
        resp_200.text = ""

        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.post = MagicMock(side_effect=[resp_500, resp_200])

        node = FusionVoiceSynthesizeNode()
        with patch("httpx.Client", return_value=fake_client):
            with patch("time.sleep") as _sleep:
                result = node.synthesize(
                    MagicMock(), "如果把太阳压缩成豌豆",
                    voice="Cherry", backend="http",
                    model_name="Qwen3-TTS-12Hz-1.7B-Base-8bit",
                )

        assert fake_client.post.call_count == 2
        assert len(result) == 2
        import os
        os.unlink(result[1])

    def test_http_retry_exhausts_on_persistent_500(self, monkeypatch):
        from nodes.voice import FusionVoiceSynthesizeNode

        monkeypatch.setenv("FUSION_MLX_API_KEY", "test-key")
        monkeypatch.setenv("FUSION_MLX_BASE", "http://127.0.0.1:11434")

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Internal server error"

        fake_client = MagicMock()
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        fake_client.post = MagicMock(return_value=resp_500)

        node = FusionVoiceSynthesizeNode()
        with patch("httpx.Client", return_value=fake_client):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="500"):
                    node.synthesize(
                        MagicMock(), "测试文本",
                        backend="http", model_name="Qwen3-TTS-12Hz-1.7B-Base-8bit",
                    )
        assert fake_client.post.call_count == 3
