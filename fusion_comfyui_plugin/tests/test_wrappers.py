import pytest
from unittest.mock import MagicMock, patch


class TestFusionModelWrapper:
    def test_init(self):
        from core.wrappers import FusionModelWrapper
        w = FusionModelWrapper(model_path="/tmp/m", model_name="test", model_type="video")
        assert w.model_name == "test"
        assert w.model_type == "video"
        assert w.patcher is None
        assert w.memory_required() == 0

    def test_repr(self):
        from core.wrappers import FusionModelWrapper
        w = FusionModelWrapper(model_path="/tmp/m", model_name="test", model_type="image")
        r = repr(w)
        assert "test" in r
        assert "image" in r

    def test_get_engine_creates(self):
        from core.wrappers import FusionModelWrapper
        with patch("core.engine_wrapper.FusionEngineWrapper") as MockEngine:
            inst = MockEngine.return_value
            w = FusionModelWrapper(model_path="/tmp/m", model_name="test", model_type="video")
            eng = w.get_engine()
            assert eng is inst

    def test_get_engine_cached(self):
        from core.wrappers import FusionModelWrapper
        with patch("core.engine_wrapper.FusionEngineWrapper") as _MockEngine:
            w = FusionModelWrapper(model_path="/tmp/m", model_name="test", model_type="video")
            e1 = w.get_engine()
            e2 = w.get_engine()
            assert e1 is e2

    def test_load_failure_raises(self):
        from core.wrappers import FusionModelWrapper
        with patch("core.engine_wrapper.FusionEngineWrapper") as MockEngine:
            MockEngine.side_effect = RuntimeError("fail")
            w = FusionModelWrapper(model_path="/tmp/m", model_name="test", model_type="video")
            with pytest.raises(RuntimeError, match="fail"):
                w.get_engine()


class TestFusionCLIPWrapper:
    def test_init(self):
        from core.wrappers import FusionCLIPWrapper
        w = FusionCLIPWrapper(model_path="/tmp/m", model_name="test", clip_type="wan")
        assert w.clip_type == "wan"
        assert w.patcher is None
        assert w.tokenizer is None

    def test_tokenize(self):
        from core.wrappers import FusionCLIPWrapper
        w = FusionCLIPWrapper(model_path="/tmp/m", model_name="test", clip_type="wan")
        result = w.tokenize("hello")
        assert result == {"text": "hello"}

    def test_encode_from_tokens_scheduled(self):
        from core.wrappers import FusionCLIPWrapper
        w = FusionCLIPWrapper(model_path="/tmp/m", model_name="test", clip_type="wan")
        result = w.encode_from_tokens_scheduled({"text": "hello"})
        assert result["text"] == "hello"
        assert result["clip"] is w

    def test_repr(self):
        from core.wrappers import FusionCLIPWrapper
        w = FusionCLIPWrapper(model_path="/tmp/m", model_name="test", clip_type="wan")
        assert "test" in repr(w)
        assert "wan" in repr(w)


class TestFusionVAEWrapper:
    def test_init(self):
        from core.wrappers import FusionVAEWrapper
        w = FusionVAEWrapper(model_path="/tmp/m", model_name="test")
        assert w.model_name == "test"
        assert w.first_stage_model is None

    def test_repr(self):
        from core.wrappers import FusionVAEWrapper
        w = FusionVAEWrapper(model_path="/tmp/m", model_name="test")
        assert "test" in repr(w)


class TestFusionConditioning:
    def test_with_embed(self):
        from core.wrappers import FusionConditioning
        embed = MagicMock()
        embed.shape = (1, 77, 768)
        c = FusionConditioning({"embed": embed})
        assert "embed_shape" in repr(c)

    def test_with_prompt(self):
        from core.wrappers import FusionConditioning
        c = FusionConditioning({"prompt": "a beautiful scene"})
        assert "prompt" in repr(c)


class TestInferModelType:
    def test_video(self):
        from core.wrappers import _infer_model_type
        assert _infer_model_type("Wan2.2-5B") == "video"
        assert _infer_model_type("ltx-video") == "video"
        assert _infer_model_type("skyreels-v3") == "video"
        assert _infer_model_type("skyreels-v3-a2v") == "video"
        assert _infer_model_type("skyreels-v3-r2v") == "video"
        assert _infer_model_type("skyreels-v3-v2v") == "video"
        assert _infer_model_type("ltx-2.3-mlx-q8") == "video"

    def test_image(self):
        from core.wrappers import _infer_model_type
        assert _infer_model_type("FLUX.2-dev") == "image"
        assert _infer_model_type("unknown") == "image"


class TestMapCheckpoint:
    def test_ltx(self):
        from core.wrappers import _map_checkpoint_to_model_name
        assert _map_checkpoint_to_model_name("ltx-video.safetensors") == "LTX-Video"

    def test_wan22(self):
        from core.wrappers import _map_checkpoint_to_model_name
        assert _map_checkpoint_to_model_name("wan2.2-5b.safetensors") == "Wan2.2-5B"

    def test_wan(self):
        from core.wrappers import _map_checkpoint_to_model_name
        assert _map_checkpoint_to_model_name("wan-1.3b.safetensors") == "Wan2.2-5B"

    def test_flux_4b(self):
        from core.wrappers import _map_checkpoint_to_model_name
        assert _map_checkpoint_to_model_name("flux-4b.safetensors") == "FLUX.2-klein-base-4B"

    def test_flux_9b(self):
        from core.wrappers import _map_checkpoint_to_model_name
        assert _map_checkpoint_to_model_name("flux-9b.safetensors") == "FLUX.2-klein-9b"

    def test_cosmos(self):
        from core.wrappers import _map_checkpoint_to_model_name
        assert _map_checkpoint_to_model_name("cosmos-7b.safetensors") == "Cosmos-7B"

    def test_hunyuan(self):
        from core.wrappers import _map_checkpoint_to_model_name
        assert _map_checkpoint_to_model_name("hunyuan-video.safetensors") == "HunyuanVideo"

    def test_unknown(self):
        from core.wrappers import _map_checkpoint_to_model_name
        assert _map_checkpoint_to_model_name("custom-model.safetensors") == "custom-model.safetensors"


class TestMapClipType:
    def test_wan(self):
        from core.wrappers import _map_clip_type_to_model_name
        assert _map_clip_type_to_model_name("wan", "x") == "Wan2.2-5B"

    def test_ltxv(self):
        from core.wrappers import _map_clip_type_to_model_name
        assert _map_clip_type_to_model_name("ltxv", "x") == "LTX-Video"

    def test_cosmos(self):
        from core.wrappers import _map_clip_type_to_model_name
        assert _map_clip_type_to_model_name("cosmos", "x") == "Cosmos-7B"

    def test_hunyuan_image(self):
        from core.wrappers import _map_clip_type_to_model_name
        assert _map_clip_type_to_model_name("hunyuan_image", "x") == "HunyuanVideo"

    def test_flux2(self):
        from core.wrappers import _map_clip_type_to_model_name
        assert _map_clip_type_to_model_name("flux2", "x") == "FLUX.2-klein-base-4B"

    def test_unknown(self):
        from core.wrappers import _map_clip_type_to_model_name
        assert _map_clip_type_to_model_name("custom", "custom-name") == "custom-name"


class TestMapUnetName:
    def test_wan22_14b(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="Wan2.2-14B"):
            result = _map_unet_name_to_model_name("wan2.2-14b")
            assert result == "Wan2.2-14B"

    def test_wan22_5b(self):
        from core.wrappers import _map_unet_name_to_model_name
        result = _map_unet_name_to_model_name("wan2.2-5b")
        assert result == "Wan2.2-5B"

    def test_wan21_14b(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="Wan2.1-14B"):
            result = _map_unet_name_to_model_name("wan2.1-14b")
            assert result == "Wan2.1-14B"

    def test_wan21_1b(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="Wan2.1-1.3B"):
            result = _map_unet_name_to_model_name("wan2.1-1.3b")
            assert result == "Wan2.1-1.3B"

    def test_wan_generic(self):
        from core.wrappers import _map_unet_name_to_model_name
        result = _map_unet_name_to_model_name("wan-something")
        assert result == "Wan2.2-5B"

    def test_ltx(self):
        from core.wrappers import _map_unet_name_to_model_name
        result = _map_unet_name_to_model_name("ltx-video")
        assert result == "LTX-Video"

    def test_skyreels_default(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="SkyReels-V3-V2V-14B-MLX"):
            result = _map_unet_name_to_model_name("skyreels-v3")
            assert result == "SkyReels-V3-V2V-14B-MLX"

    def test_skyreels_a2v(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="SkyReels-V3-A2V-19B-MLX"):
            result = _map_unet_name_to_model_name("skyreels-v3-a2v")
            assert result == "SkyReels-V3-A2V-19B-MLX"

    def test_skyreels_a2v_19b(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="SkyReels-V3-A2V-19B-MLX"):
            result = _map_unet_name_to_model_name("skyreels-v3-19b")
            assert result == "SkyReels-V3-A2V-19B-MLX"

    def test_skyreels_r2v(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="SkyReels-V3-R2V-14B-MLX"):
            result = _map_unet_name_to_model_name("skyreels-v3-r2v")
            assert result == "SkyReels-V3-R2V-14B-MLX"

    def test_skyreels_v2v(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="SkyReels-V3-V2V-14B-MLX"):
            result = _map_unet_name_to_model_name("skyreels-v3-v2v")
            assert result == "SkyReels-V3-V2V-14B-MLX"

    def test_wan22_ti2v(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="Wan2.2-TI2V-5B-mlx-q8"):
            result = _map_unet_name_to_model_name("wan2.2-ti2v-5b")
            assert result == "Wan2.2-TI2V-5B-mlx-q8"

    def test_ltx2(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="ltx-2.3-mlx-q8"):
            result = _map_unet_name_to_model_name("ltx-2")
            assert result == "ltx-2.3-mlx-q8"

    def test_ltx_2_3(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="ltx-2.3-mlx-q8"):
            result = _map_unet_name_to_model_name("ltx-2.3")
            assert result == "ltx-2.3-mlx-q8"

    def test_ltx_2_underscore(self):
        from core.wrappers import _map_unet_name_to_model_name
        with patch("core.wrappers._fallback_model", return_value="ltx-2.3-mlx-q8"):
            result = _map_unet_name_to_model_name("ltx_2_something")
            assert result == "ltx-2.3-mlx-q8"

    def test_cosmos_predict2(self):
        from core.wrappers import _map_unet_name_to_model_name
        result = _map_unet_name_to_model_name("cosmos-predict2-2b")
        assert result == "Cosmos-Predict2"

    def test_flux_4b(self):
        from core.wrappers import _map_unet_name_to_model_name
        result = _map_unet_name_to_model_name("flux-base-4b")
        assert result == "FLUX.2-klein-base-4B"

    def test_flux_klein(self):
        from core.wrappers import _map_unet_name_to_model_name
        result = _map_unet_name_to_model_name("flux-klein")
        assert result == "FLUX.2-klein-9b"

    def test_cosmos(self):
        from core.wrappers import _map_unet_name_to_model_name
        result = _map_unet_name_to_model_name("cosmos-7b")
        assert result == "Cosmos-7B"

    def test_hunyuan(self):
        from core.wrappers import _map_unet_name_to_model_name
        result = _map_unet_name_to_model_name("hunyuan-video")
        assert result == "HunyuanVideo"

    def test_unknown(self):
        from core.wrappers import _map_unet_name_to_model_name
        result = _map_unet_name_to_model_name("custom-model")
        assert result == "custom-model"


class TestMapVaeName:
    def test_wan22(self):
        from core.wrappers import _map_vae_name_to_model_name
        result = _map_vae_name_to_model_name("wan2.2-vae")
        assert result == "Wan2.2-5B"

    def test_wan21_14b(self):
        from core.wrappers import _map_vae_name_to_model_name
        result = _map_vae_name_to_model_name("wan2.1-14b-vae")
        assert result == "Wan2.1-14B"

    def test_wan21(self):
        from core.wrappers import _map_vae_name_to_model_name
        result = _map_vae_name_to_model_name("wan2.1-vae")
        assert result == "Wan2.1-1.3B"

    def test_wan_generic(self):
        from core.wrappers import _map_vae_name_to_model_name
        result = _map_vae_name_to_model_name("wan-vae")
        assert result == "Wan2.2-5B"

    def test_cosmos(self):
        from core.wrappers import _map_vae_name_to_model_name
        result = _map_vae_name_to_model_name("cosmos-vae")
        assert result == "Cosmos-7B"

    def test_hunyuan(self):
        from core.wrappers import _map_vae_name_to_model_name
        result = _map_vae_name_to_model_name("hunyuan-vae")
        assert result == "HunyuanVideo"

    def test_unknown(self):
        from core.wrappers import _map_vae_name_to_model_name
        result = _map_vae_name_to_model_name("custom-vae")
        assert result == "custom-vae"


class TestResolveModelPath:
    def test_registry_hit(self):
        from core.wrappers import _resolve_model_path
        with patch("fusion_mlx.model_registry.get_registry") as mock_reg:
            mock_reg.return_value = {"test-model": {"path": "/models/test", "name": "test-model"}}
            result = _resolve_model_path("test-model")
            assert result == "/models/test"

    def test_registry_exception(self):
        from core.wrappers import _resolve_model_path
        with patch("fusion_mlx.model_registry.get_registry", side_effect=ImportError):
            with patch("os.path.isdir", return_value=True):
                result = _resolve_model_path("test-model")
                assert result is not None

    def test_dir_exists(self):
        from core.wrappers import _resolve_model_path
        with patch("fusion_mlx.model_registry.get_registry", return_value={}), \
             patch("os.path.isdir", return_value=True):
            result = _resolve_model_path("test-model")
            assert result is not None

    def test_fallback(self):
        from core.wrappers import _resolve_model_path
        with patch("fusion_mlx.model_registry.get_registry", return_value={}), \
             patch("os.path.isdir", return_value=False), \
             patch("core.wrappers._fallback_model", return_value="Wan2.2-5B"):
            result = _resolve_model_path("test-model")
            assert isinstance(result, str)


class TestFallbackModel:
    def test_direct_available(self):
        from core.wrappers import _fallback_model
        with patch("os.path.isdir", return_value=True):
            result = _fallback_model("Wan2.2-5B")
            assert result == "Wan2.2-5B"

    def test_wan_fallback(self):
        from core.wrappers import _fallback_model
        with patch("os.path.isdir", side_effect=lambda p: "Wan2.2-5B" in p):
            result = _fallback_model("wan-custom")
            assert result == "Wan2.2-5B"

    def test_ltx_fallback(self):
        from core.wrappers import _fallback_model
        with patch("os.path.isdir", side_effect=lambda p: "LTX-Video" in p):
            result = _fallback_model("ltx-custom")
            assert result == "LTX-Video"

    def test_flux_fallback(self):
        from core.wrappers import _fallback_model
        with patch("os.path.isdir", side_effect=lambda p: "FLUX.2-klein-base-4B" in p):
            result = _fallback_model("flux-custom")
            assert result == "FLUX.2-klein-base-4B"

    def test_no_available(self):
        from core.wrappers import _fallback_model
        with patch("os.path.isdir", return_value=False):
            result = _fallback_model("custom-model")
            assert result == "custom-model"
