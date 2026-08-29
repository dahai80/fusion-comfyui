from fusion_comfyui_plugin.nodes.loaders import (
    CLIPLoader,
    _get_diffusion_models,
    _get_vae_models,
)


class TestCLIPLoaderMinimaxType:
    def test_type_enum_includes_minimax(self):
        req = CLIPLoader.INPUT_TYPES()["required"]
        types = req["type"][0]
        assert "minimax" in types, types

    def test_clip_name_list_includes_h3_text_encoder(self):
        req = CLIPLoader.INPUT_TYPES()["required"]
        names = req["clip_name"][0]
        assert any("minimax" in n.lower() and "h3" in n.lower() for n in names), names


class TestUNETLoaderH3Weights:
    def test_diffusion_models_include_fl2va_pruned(self):
        models = _get_diffusion_models()
        assert any("fl2va" in m.lower() and "pruned" in m.lower() for m in models), models

    def test_diffusion_models_include_ref2va_pruned(self):
        models = _get_diffusion_models()
        assert any("ref2va" in m.lower() and "pruned" in m.lower() for m in models), models


class TestVAELoaderH3Weights:
    def test_vae_models_include_h3_video_vae(self):
        vaes = _get_vae_models()
        assert any("h3" in v.lower() and "video_vae" in v.lower() for v in vaes), vaes

    def test_vae_models_include_h3_audio_vae(self):
        vaes = _get_vae_models()
        assert any("h3" in v.lower() and "audio_vae" in v.lower() for v in vaes), vaes


class TestClipTypeMinimaxMapping:
    def test_minimax_type_maps_to_h3(self):
        from fusion_comfyui.core.wrappers import _map_clip_type_to_model_name
        model = _map_clip_type_to_model_name("minimax", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
        assert "h3" in model.lower() or "minimax" in model.lower(), model


class TestH3SamplerCompat:
    def test_sampler_names_include_res_multistep(self):
        from fusion_comfyui_plugin.nodes._sampler_constants import SAMPLER_NAMES
        assert "res_multistep" in SAMPLER_NAMES, SAMPLER_NAMES

    def test_res_multistep_normalizes_to_known_solver(self):
        from fusion_comfyui_plugin.nodes._sampler_constants import normalize_sampler
        out = normalize_sampler("res_multistep")
        assert out in ("euler", "dpm++", "unipc", "res_multistep"), out


class TestEmptyH3LatentBatchSizeOptional:
    def test_batch_size_optional_when_omitted(self):
        from fusion_comfyui_plugin.nodes.h3 import EmptyMiniMaxH3LatentAV
        req = EmptyMiniMaxH3LatentAV.INPUT_TYPES()
        has_required_bs = "batch_size" in req.get("required", {})
        has_optional_bs = "batch_size" in req.get("optional", {})
        assert not has_required_bs, "batch_size must not be required (AICF omits it)"
        assert has_optional_bs or "batch_size" not in {**req.get("required", {}), **req.get("optional", {})}, \
            "batch_size should be optional or absent so AICF workflows validate"
