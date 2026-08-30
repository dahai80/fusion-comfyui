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


class TestH3VideoRoutingMonolith:
    # H3 backend has no stage API (load_text_encoder raises NotImplementedError,
    # issue #170 phase 2). Any H3 video job — i2v (first/last frame), t2va, ref2va —
    # must route to _generate_monolithic, never the staged path.

    def _wrapper(self, name, model_type):
        class _W:
            pass
        w = _W()
        w.model_name = name
        w.model_type = model_type
        return w

    def test_h3_i2v_first_frame_routes_to_monolith(self):
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("minimax-h3", "video")
        latent = {
            "samples": None,
            "_h3_quantize": "dit8_te4",
            "_h3_first_frame_path": "/tmp/fusion_h3_i2v_first_8tt8eevv.png",
        }
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 1.0) is False

    def test_h3_i2v_last_frame_routes_to_monolith(self):
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("minimax-h3", "video")
        latent = {
            "samples": None,
            "_h3_last_frame_path": "/tmp/fusion_h3_i2v_last_tyvz7h2c.png",
        }
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 1.0) is False

    def test_h3_audio_t2va_routes_to_monolith(self):
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("minimax-h3", "video")
        latent = {"samples": None, "_h3_audio": True}
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 1.0) is False

    def test_h3_ref_images_routes_to_monolith(self):
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("minimax-h3", "video")
        latent = {"samples": None, "_h3_ref_images": ["/tmp/ref0.png"]}
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 1.0) is False

    def test_h3_quantize_only_t2v_routes_to_monolith(self):
        # H3 T2V (no i2v frame, no audio) still has no stage API -> monolith.
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("minimax-h3", "video")
        latent = {"samples": None, "_h3_quantize": "dit8_te4"}
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 1.0) is False

    def test_h3_plain_t2v_routes_to_monolith(self):
        # H3 backend never implements staged; even a bare T2V must not stage.
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("minimax-h3", "video")
        latent = {"samples": None}
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 1.0) is False

    def test_non_h3_video_t2v_still_routes_to_staged(self):
        # Regression guard: wan2 T2V (no h3 keys, non-h3 name) keeps staged path.
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("mlx-community/wan2.1-t2v-5B", "video")
        latent = {"samples": None}
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 1.0) is True
