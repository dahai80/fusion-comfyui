from fusion_comfyui_plugin.nodes.loaders import (
    CLIPLoader,
    _get_diffusion_models,
    _get_text_encoders,
    _get_vae_models,
)


class TestQwenT2iFilenamesKnown:
    def test_diffusion_models_include_qwen_gguf(self):
        models = _get_diffusion_models()
        assert any("qwen-image" in m.lower() and ".gguf" in m.lower() for m in models), models

    def test_text_encoders_include_qwen_25_vl(self):
        encs = _get_text_encoders()
        assert any("qwen_2.5_vl" in e.lower() for e in encs), encs

    def test_vae_models_include_qwen_image_vae(self):
        vaes = _get_vae_models()
        assert any("qwen_image_vae" in v.lower() for v in vaes), vaes


class TestCLIPLoaderQwenImageType:
    def test_type_enum_includes_qwen_image(self):
        req = CLIPLoader.INPUT_TYPES()["required"]
        types = req["type"][0]
        assert "qwen_image" in types, types


class TestUnetLoaderGGUFRegistered:
    def test_unet_loader_gguf_class_exists(self):
        from fusion_comfyui_plugin.nodes.passthrough import UnetLoaderGGUF
        assert UnetLoaderGGUF.RETURN_TYPES == ("MODEL",)
        assert UnetLoaderGGUF.FUNCTION == "load_unet"

    def test_unet_loader_gguf_in_node_mappings(self):
        import fusion_comfyui_plugin
        mappings = fusion_comfyui_plugin.NODE_CLASS_MAPPINGS
        assert "UnetLoaderGGUF" in mappings, list(mappings.keys())[:20]

    def test_unet_loader_gguf_returns_fusion_wrapper(self):
        from fusion_comfyui_plugin.nodes.passthrough import UnetLoaderGGUF
        from fusion_comfyui.core.wrappers import FusionModelWrapper
        node = UnetLoaderGGUF()
        result = node.load_unet("qwen-image-2512-Q4_K_M.gguf")
        assert isinstance(result[0], FusionModelWrapper)


class TestQwenUnetNameMapping:
    def test_qwen_gguf_maps_to_qwen_repo(self):
        from fusion_comfyui.core.wrappers import _map_unet_name_to_model_name
        mapped = _map_unet_name_to_model_name("qwen-image-2512-Q4_K_M.gguf")
        assert "qwen" in mapped.lower(), mapped

    def test_qwen_inferred_as_image_type(self):
        from fusion_comfyui.core.wrappers import _infer_model_type
        assert _infer_model_type("qwen-image-2512-Q4_K_M.gguf") == "image"

    def test_qwen_does_not_infer_as_video(self):
        from fusion_comfyui.core.wrappers import _infer_model_type
        assert _infer_model_type("qwen-image-2512-Q4_K_M.gguf") != "video"


class TestQwenImageRoutingMonolith:
    def _wrapper(self, name, model_type):
        class _W:
            pass
        w = _W()
        w.model_name = name
        w.model_type = model_type
        return w

    def test_qwen_image_txt2img_routes_to_monolith(self):
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("mlx-community/Qwen-Image-2512-4bit", "image")
        latent = {"samples": None}
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 1.0) is False

    def test_flux_image_txt2img_still_routes_to_staged(self):
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("flux2-4b", "image")
        latent = {"samples": None}
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 1.0) is True

    def test_qwen_image_img2img_routes_to_monolith(self):
        from fusion_comfyui_plugin.nodes.samplers import _should_use_staged
        w = self._wrapper("mlx-community/Qwen-Image-2512-4bit", "image")
        latent = {"samples": None, "_image_init_path": "/tmp/x.png"}
        assert _should_use_staged(w, {"prompt": "x"}, {"prompt": ""}, latent, 0.6) is False
