import logging

import numpy as np
import pytest

logger = logging.getLogger("fusion.test.cascade_bridge")

_CASCADE_DIR = "models--stabilityai--stable-cascade-prior"


@pytest.fixture
def cascade_model_installed(monkeypatch):
    # _available_cascade_model() probes ~/.fusion-mlx/models for the cascade
    # dir. On CI (no model downloaded) it returns None and the wuerstchen ->
    # cascade routing falls through, so stub it to the expected dir to test
    # the routing logic independent of the local filesystem.
    import fusion_comfyui_plugin.core.wrappers as w
    monkeypatch.setattr(w, "_available_cascade_model", lambda: _CASCADE_DIR)
    return _CASCADE_DIR


class TestCascadeRouting:
    def test_fallback_model_routes_cascade_to_prior(self, cascade_model_installed):
        from fusion_comfyui_plugin.core.wrappers import _fallback_model
        resolved = _fallback_model("stable_cascade_stage_b.safetensors")
        assert "cascade" in resolved.lower(), resolved
        logger.info("fallback stage_b -> %s", resolved)

    def test_fallback_model_routes_wuerstchen_to_prior(self, cascade_model_installed):
        from fusion_comfyui_plugin.core.wrappers import _fallback_model
        resolved = _fallback_model("wuerstchen_v3_stage_c.safetensors")
        assert "cascade" in resolved.lower(), resolved

    def test_fallback_cascade_never_returns_video_model(self, cascade_model_installed):
        from fusion_comfyui_plugin.core.wrappers import _fallback_model
        for ckpt in ("stable_cascade_stage_b.safetensors",
                     "stable_cascade_stage_c.safetensors",
                     "wuerstchen_v3_stage_c.safetensors"):
            resolved = _fallback_model(ckpt)
            low = resolved.lower()
            assert "wan" not in low and "flux" not in low and "ltx" not in low, resolved

    def test_map_checkpoint_cascade(self, cascade_model_installed):
        from fusion_comfyui_plugin.core.wrappers import _map_checkpoint_to_model_name
        for ckpt in ("stable_cascade_stage_b.safetensors",
                     "stable_cascade_stage_c.safetensors",
                     "wuerstchen_v3.safetensors"):
            mapped = _map_checkpoint_to_model_name(ckpt)
            assert "cascade" in mapped.lower(), (ckpt, mapped)

    def test_is_cascade_name(self):
        from fusion_comfyui_plugin.core.wrappers import _is_cascade_name
        assert _is_cascade_name("models--stabilityai--stable-cascade-prior") is True
        assert _is_cascade_name("wuerstchen_v3") is True
        assert _is_cascade_name("wan2.2_vae") is False
        assert _is_cascade_name("") is False


class TestFusionVAEWrapperCascade:
    def test_cascade_vae_has_downscale_ratio_4(self):
        from fusion_comfyui_plugin.core.wrappers import FusionVAEWrapper
        vae = FusionVAEWrapper(
            model_path="/x", model_name="models--stabilityai--stable-cascade-prior",
        )
        assert vae.downscale_ratio == 4
        assert vae._is_cascade is True

    def test_non_cascade_vae_downscale_ratio_8(self):
        from fusion_comfyui_plugin.core.wrappers import FusionVAEWrapper
        vae = FusionVAEWrapper(model_path="/x", model_name="wan2.2_vae")
        assert vae.downscale_ratio == 8
        assert vae._is_cascade is False

    def test_cascade_vae_encode_shape(self):
        from fusion_comfyui_plugin.core.wrappers import FusionVAEWrapper
        vae = FusionVAEWrapper(
            model_path="/x", model_name="models--stabilityai--stable-cascade-prior",
        )
        img = np.zeros((1, 3, 512, 512), dtype=np.float32)
        latent = vae.encode(img)
        assert latent.shape == (1, 16, 128, 128), latent.shape
        assert latent.dtype == np.float32


class TestStableCascadeNodeOverrides:
    def test_empty_latent_image_shapes(self):
        from fusion_comfyui_plugin.nodes.loaders import StableCascade_EmptyLatentImage
        node = StableCascade_EmptyLatentImage()
        rt = node.RETURN_TYPES
        assert rt == ("LATENT", "LATENT")
        stage_c, stage_b = node.execute(width=1024, height=1024, compression=42, batch_size=1)
        assert stage_c["samples"].shape == (1, 16, 1024 // 42, 1024 // 42)
        assert stage_b["samples"].shape == (1, 4, 256, 256)
        assert stage_c["width"] == 1024 and stage_c["height"] == 1024
        logger.info("empty latent c=%s b=%s", stage_c["samples"].shape, stage_b["samples"].shape)

    def test_empty_latent_image_input_types(self):
        from fusion_comfyui_plugin.nodes.loaders import StableCascade_EmptyLatentImage
        inputs = StableCascade_EmptyLatentImage.INPUT_TYPES()
        assert set(inputs["required"]) == {"width", "height", "compression", "batch_size"}

    def test_stage_c_vae_encode_uses_wrapper(self):
        from fusion_comfyui_plugin.nodes.loaders import StableCascade_StageC_VAEEncode
        from fusion_comfyui_plugin.core.wrappers import FusionVAEWrapper

        class _Img:
            shape = (1, 512, 512, 3)
            def movedim(self, src, dst):
                arr = np.zeros(self.shape, dtype=np.float32)
                return np.moveaxis(arr, src, dst)

        vae = FusionVAEWrapper(
            model_path="/x", model_name="models--stabilityai--stable-cascade-prior",
        )
        node = StableCascade_StageC_VAEEncode()
        rt = node.RETURN_TYPES
        assert rt == ("LATENT", "LATENT")
        stage_c, stage_b = node.execute(image=_Img(), vae=vae, compression=42)
        assert stage_c["samples"].shape[1] == 16
        assert stage_b["samples"].shape[1] == 4
        assert stage_c["width"] == 512 and stage_c["height"] == 512

    def test_stage_b_conditioning_dict_format(self):
        from fusion_comfyui_plugin.nodes.loaders import StableCascade_StageB_Conditioning
        node = StableCascade_StageB_Conditioning()
        cond = {"prompt": "a cat", "clip": None, "embed": None}
        stage_c = {"samples": np.zeros((1, 16, 24, 24), dtype=np.float32)}
        (out,) = node.execute(conditioning=cond, stage_c=stage_c)
        assert isinstance(out, dict)
        assert out["stable_cascade_prior"] is stage_c["samples"]
        assert out["prompt"] == "a cat"

    def test_stage_b_conditioning_native_list_format(self):
        from fusion_comfyui_plugin.nodes.loaders import StableCascade_StageB_Conditioning
        node = StableCascade_StageB_Conditioning()
        cond = [[None, {"pooled_output": None}], [None, {"pooled_output": None}]]
        stage_c = {"samples": np.zeros((1, 16, 24, 24), dtype=np.float32)}
        (out,) = node.execute(conditioning=cond, stage_c=stage_c)
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(t[1]["stable_cascade_prior"] is stage_c["samples"] for t in out)

    def test_super_resolution_controlnet_shapes(self):
        from fusion_comfyui_plugin.nodes.loaders import StableCascade_SuperResolutionControlnet
        from fusion_comfyui_plugin.core.wrappers import FusionVAEWrapper

        class _Img:
            shape = (1, 512, 512, 3)
            def movedim(self, src, dst):
                arr = np.zeros(self.shape, dtype=np.float32)
                return np.moveaxis(arr, src, dst)

        vae = FusionVAEWrapper(
            model_path="/x", model_name="models--stabilityai--stable-cascade-prior",
        )
        node = StableCascade_SuperResolutionControlnet()
        rt = node.RETURN_TYPES
        assert rt == ("IMAGE", "LATENT", "LATENT")
        cnet, stage_c, stage_b = node.execute(image=_Img(), vae=vae)
        assert stage_c["samples"].shape == (1, 16, 32, 32)
        assert stage_b["samples"].shape == (1, 4, 256, 256)


class TestCascadeRegistration:
    def test_nodes_registered_in_mappings(self):
        import fusion_comfyui_plugin as p
        names = [
            "StableCascade_EmptyLatentImage",
            "StableCascade_StageC_VAEEncode",
            "StableCascade_StageB_Conditioning",
            "StableCascade_SuperResolutionControlnet",
        ]
        for n in names:
            assert n in p.NODE_CLASS_MAPPINGS, n
            assert n in p._native_overrides, n
        logger.info("all 4 cascade overrides registered in both dicts")
