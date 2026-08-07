
from fusion_comfyui.core.engine_wrapper import _infer_model_type, _get_latent_channels


class TestInferModelType:
    def test_image_models(self):
        assert _infer_model_type("flux2-dev") == "image"
        assert _infer_model_type("flux1-schnell") == "image"

    def test_wan_models(self):
        assert _infer_model_type("wan2.1_t2v_1.3B_fp16.safetensors") == "video"
        assert _infer_model_type("wan2.1-i2v-14B") == "video"

    def test_hunyuan_routes_to_video(self):
        assert _infer_model_type("hunyuan_video_t2v_720p_bf16.safetensors") == "video"

    def test_cosmos_routes_to_video(self):
        assert _infer_model_type("cosmos-1.0-text2video") == "video"

    def test_svd_routes_to_video(self):
        assert _infer_model_type("svd_xt.safetensors") == "video"

    def test_skyreels_and_ltx(self):
        assert _infer_model_type("skyreels-v3-r2v-14B") == "video"
        assert _infer_model_type("ltx-video-2b") == "video"

    def test_unknown_falls_back_to_image(self):
        assert _infer_model_type("some-unknown-model") == "image"


class TestLatentChannels:
    def test_hunyuan_16_channels(self):
        assert _get_latent_channels("hunyuan_video_t2v_720p_bf16.safetensors") == 16

    def test_cosmos_16_channels(self):
        assert _get_latent_channels("cosmos-1.0-text2video") == 16

    def test_svd_4_channels(self):
        assert _get_latent_channels("svd_xt.safetensors") == 4

    def test_flux_4_channels(self):
        assert _get_latent_channels("flux1-schnell") == 4

    def test_wan_16_channels(self):
        assert _get_latent_channels("wan2.1_t2v_1.3B_fp16.safetensors") == 16
