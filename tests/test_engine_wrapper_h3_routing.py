from fusion_comfyui.core.engine_wrapper import _infer_model_type, _get_latent_channels


class TestH3Routing:
    def test_minimax_routes_to_video(self):
        assert _infer_model_type("minimax-h3") == "video"
        assert _infer_model_type("MiniMax-H3") == "video"

    def test_h3_routes_to_video(self):
        assert _infer_model_type("h3-fl2va") == "video"
        assert _infer_model_type("h3-ref2va") == "video"

    def test_fl2va_ref2va_routes_to_video(self):
        assert _infer_model_type("fl2va-14B") == "video"
        assert _infer_model_type("ref2va-14B") == "video"

    def test_h3_latent_channels_24(self):
        assert _get_latent_channels("minimax-h3") == 24
        assert _get_latent_channels("h3-fl2va") == 24
        assert _get_latent_channels("fl2va-14B") == 24
