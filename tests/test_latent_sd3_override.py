

from fusion_comfyui_plugin.nodes.latent import EmptySD3LatentImage


class TestEmptySD3LatentImageOverride:
    # AICF qwen-2512-t2i.json node 5 = EmptySD3LatentImage. ComfyUI native
    # (comfy_extras/nodes_sd3.py) returns {"samples": torch.zeros(...)} with NO
    # width/height keys, and samples is a torch.Tensor. samplers.py KSampler only
    # handles mx.array / np.ndarray shapes -> torch.Tensor hits the else fallback
    # shape=(1,16,1,64,64) -> 512x512 regardless of the requested 2560x1440. This
    # made every AICF keyframe render at 512x512 (wrong) and bypass the 768p cap.
    # Fix: fusion-comfyui overrides EmptySD3LatentImage to return mx.zeros samples
    # WITH embedded width/height (matching plugin EmptyLatentImage pattern), so
    # KSampler reads explicit width/height and the cap applies.

    def test_returns_mx_array_samples(self):
        node = EmptySD3LatentImage()
        (latent,) = node.generate(width=2560, height=1440, batch_size=1)
        import mlx.core as mx

        assert isinstance(latent["samples"], mx.array)

    def test_embeds_width_and_height_keys(self):
        node = EmptySD3LatentImage()
        (latent,) = node.generate(width=2560, height=1440, batch_size=1)
        assert latent["width"] == 2560
        assert latent["height"] == 1440

    def test_shape_matches_requested_resolution(self):
        # 16 channels (SD3/qwen latent), h//8 x w//8 spatial. 2560x1440 -> 320x180.
        node = EmptySD3LatentImage()
        (latent,) = node.generate(width=2560, height=1440, batch_size=1)
        assert tuple(latent["samples"].shape) == (1, 16, 180, 320)

    def test_default_1024(self):
        node = EmptySD3LatentImage()
        (latent,) = node.generate()
        assert latent["width"] == 1024
        assert latent["height"] == 1024
        assert tuple(latent["samples"].shape) == (1, 16, 128, 128)

    def test_batch_size(self):
        node = EmptySD3LatentImage()
        (latent,) = node.generate(width=1024, height=1024, batch_size=3)
        assert latent["samples"].shape[0] == 3

    def test_registered_in_node_class_mappings(self):
        from fusion_comfyui_plugin import NODE_CLASS_MAPPINGS

        assert "EmptySD3LatentImage" in NODE_CLASS_MAPPINGS
        assert NODE_CLASS_MAPPINGS["EmptySD3LatentImage"] is EmptySD3LatentImage
