from unittest.mock import patch


class TestFusionMemoryGuardian:
    def test_purge_memory(self):
        from fusion_comfyui.core.lifecycle import FusionMemoryGuardian
        with patch("mlx.core.clear_cache"):
            FusionMemoryGuardian.purge_memory()

    def test_purge_memory_deep_clean(self):
        from fusion_comfyui.core.lifecycle import FusionMemoryGuardian
        with patch("mlx.core.clear_cache"), \
             patch("gc.collect"):
            FusionMemoryGuardian.purge_memory(deep_clean=True)

    def test_purge_uses_non_deprecated_clear_cache(self):
        # Regression guard: purge_memory MUST NOT call the deprecated
        # mx.metal.clear_cache(), which corrupts live Metal command buffers
        # mid-generation (Invalid Resource abort). It must use mx.clear_cache.
        from fusion_comfyui.core.lifecycle import FusionMemoryGuardian
        import mlx.core as mx
        with patch("mlx.core.clear_cache") as mock_clear, \
             patch.object(mx.metal, "clear_cache") as mock_deprecated:
            FusionMemoryGuardian.purge_memory()
            assert mock_clear.called, "purge_memory must call mx.clear_cache"
            assert not mock_deprecated.called, (
                "purge_memory must NOT call deprecated mx.metal.clear_cache"
            )

    def test_setup_environment(self):
        from fusion_comfyui.core.lifecycle import FusionMemoryGuardian
        FusionMemoryGuardian.setup_environment()


class TestSetupEnvironment:
    def test_setup(self):
        from fusion_comfyui.core.lifecycle import FusionMemoryGuardian
        FusionMemoryGuardian._initialized = False
        FusionMemoryGuardian.setup_environment()
