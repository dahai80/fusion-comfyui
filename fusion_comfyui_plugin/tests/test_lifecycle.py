from unittest.mock import MagicMock, patch


class TestFusionMemoryGuardian:
    def test_purge_memory(self):
        from core.lifecycle import FusionMemoryGuardian
        with patch("mlx.core.metal.clear_cache"):
            FusionMemoryGuardian.purge_memory()

    def test_purge_memory_deep_clean(self):
        from core.lifecycle import FusionMemoryGuardian
        with patch("mlx.core.metal.clear_cache"), \
             patch("gc.collect"):
            FusionMemoryGuardian.purge_memory(deep_clean=True)

    def test_setup_environment(self):
        from core.lifecycle import FusionMemoryGuardian
        FusionMemoryGuardian.setup_environment()


class TestPipelineStageContext:
    def test_init(self):
        from core.lifecycle import PipelineStageContext
        mock_wrapper = MagicMock()
        ctx = PipelineStageContext(model_wrapper=mock_wrapper, stage_name="encode")
        assert ctx.stage_name == "encode"

    def test_context_manager(self):
        from core.lifecycle import PipelineStageContext
        mock_wrapper = MagicMock()
        mock_wrapper.load_stage.return_value = "handle"
        with patch("core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            ctx = PipelineStageContext(model_wrapper=mock_wrapper, stage_name="encode")
            with ctx as handle:
                assert handle == "handle"
            mock_wrapper.unload_stage.assert_called_with("encode")


class TestSetupEnvironment:
    def test_setup(self):
        from core.lifecycle import FusionMemoryGuardian
        FusionMemoryGuardian._initialized = False
        FusionMemoryGuardian.setup_environment()
