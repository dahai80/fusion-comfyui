import gc
import logging

import mlx.core as mx

logger = logging.getLogger("fusion_comfyui.core.lifecycle")


class FusionMemoryGuardian:
    _initialized = False

    @classmethod
    def setup_environment(cls):
        if cls._initialized:
            return
        import os
        os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        logger.info("FusionMemoryGuardian: pure MLX mode, no MPS suppression needed")
        cls._initialized = True

    @classmethod
    def purge_memory(cls, deep_clean: bool = True):
        gc.collect()
        try:
            mx.metal.clear_cache()
            active = mx.metal.get_active_memory()
            peak = mx.metal.get_peak_memory()
            logger.debug(
                "purge: active=%.1fMB peak=%.1fMB",
                active / 1024 / 1024,
                peak / 1024 / 1024,
            )
        except Exception as e:
            logger.warning("mx.metal.clear_cache failed: %s", e)
        if deep_clean:
            gc.collect()


class PipelineStageContext:
    def __init__(self, model_wrapper, stage_name: str):
        self.model_wrapper = model_wrapper
        self.stage_name = stage_name
        self._stage_handle = None

    def __enter__(self):
        FusionMemoryGuardian.purge_memory()
        logger.info("PipelineStage: loading '%s'", self.stage_name)
        self._stage_handle = self.model_wrapper.load_stage(self.stage_name)
        return self._stage_handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("PipelineStage: unloading '%s'", self.stage_name)
        self.model_wrapper.unload_stage(self.stage_name)
        self._stage_handle = None
        FusionMemoryGuardian.purge_memory()
        return False
