import gc
import logging
import os

logger = logging.getLogger("fusion_comfyui.lifecycle")

_PURGE_THRESHOLD_MB = 1024


class FusionMemoryGuardian:
    _initialized = False

    @classmethod
    def setup_environment(cls):
        if cls._initialized:
            return
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("FusionMemoryGuardian: environment configured (no PyTorch)")
        cls._initialized = True

    @classmethod
    def purge_memory(cls, deep_clean: bool = True):
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
            active = mx.metal.get_active_memory()
            peak = mx.metal.get_peak_memory()
            logger.debug(
                "FusionMemoryGuardian purge: active=%.1fMB peak=%.1fMB",
                active / 1024 / 1024,
                peak / 1024 / 1024,
            )
        except Exception as e:
            logger.warning("FusionMemoryGuardian: mx.metal.clear_cache failed: %s", e)

        if deep_clean:
            gc.collect()

    @classmethod
    def maybe_purge(cls, threshold_mb=_PURGE_THRESHOLD_MB):
        try:
            import mlx.core as mx
            active_mb = mx.metal.get_active_memory() / 1024 / 1024
            if active_mb < threshold_mb:
                logger.debug(
                    "FusionMemoryGuardian maybe_purge: skipping, active=%.0fMB < threshold=%dMB",
                    active_mb, threshold_mb,
                )
                return
            logger.info(
                "FusionMemoryGuardian maybe_purge: active=%.0fMB >= threshold=%dMB, purging",
                active_mb, threshold_mb,
            )
        except Exception:
            pass
        cls.purge_memory()


class PipelineStageContext:
    def __init__(self, model_wrapper, stage_name: str):
        self.model_wrapper = model_wrapper
        self.stage_name = stage_name
        self._stage_handle = None

    def __enter__(self):
        FusionMemoryGuardian.maybe_purge()
        logger.info("PipelineStage: loading '%s'", self.stage_name)
        self._stage_handle = self.model_wrapper.load_stage(self.stage_name)
        return self._stage_handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("PipelineStage: unloading '%s'", self.stage_name)
        self.model_wrapper.unload_stage(self.stage_name)
        self._stage_handle = None
        FusionMemoryGuardian.maybe_purge()
        return False
