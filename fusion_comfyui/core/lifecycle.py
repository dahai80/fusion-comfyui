import gc
import logging
import os

logger = logging.getLogger("fusion_comfyui.core.lifecycle")

_PURGE_THRESHOLD_MB = 1024


class FusionMemoryGuardian:
    _initialized = False

    @classmethod
    def setup_environment(cls):
        if cls._initialized:
            return
        os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        logger.info("FusionMemoryGuardian: pure MLX mode, no MPS suppression needed")
        cls._initialized = True

    @classmethod
    def purge_memory(cls, deep_clean: bool = True):
        gc.collect()
        try:
            import mlx.core as mx
            clear = getattr(mx, "clear_cache", None)
            if clear is not None:
                clear()
            else:
                mx.metal.clear_cache()
            active_fn = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
            peak_fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
            active = active_fn()
            peak = peak_fn()
            logger.debug(
                "FusionMemoryGuardian purge: active=%.1fMB peak=%.1fMB",
                active / 1024 / 1024,
                peak / 1024 / 1024,
            )
        except Exception as e:
            logger.warning("FusionMemoryGuardian: clear_cache failed: %s", e)

        if deep_clean:
            gc.collect()

    @classmethod
    def maybe_purge(cls, threshold_mb=_PURGE_THRESHOLD_MB):
        try:
            import mlx.core as mx
            active_fn = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
            active_mb = active_fn() / 1024 / 1024
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
