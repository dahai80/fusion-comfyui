import json
import logging

logger = logging.getLogger("fusion_comfyui.nodes.stats")


class FusionDenoiseStatsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FUSION_PIPELINE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats_json",)
    FUNCTION = "get_stats"
    CATEGORY = "Fusion-MLX/Debug"

    def get_stats(self, pipeline):
        fn = getattr(pipeline, "last_denoise_stats", None)
        if fn is None:
            engine = getattr(pipeline, "_engine", None)
            fn = getattr(engine, "last_denoise_stats", None) if engine else None
        if fn is None:
            stats = {"available": False, "enabled": False, "reason": "no spec-denoise backend"}
        else:
            try:
                stats = fn()
            except Exception as e:
                logger.warning("FusionDenoiseStats: query failed: %s", e)
                stats = {"available": False, "enabled": False, "error": str(e)}
        logger.info(
            "FusionDenoiseStats: available=%s enabled=%s",
            stats.get("available"), stats.get("enabled"),
        )
        return (json.dumps(stats, indent=2, default=str),)
