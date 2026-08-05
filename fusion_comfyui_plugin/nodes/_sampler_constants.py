import logging

logger = logging.getLogger("fusion_comfyui.nodes._sampler_constants")

SAMPLER_NAMES = [
    "euler",
    "euler_ancestral",
    "dpm++",
    "dpmpp_2m",
    "unipc",
    "dpm_fast",
    "dpm_adaptive",
]

SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
]

MLX_SCHEDULER_NAMES = ["euler", "dpm++", "unipc"]

logger.info(
    "sampler constants loaded: %d samplers, %d schedulers, mlx_schedulers=%s",
    len(SAMPLER_NAMES), len(SCHEDULER_NAMES), MLX_SCHEDULER_NAMES,
)
