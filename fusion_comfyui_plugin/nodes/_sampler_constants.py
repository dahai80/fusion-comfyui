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
    # upstream ComfyUI spellings accepted (MLX engine ignores sampler_name;
    # these aliases let workflow_templates pass prompt validation without 400)
    "uni_pc",
    "uni_pc_bh2",
    "lcm",
    "dpmpp_2m_sde",
    "dpmpp_sde",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "dpmpp_2m_gpu",
    "dpmpp_sde_gpu",
    "heun",
    "heunpp2",
    "ipndm",
    "ipndm2",
    "lms",
]

SAMPLER_ALIASES = {
    "uni_pc": "unipc",
    "uni_pc_bh2": "unipc",
    "lcm": "euler",
    "dpmpp_2m_sde": "dpmpp_2m",
    "dpmpp_sde": "dpmpp_2m",
    "dpmpp_3m_sde": "dpmpp_2m",
    "dpmpp_3m_sde_gpu": "dpmpp_2m",
    "dpmpp_2m_gpu": "dpmpp_2m",
    "dpmpp_sde_gpu": "dpmpp_2m",
    "heun": "euler",
    "heunpp2": "euler",
    "ipndm": "euler",
    "ipndm2": "euler",
    "lms": "euler",
}


def normalize_sampler(name):
    return SAMPLER_ALIASES.get(name, name)

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
