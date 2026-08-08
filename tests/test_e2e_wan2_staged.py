# SPDX-License-Identifier: Apache-2.0
# E2E regression test for upstream fusion-mlx #418 / PR #419 (staged VAE decode
# Stream(gpu, N) cross-thread error). Drives the real Wan2.1-T2V-1.3B model
# through the Phase-2 staged path:
#   FusionTextEncoder(+) -> FusionTextEncoder(-) -> FusionKSampler ->
#   FusionVAEDecoder, via FusionEngineWrapper -> VideoGenEngine -> Wan2Backend.
# Verifies the 10 stage methods work on a real model (not just unit fakes) and
# that lazy cross-thread MLX arrays no longer raise at VAE decode.
# Per project rule: 涉及到大模型测试，须真实加载模型.
#
# Skips cleanly when the model is absent or no Metal GPU is available, so it is
# safe to run in CI without Apple Silicon / downloaded weights.

import asyncio
import logging
import os

import pytest

logger = logging.getLogger("test_e2e_wan2_staged")

MODEL_NAME = "Wan2.1-1.3B"
MODEL_PATH = os.path.expanduser("~/.fusion-mlx/models/Wan2.1-1.3B")
NUM_FRAMES = 17
WIDTH = 832
HEIGHT = 480
STEPS = 8
CFG = 5.0
SEED = 42
PROMPT = "a cute cat playing with a ball of yarn, cinematic lighting"
NEG_PROMPT = "blurry, low quality, distorted"


def _has_metal_gpu():
    try:
        import mlx.core as mx
        return mx.default_device().type == getattr(mx.DeviceType, "gpu", -1)
    except Exception as e:
        logger.debug("metal probe failed: %s", e)
        return False


def _model_present():
    return os.path.isdir(MODEL_PATH)


pytestmark = pytest.mark.skipif(
    not _has_metal_gpu() or not _model_present(),
    reason=(
        "requires Apple Silicon Metal GPU + Wan2.1-1.3B at "
        f"{MODEL_PATH}; run on a real machine with the model downloaded"
    ),
)


async def _run_staged_t2v():
    from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper

    wrapper = FusionEngineWrapper(
        MODEL_NAME, offload_strategy="sequential", quant_bit="fp8_e4m3"
    )
    logger.info(
        "=== e2e staged T2V: %s %dx%d %dframes steps=%d ===",
        MODEL_NAME, WIDTH, HEIGHT, NUM_FRAMES, STEPS,
    )

    logger.info("[1/3] load_text_encoder")
    await wrapper.load_text_encoder()
    logger.info("[1/3] encode_text positive")
    positive = await wrapper.encode_text(PROMPT)
    assert "embed" in positive, f"positive missing embed: {list(positive.keys())}"
    logger.info("[1/3] positive embed shape=%s", tuple(positive["embed"].shape))

    logger.info("[1/3] encode_text negative")
    negative = await wrapper.encode_text(NEG_PROMPT)
    logger.info("[1/3] negative embed shape=%s", tuple(negative["embed"].shape))

    logger.info("[1/3] unload_text_encoder")
    await wrapper.unload_text_encoder()

    logger.info("[2/3] load_dit")
    await wrapper.load_dit()
    latent = wrapper.create_empty_latent(HEIGHT, WIDTH, NUM_FRAMES)
    logger.info("[2/3] empty latent shape=%s dtype=%s", tuple(latent.shape), latent.dtype)

    logger.info("[2/3] denoise steps=%d cfg=%.1f seed=%d", STEPS, CFG, SEED)
    denoised = await wrapper.denoise(
        latent, positive, negative,
        steps=STEPS, cfg=CFG, seed=SEED,
        width=WIDTH, height=HEIGHT, num_frames=NUM_FRAMES,
    )
    logger.info("[2/3] denoised latent shape=%s", tuple(denoised.shape))
    assert denoised.ndim >= 4, f"denoised latent wrong ndim: {denoised.ndim}"

    logger.info("[2/3] unload_dit")
    await wrapper.unload_dit()

    logger.info("[3/3] load_vae")
    await wrapper.load_vae()
    use_tiled = os.environ.get("E2E_TILED", "1") == "1"
    if use_tiled:
        logger.info("[3/3] decode_tiled")
        image = await wrapper.decode_tiled(denoised, tile_size=256)
    else:
        logger.info("[3/3] decode (non-tiled)")
        image = await wrapper.decode(denoised, tile_size=256)
    logger.info("[3/3] decoded image shape=%s", tuple(image.shape))
    assert image.ndim >= 3, f"decoded image wrong ndim: {image.ndim}"
    assert image.shape[-1] == 3 or image.shape[1] == 3, (
        f"decoded image wrong channels: {image.shape}"
    )

    logger.info("[3/3] unload_vae")
    await wrapper.unload_vae()

    import mlx.core as mx
    mx.eval(image)
    arr = mx.array(image)
    non_zero = float(mx.sum(arr > 0.01))
    total = arr.size
    frac = non_zero / total
    logger.info("non-zero pixel fraction: %.4f (%d/%d)", frac, int(non_zero), total)
    assert frac > 0.01, f"output looks empty (non-zero fraction {frac:.4f})"

    await wrapper.stop()
    logger.info("=== cleanup done ===")
    return image


def test_staged_t2v_wan2_1_3b():
    # Regression for fusion-mlx #418: the staged path round-trips denoised
    # latents through the main thread; lazy cross-thread MLX arrays used to
    # raise "There is no Stream(gpu, N) in current thread" at VAE decode.
    # A passing run means the eval-on-producer-thread fix holds.
    image = asyncio.run(_run_staged_t2v())
    assert image is not None
    logger.info("staged T2V regression PASSED, output shape=%s", tuple(image.shape))
