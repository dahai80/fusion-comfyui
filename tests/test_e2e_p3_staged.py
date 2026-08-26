# SPDX-License-Identifier: Apache-2.0
# P3 Task 6: real-model e2e for the _run_staged_pipeline orchestration helper
# on FusionEngineWrapper (fusion_comfyui/core/engine_wrapper.py:289).
#
# Drives Wan2.1-T2V-1.3B through the full staged path end-to-end:
#   load_text_encoder -> encode(+) -> encode(-) -> unload_text_encoder ->
#   load_dit -> denoise -> unload_dit -> load_vae -> decode -> unload_vae.
# The existing test_e2e_wan2_staged.py already covers the 10 individual stage
# methods; this test's unique value is exercising the NEW _run_staged_pipeline
# helper that chains them with strict offload + purge_memory between stages.
#
# Per project rule: 涉及到大模型测试，须真实加载模型.
# Skips cleanly when the model is absent or no Metal GPU is available, so it
# is safe to run in CI without Apple Silicon / downloaded weights.

import asyncio
import logging
import os

import pytest

logger = logging.getLogger("test_e2e_p3_staged")

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
    not _has_metal_gpu()
    or not _model_present()
    or os.environ.get("RUN_E2E", "").strip().lower() not in ("1", "true", "yes"),
    reason=(
        "requires Apple Silicon Metal GPU + Wan2.1-1.3B at "
        f"{MODEL_PATH} + RUN_E2E=1; run on a real machine with the model "
        "downloaded and real model loading enabled"
    ),
)


async def _run_p3_staged():
    import mlx.core as mx
    import numpy as np

    from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper

    wrapper = FusionEngineWrapper(
        MODEL_NAME, offload_strategy="sequential", quant_bit="fp8_e4m3"
    )
    logger.info(
        "=== P3 staged pipeline e2e: %s %dx%d %dframes steps=%d cfg=%.1f seed=%d ===",
        MODEL_NAME, WIDTH, HEIGHT, NUM_FRAMES, STEPS, CFG, SEED,
    )

    latent = wrapper.create_empty_latent(HEIGHT, WIDTH, NUM_FRAMES)
    logger.info("empty latent shape=%s dtype=%s", tuple(latent.shape), latent.dtype)

    pixels = await wrapper._run_staged_pipeline(
        latent, PROMPT, NEG_PROMPT, STEPS, CFG, SEED, num_frames=NUM_FRAMES,
    )
    logger.info("staged pixels shape=%s dtype=%s", tuple(pixels.shape), pixels.dtype)

    mx.eval(pixels)
    arr = np.array(pixels)
    assert arr.ndim >= 3, f"pixels wrong ndim: {arr.ndim}"
    assert arr.shape[-1] == 3 or arr.shape[1] == 3, (
        f"pixels wrong channels: {arr.shape}"
    )
    assert not np.isnan(arr).any(), f"pixels contain NaN (shape={arr.shape})"
    non_zero = int((np.abs(arr) > 0.01).sum())
    total = int(arr.size)
    frac = non_zero / total
    logger.info(
        "non-zero pixel fraction: %.4f (%d/%d) std=%.4f",
        frac, non_zero, total, float(arr.std()),
    )
    assert frac > 0.01, f"output looks empty (non-zero fraction {frac:.4f})"

    await wrapper.stop()
    logger.info("=== cleanup done ===")
    return arr


def test_p3_run_staged_pipeline_wan2_1_3b():
    # P3 exit criterion #6: _run_staged_pipeline orchestration helper produces
    # valid pixels end-to-end on a real Wan2.1-T2V-1.3B model, with strict
    # load/unload between text_encoder, dit, and vae stages (memory sawtooth).
    arr = asyncio.run(_run_p3_staged())
    assert arr is not None
    logger.info(
        "P3 staged pipeline PASSED, output shape=%s std=%.4f",
        arr.shape, float(arr.std()),
    )
