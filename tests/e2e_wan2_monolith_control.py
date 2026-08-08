# SPDX-License-Identifier: Apache-2.0
# Control: monolithic generate() at the same resolution to isolate whether the
# VAE tiled-decode Stream error is specific to the stage split or general.

import asyncio
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("e2e_control")

MODEL_NAME = "Wan2.1-1.3B"
NUM_FRAMES = 17
WIDTH = 832
HEIGHT = 480
STEPS = 8
SEED = 42
PROMPT = "a cute cat playing with a ball of yarn, cinematic lighting"


async def main():
    from fusion_mlx.engines.video import VideoGenEngine
    from fusion_mlx.model_registry import list_available_models

    model_path = MODEL_NAME
    for m in list_available_models("video"):
        if m["name"] == MODEL_NAME:
            model_path = m["path"]
            break
    logger.info("resolved model_path=%s", model_path)
    engine = VideoGenEngine(model_path)
    await engine.start()
    logger.info("=== control monolith generate(): %s %dx%d %dframes ===", MODEL_NAME, WIDTH, HEIGHT, NUM_FRAMES)
    t0 = time.time()
    result = await engine.generate(
        prompt=PROMPT,
        num_frames=NUM_FRAMES,
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        seed=SEED,
        n=1,
    )
    elapsed = time.time() - t0
    logger.info("=== control PASSED: monolith in %.1fs, result type=%s len=%s ===", elapsed, type(result).__name__, len(result) if hasattr(result, "__len__") else "?")
    await engine.stop()
    return 0


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
    except Exception:
        logger.exception("=== control FAILED ===")
        rc = 1
    sys.exit(rc)
