#!/usr/bin/env python3
import json
import logging
import time
import statistics
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("bench_perf")

MODEL_BASE = os.path.expanduser("~/.fusion-mlx/models")


def bench_image_generation(model_path=None, num_runs=3, steps=4):
    from fusion_mlx.engines.image_gen import ImageGenEngine
    from core.lifecycle import FusionMemoryGuardian

    if model_path is None:
        model_path = os.path.join(MODEL_BASE, "FLUX.2-klein-base-4B")
    model_name = os.path.basename(model_path)

    logger.info("=== Image Generation Benchmark ===")
    logger.info("Model: %s, Runs: %d, Steps: %d", model_name, num_runs, steps)

    start_times = []
    gen_times = []
    total_times = []

    for i in range(num_runs):
        FusionMemoryGuardian.maybe_purge()
        t0 = time.perf_counter()
        logger.info("--- Run %d/%d ---", i + 1, num_runs)

        engine = ImageGenEngine(model_path)
        t1 = time.perf_counter()
        asyncio.run(engine.start())
        t2 = time.perf_counter()
        start_time = t2 - t1
        start_times.append(start_time)
        logger.info("Engine start: %.2fs", start_time)

        t3 = time.perf_counter()
        result = asyncio.run(engine.generate(
            prompt="a beautiful sunset over mountains, oil painting",
            width=512,
            height=512,
            steps=steps,
            seed=42 + i,
            output_format="raw",
        ))
        t4 = time.perf_counter()
        gen_time = t4 - t3
        gen_times.append(gen_time)
        logger.info("Generate: %.2fs (result type=%s)", gen_time, type(result[0]).__name__)

        asyncio.run(engine.stop())
        total_time = t4 - t0
        total_times.append(total_time)
        logger.info("Total run %d: %.2fs", i + 1, total_time)
        FusionMemoryGuardian.maybe_purge()

    def _stat(vals):
        return {
            "mean": round(statistics.mean(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    stats = {
        "model": model_name,
        "num_runs": num_runs,
        "steps": steps,
        "engine_start": _stat(start_times),
        "generate": _stat(gen_times),
        "total": _stat(total_times),
    }

    logger.info("=== Results ===")
    logger.info("Engine start: mean=%.2fs", stats["engine_start"]["mean"])
    logger.info("Generate:     mean=%.2fs", stats["generate"]["mean"])
    logger.info("Total:        mean=%.2fs", stats["total"]["mean"])

    logger.info("Stats JSON: %s", json.dumps(stats, indent=2))
    return stats


def bench_video_generation(model_path=None, num_runs=2, steps=4):
    from fusion_mlx.engines.video import VideoGenEngine
    from core.lifecycle import FusionMemoryGuardian

    if model_path is None:
        model_path = os.path.join(MODEL_BASE, "Wan2.1-1.3B")
    model_name = os.path.basename(model_path)

    logger.info("=== Video Generation Benchmark ===")
    logger.info("Model: %s, Runs: %d, Steps: %d", model_name, num_runs, steps)

    start_times = []
    gen_times = []
    total_times = []

    for i in range(num_runs):
        FusionMemoryGuardian.maybe_purge()
        t0 = time.perf_counter()
        logger.info("--- Run %d/%d ---", i + 1, num_runs)

        engine = VideoGenEngine(model_path)
        t1 = time.perf_counter()
        asyncio.run(engine.start())
        t2 = time.perf_counter()
        start_time = t2 - t1
        start_times.append(start_time)
        logger.info("Engine start: %.2fs", start_time)

        t3 = time.perf_counter()
        result = asyncio.run(engine.generate(
            prompt="a cat walking in a garden",
            num_frames=33,
            width=480,
            height=272,
            num_inference_steps=steps,
            seed=42 + i,
        ))
        t4 = time.perf_counter()
        gen_time = t4 - t3
        gen_times.append(gen_time)
        logger.info("Generate: %.2fs (result len=%d)", gen_time, len(result))

        asyncio.run(engine.stop())
        total_time = t4 - t0
        total_times.append(total_time)
        logger.info("Total run %d: %.2fs", i + 1, total_time)
        FusionMemoryGuardian.maybe_purge()

    def _stat(vals):
        return {
            "mean": round(statistics.mean(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    stats = {
        "model": model_name,
        "num_runs": num_runs,
        "steps": steps,
        "engine_start": _stat(start_times),
        "generate": _stat(gen_times),
        "total": _stat(total_times),
    }

    logger.info("=== Video Results ===")
    logger.info("Engine start: mean=%.2fs", stats["engine_start"]["mean"])
    logger.info("Generate:     mean=%.2fs", stats["generate"]["mean"])
    logger.info("Total:        mean=%.2fs", stats["total"]["mean"])

    logger.info("Stats JSON: %s", json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "image"
    if mode == "image":
        bench_image_generation()
    elif mode == "video":
        bench_video_generation()
    elif mode == "both":
        bench_image_generation()
        bench_video_generation()
    else:
        logger.error("Usage: %s [image|video|both]", sys.argv[0])
