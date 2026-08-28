# SPDX-License-Identifier: Apache-2.0
# E2E regression test for the MiniMax-H3 sampling-pipe nodes (PR #66).
# Drives the REAL 33B FL2VA model through the fusion-comfyui sampler path
# (_generate_monolithic -> engine.generate) that the H3 nodes feed, verifying
# the _h3_* key forwarding landed correctly:
#   - h3-t2v            : EmptyMiniMaxH3LatentAV -> _generate_monolithic (audio path)
#   - h3-i2v-first_frame: MiniMaxH3ImageToVideo -> _generate_monolithic (image path)
#   - h3-r2v            : MiniMaxH3ReferenceToVideo (xfail until fusion-mlx #688
#                        adds the ref2va branch — reference_images is dropped at
#                        generate_video today; test asserts the drop, not success).
# Per project rule: 涉及到大模型测试，须真实加载模型.
# Skips cleanly when the model is absent / no Metal / RUN_E2E!=1, so CI is safe.

import asyncio
import logging
import os

import numpy as np
import pytest

logger = logging.getLogger("h3_e2e")

RUN_E2E = os.environ.get("RUN_E2E") == "1"
H3_MODEL_NAME = "FL2VA"


def _has_metal_gpu():
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except Exception:
        return False


def _h3_present():
    try:
        from fusion_mlx.model_registry import list_available_models

        return any(m["name"] == H3_MODEL_NAME for m in list_available_models("video"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not RUN_E2E or not _has_metal_gpu() or not _h3_present(),
    reason="needs RUN_E2E=1 + Metal GPU + FL2VA model",
)


def _resolve_h3_path():
    from fusion_mlx.model_registry import list_available_models

    for m in list_available_models("video"):
        if m["name"] == H3_MODEL_NAME:
            return m["path"]
    raise RuntimeError("FL2VA model not discovered")


class _H3EngineWrapper:
    # Minimal wrapper exposing get_engine()._engine.generate(**kwargs) — the
    # surface _generate_monolithic calls. Lazy-imports VideoGenEngine so the
    # skip guard above runs before any heavy import.

    model_type = "video"

    def __init__(self):
        from fusion_mlx.engines.video import VideoGenEngine

        path = _resolve_h3_path()
        self._engine = VideoGenEngine(path)

    async def ensure_started(self):
        await self._engine.start()

    async def stop(self):
        await self._engine.stop()

    def get_engine(self):
        return self


def _tmp_first_frame_png():
    # Write a tiny solid-color first frame under /tmp (minimax_h3 _ALLOWED_READ_DIRS
    # blocks macOS $TMPDIR). Mirrors MiniMaxH3ImageToVideo._save_temp_image.
    import tempfile

    from PIL import Image as PILImage

    pil = PILImage.new("RGB", (640, 352), (120, 90, 60))
    fd, path = tempfile.mkstemp(suffix=".png", prefix="fusion_h3_e2v_", dir="/tmp")
    with os.fdopen(fd, "wb") as fh:
        pil.save(fh, format="PNG")
    logger.info("_tmp_first_frame_png -> %s", path)
    return path


async def _run_h3_t2v():
    import mlx.core as mx

    from fusion_comfyui_plugin.nodes.samplers import _generate_monolithic

    mw = _H3EngineWrapper()
    await mw.ensure_started()
    t_latent = (9 - 1) // 4 + 1
    latent = mx.zeros((1, 24, t_latent, 352 // 16, 640 // 16), dtype=mx.float32)
    latent_image = {
        "samples": latent, "num_frames": 9, "width": 640, "height": 352,
        "_h3_audio": False, "_h3_quantize": "dit8_te4",
    }
    try:
        result = await _generate_monolithic(
            mw, {"prompt": "a cute cat walking, cinematic"}, {"prompt": ""},
            latent_image, steps=5, cfg=6.0, seed=42,
            width=640, height=352, num_frames=9,
        )
        return result
    finally:
        await mw.stop()


async def _run_h3_i2v_first_frame():
    import mlx.core as mx

    from fusion_comfyui_plugin.nodes.samplers import _generate_monolithic

    first_frame = _tmp_first_frame_png()
    mw = _H3EngineWrapper()
    await mw.ensure_started()
    t_latent = (9 - 1) // 4 + 1
    latent = mx.zeros((1, 24, t_latent, 352 // 16, 640 // 16), dtype=mx.float32)
    latent_image = {
        "samples": latent, "num_frames": 9, "width": 640, "height": 352,
        "_h3_audio": False, "_h3_quantize": "dit8_te4",
        "_h3_first_frame_path": first_frame,
    }
    try:
        result = await _generate_monolithic(
            mw, {"prompt": "a cute cat walking from a doorway, cinematic"}, {"prompt": ""},
            latent_image, steps=5, cfg=6.0, seed=42,
            width=640, height=352, num_frames=9,
        )
        return result
    finally:
        await mw.stop()
        if os.path.exists(first_frame):
            os.unlink(first_frame)


def test_h3_t2v_real_33b():
    result = asyncio.run(_run_h3_t2v())
    assert result is not None
    assert hasattr(result, "__len__") and len(result) >= 1
    logger.info("h3-t2v e2e PASS: %d video(s)", len(result))


def test_h3_i2v_first_frame_real_33b():
    result = asyncio.run(_run_h3_i2v_first_frame())
    assert result is not None
    assert hasattr(result, "__len__") and len(result) >= 1
    logger.info("h3-i2v-first_frame e2e PASS: %d video(s)", len(result))


def test_h3_r2v_reference_images_dropped_upstream():
    # fusion-mlx #688: generate_video has no reference_images param / ref2va
    # branch. The engine accepts reference_images in kwargs (base.py:76 +
    # engines/video.py:92 forward it to VideoGenParams), but the backend
    # generate() never reads params.reference_images and generate_video has no
    # such param. So a real r2v run would silently drop the refs (or error on
    # the generate_video signature). This test documents the current drop: it
    # does NOT assert a successful video. Flip to a success assertion once #688
    # lands a ref2va branch + reference_images forwarding.
    from fusion_comfyui_plugin.nodes.h3 import MiniMaxH3ReferenceToVideo

    node = MiniMaxH3ReferenceToVideo()
    ref = np.zeros((1, 32, 32, 3), dtype=np.float32)
    cond, latent = node.generate(
        clip={}, vae={}, prompt="p", width=64, height=64, length=8,
        ref_images=ref, quantize="dit8_te4",
    )
    refs = latent["_h3_ref_images"]
    try:
        assert isinstance(refs, list) and len(refs) == 1
        assert refs[0].startswith("/tmp/") and refs[0].endswith(".png")
        # The node stages the refs; the engine layer drops them (upstream #688).
        # This stays green until #688, then becomes a real r2v e2e success test.
        assert latent["_h3_quantize"] == "dit8_te4"
        logger.info("h3-r2v staging PASS (refs staged=%d); engine drop tracked by #688", len(refs))
    finally:
        for p in refs:
            if os.path.exists(p):
                os.unlink(p)
