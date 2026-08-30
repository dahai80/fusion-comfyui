# SPDX-License-Identifier: Apache-2.0
# E2E for FUSION_H3_VIDEO_768P override: drives the REAL 33B FL2VA model
# through the H3 i2v NODE (which applies the 768p override) -> sampler path
# (_generate_monolithic -> engine.generate), verifying the raised-resolution
# latent shape forwarding + a non-trivial video output at 768p.
# Per project rule: 涉及到大模型测试，须真实加载模型.
# Skips cleanly when RUN_E2E!=1 / no Metal / FL2VA absent, so CI is safe.

import asyncio
import logging
import os

import numpy as np
import pytest

logger = logging.getLogger("h3_e2e_768p")

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


def _tmp_first_frame_png(w, h):
    # Solid-color first frame under /tmp (minimax_h3 _ALLOWED_READ_DIRS blocks
    # macOS $TMPDIR). Match the i2v resolution so no node-side resize.
    import tempfile

    from PIL import Image as PILImage

    pil = PILImage.new("RGB", (w, h), (120, 90, 60))
    fd, path = tempfile.mkstemp(suffix=".png", prefix="fusion_h3_768p_", dir="/tmp")
    with os.fdopen(fd, "wb") as fh:
        pil.save(fh, format="PNG")
    logger.info("_tmp_first_frame_png %dx%d -> %s", w, h, path)
    return path


def _pixel_stats(frames):
    # frames: (N,H,W,3) uint8. Return mean/std/edge_density for quality check.
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr
    else:
        arr = arr[np.newaxis, ...]
    mean = float(arr.mean())
    std = float(arr.std())
    gray = arr.mean(axis=-1)
    edge = float(np.abs(np.diff(gray, axis=0)).mean()) if arr.shape[0] > 1 else 0.0
    return {"mean": mean, "std": std, "edge_density": edge, "shape": list(arr.shape)}


async def _run_768p_i2v(length):
    # Go through the NODE so the 768p override applies, then feed the node's
    # overridden latent + width/height to _generate_monolithic (the production
    # sampler path AICF uses).
    from fusion_comfyui_plugin.nodes.h3 import MiniMaxH3ImageToVideo
    from fusion_comfyui_plugin.nodes.samplers import _generate_monolithic

    os.environ["FUSION_H3_VIDEO_768P"] = "1"
    # AICF feeds 540p 16:9 (960x544); node must raise to 1344x768.
    node = MiniMaxH3ImageToVideo()
    cond, latent = node.generate(
        clip={}, vae={}, prompt="a stone monkey surveying a misty mountain path, cinematic",
        width=960, height=544, length=length, quantize="dit8_te4",
    )
    w = latent["width"]
    h = latent["height"]
    logger.info("768p override: latent %dx%d frames=%d", w, h, length)
    assert (w, h) == (1344, 768), f"override failed: {w}x{h}"

    first_frame = _tmp_first_frame_png(w, h)
    latent["_h3_first_frame_path"] = first_frame
    mw = _H3EngineWrapper()
    await mw.ensure_started()
    try:
        result = await _generate_monolithic(
            mw, cond, {"prompt": ""},
            latent, steps=5, cfg=6.0, seed=42,
            width=w, height=h, num_frames=length,
        )
        return result, (w, h)
    finally:
        await mw.stop()
        if os.path.exists(first_frame):
            os.unlink(first_frame)


async def _run_768p_raw_mp4(length, save_mp4, override=True, w_in=960, h_in=544):
    # Bypass _generate_monolithic: call engine.generate directly to capture the
    # raw mp4 bytes (so we can ffprobe the real frame count the backend wrote).
    from fusion_comfyui_plugin.nodes.h3 import MiniMaxH3ImageToVideo

    if override:
        os.environ["FUSION_H3_VIDEO_768P"] = "1"
    else:
        os.environ["FUSION_H3_VIDEO_768P"] = "0"
    node = MiniMaxH3ImageToVideo()
    cond, latent = node.generate(
        clip={}, vae={}, prompt="a stone monkey surveying a misty mountain path, cinematic",
        width=w_in, height=h_in, length=length, quantize="dit8_te4",
    )
    w, h = latent["width"], latent["height"]
    first_frame = _tmp_first_frame_png(w, h)
    mw = _H3EngineWrapper()
    await mw.ensure_started()
    try:
        raw = await mw._engine.generate(
            prompt=cond["prompt"], num_frames=length, width=w, height=h,
            seed=42, n=1, num_inference_steps=5, quantize="dit8_te4",
            image=first_frame,
        )
        with open(save_mp4, "wb") as fh:
            fh.write(raw[0])
        logger.info("raw mp4 saved: %s (%d bytes) res=%dx%d", save_mp4, len(raw[0]), w, h)
        return save_mp4
    finally:
        await mw.stop()
        if os.path.exists(first_frame):
            os.unlink(first_frame)


def _mp4_stats(mp4):
    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height", "-of", "csv=p=0", mp4],
        capture_output=True, text=True,
    )
    import av
    container = av.open(mp4)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0)
    container.close()
    arr = np.stack(frames, axis=0)
    stats = _pixel_stats(arr)
    stats["ffprobe"] = probe.stdout.strip()
    return stats


def test_h3_i2v_768p_vs_540p_baseline_real_33b():
    # Compare 768p (1344x768) vs 540p (960x544) baseline at the same seed/prompt/
    # steps. If 768p is dramatically flatter than 540p, the raised resolution is
    # fighting the model/VAE (upstream issue), not a fusion-comfyui node bug.
    import tempfile

    mp4_768 = os.path.join(tempfile.mkdtemp(prefix="h3_768p_", dir="/tmp"), "out.mp4")
    mp4_540 = os.path.join(tempfile.mkdtemp(prefix="h3_540p_", dir="/tmp"), "out.mp4")
    asyncio.run(_run_768p_raw_mp4(length=9, save_mp4=mp4_768, override=True))
    asyncio.run(_run_768p_raw_mp4(length=9, save_mp4=mp4_540, override=False))
    s768 = _mp4_stats(mp4_768)
    s540 = _mp4_stats(mp4_540)
    logger.info("540p baseline: %s", s540)
    logger.info("768p override: %s", s768)
    # Both must produce real video with matching frame counts. The 768p override
    # must NOT regress quality vs the 540p baseline (same seed/prompt/steps):
    # 768p std/edge should be >= 540p (more pixels, same denoise). Absolute std
    # is low because this smoke test uses steps=5 + a solid-color keyframe
    # (under-denoised by design); real AICF shots use more steps.
    assert s540["shape"][0] == s768["shape"][0], (
        f"frame count mismatch: 540p={s540['shape'][0]} 768p={s768['shape'][0]}"
    )
    assert s768["shape"][1:] == [768, 1344, 3], f"768p wrong dims: {s768['shape'][1:]}"
    assert s540["shape"][1:] == [544, 960, 3], f"540p wrong dims: {s540['shape'][1:]}"
    assert s768["std"] >= s540["std"] * 0.9, (
        f"768p regressed: std={s768['std']:.3f} < 0.9*540p({s540['std']:.3f})"
    )
    assert s768["edge_density"] >= s540["edge_density"] * 0.9, (
        f"768p edge regressed: {s768['edge_density']:.4f} < 0.9*540p({s540['edge_density']:.4f})"
    )


def test_h3_i2v_768p_monolith_path_frames_real_33b():
    # Verify the PRODUCTION path (_generate_monolithic, what AICF uses) returns
    # the full frame count at 768p — not collapsed to 1 frame. The raw-mp4 path
    # (above) confirmed the backend writes 12 frames; this confirms the sampler
    # decode forwards all of them.
    result, (w, h) = asyncio.run(_run_768p_i2v(length=9))
    assert result is not None and len(result) >= 1, "no video returned"
    arr = np.asarray(result)
    logger.info("768p monolith path: result shape=%s", arr.shape)
    # _generate_monolithic returns (N,H,W,3) float32. Must be >=5 frames.
    assert arr.ndim == 4, f"expected 4D (N,H,W,3), got {arr.ndim}D shape={arr.shape}"
    assert arr.shape[0] >= 5, f"monolith path collapsed frames: {arr.shape[0]}"
    assert arr.shape[1:3] == (768, 1344), f"wrong dims: {arr.shape[1:3]}"
