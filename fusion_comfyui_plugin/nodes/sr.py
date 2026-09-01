import logging

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.sr")


def _mock_resize(images: np.ndarray, scale: int) -> np.ndarray:
    from PIL import Image
    n, h, w, c = images.shape
    out = np.empty((n, h * scale, w * scale, c), dtype=np.float32)
    for i in range(n):
        arr = (np.clip(images[i], 0, 1) * 255).astype(np.uint8)[:, :, :3]
        pil = Image.fromarray(arr).resize((w * scale, h * scale), Image.LANCZOS)
        out[i] = np.array(pil).astype(np.float32) / 255.0
    logger.info("sr mock: %d frames %dx%d -> %dx%d (scale=%d)",
                n, w, h, w * scale, h * scale, scale)
    return out


def super_resolve_frames(images, mode="0", scale=4,
                         endpoint="http://127.0.0.1:11434/v1/images/super-resolution"):
    if mode == "0" or not mode:
        return images
    if mode == "mock":
        return _mock_resize(images, scale)
    if mode == "1":
        return _real_sr(images, scale, endpoint)
    raise ValueError(f"unknown FUSION_H3_SR mode: {mode!r}")


def _real_sr(images, scale, endpoint):
    raise NotImplementedError("real SR implemented in Task 2")
