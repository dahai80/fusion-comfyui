import base64
import io
import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.sr")

_MLX_SETTINGS_PATH = os.environ.get(
    "FUSION_MLX_SETTINGS", "~/.fusion-mlx/settings.json"
)


def _resolve_mlx_api_key():
    # Mirror nodes/voice.py:_resolve_mlx_api_key — the SR endpoint shares the
    # same verify_api_key dependency as every other fusion-mlx route. A
    # key-configured server (FUSION_ALLOW_ANONYMOUS unset) rejects a headerless
    # request with 401, so real-mode SR MUST authenticate.
    key = os.environ.get("FUSION_MLX_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(os.path.expanduser(_MLX_SETTINGS_PATH)) as f:
            data = json.load(f)
        k = data.get("auth", {}).get("api_key", "")
        if k:
            return k
    except Exception as e:
        logger.debug("sr: read mlx settings api_key failed: %s", e)
    return ""


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
    import httpx
    from PIL import Image

    n, h, w, _ = images.shape
    if n == 0:
        return np.empty((0, 0, 0, 3), dtype=np.float32)
    url = endpoint.rstrip('/')
    if not url.endswith("/v1/images/super-resolution"):
        url = f"{url}/v1/images/super-resolution"
    logger.info("sr real: %d frames scale=%d endpoint=%s", n, scale, url)
    headers = {}
    api_key = _resolve_mlx_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key
    else:
        logger.info("sr real: no mlx api_key resolved (anonymous mode expected)")
    max_retries = 3
    t0 = time.time()
    sr_frames = []
    with httpx.Client(timeout=120.0) as client:
        for i in range(n):
            frame = (np.clip(images[i], 0, 1) * 255).astype(np.uint8)[:, :, :3]
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="PNG")
            png = buf.getvalue()
            ft0 = time.time()
            resp = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = client.post(
                        url,
                        headers=headers,
                        files={"image": (f"frame_{i}.png", png, "image/png")},
                        data={"scale": str(scale), "tile_size": "512"},
                    )
                except (httpx.TransportError, httpx.RequestError) as e:
                    logger.warning("sr real: frame %d attempt %d request failed: %s",
                                   i, attempt, e)
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"SR frame {i} failed after {max_retries} attempts "
                            f"({endpoint}): {e}") from e
                    time.sleep(2 * attempt)
                    continue
                if resp.status_code == 200:
                    break
                transient = resp.status_code >= 500
                logger.warning("sr real: frame %d attempt %d HTTP %s: %s",
                               i, attempt, resp.status_code, resp.text[:200])
                if not transient or attempt >= max_retries:
                    raise RuntimeError(
                        f"SR frame {i} HTTP {resp.status_code} ({endpoint}): "
                        f"{resp.text[:200]}")
                time.sleep(2 * attempt)
            if resp is None or resp.status_code != 200:
                raise RuntimeError(
                    f"SR frame {i}: no successful response from {endpoint}")
            body = resp.json()
            sr_bytes = base64.b64decode(body["image_b64"])
            if not sr_bytes:
                raise RuntimeError(
                    f"SR frame {i}: empty image_b64 from {endpoint}")
            sr_arr = np.array(Image.open(io.BytesIO(sr_bytes)).convert("RGB"),
                              dtype=np.float32) / 255.0
            sr_frames.append(sr_arr)
            logger.info("sr real: frame %d/%d in=%dx%d out=%dx%d elapsed=%.2fs",
                        i + 1, n, w, h, sr_arr.shape[1], sr_arr.shape[0],
                        time.time() - ft0)
    logger.info("sr real: done %d frames total elapsed=%.2fs",
                n, time.time() - t0)
    return np.stack(sr_frames, axis=0).astype(np.float32)
