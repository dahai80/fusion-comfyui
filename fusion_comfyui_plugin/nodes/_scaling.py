# fusion_comfyui_plugin/nodes/_scaling.py
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("fusion_comfyui.scaling")


def _to_uint8(arr):
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def lanczos(samples, width, height):
    # Grayscale squeezed in native; here samples is NCHW float32 [B,C,H,W].
    n, c, h, w = samples.shape
    out = np.empty((n, c, height, width), dtype=np.float32)
    for i in range(n):
        # NCHW -> HWC uint8 for PIL
        frame = np.transpose(samples[i], (1, 2, 0))
        if c == 1:
            frame = frame[:, :, 0]
        img = Image.fromarray(_to_uint8(frame))
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        resized = np.array(img).astype(np.float32) / 255.0
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        out[i] = np.transpose(resized, (2, 0, 1))
    logger.debug("lanczos: %s -> (%d,%d)", samples.shape, height, width)
    return out


def _slerp(b1, b2, r):
    # b1, b2: flat [..., C]; r: [..., 1]
    b1_norms = np.linalg.norm(b1, axis=-1, keepdims=True)
    b2_norms = np.linalg.norm(b2, axis=-1, keepdims=True)
    b1_normed = np.divide(b1, b1_norms, out=np.zeros_like(b1), where=b1_norms != 0)
    b2_normed = np.divide(b2, b2_norms, out=np.zeros_like(b2), where=b2_norms != 0)
    dot = np.clip((b1_normed * b2_normed).sum(axis=-1, keepdims=True), -1.0, 1.0)
    omega = np.arccos(dot)
    so = np.sin(omega)
    so_safe = np.where(so == 0.0, 1.0, so)
    r_flat = r
    res = (np.sin((1.0 - r_flat) * omega) / so_safe) * b1_normed + \
          (np.sin(r_flat * omega) / so_safe) * b2_normed
    res = res * (b1_norms * (1.0 - r_flat) + b2_norms * r_flat)
    same = dot > 1 - 1e-5
    polar = dot < 1e-5 - 1
    res = np.where(same, b1, res)
    res = np.where(polar, b1 * (1.0 - r_flat) + b2 * r_flat, res)
    return res


def _bilinear_data(length_old, length_new):
    coords_new = np.linspace(0, length_old - 1, length_new, dtype=np.float32).reshape(1, 1, 1, -1)
    ratios = (coords_new - np.floor(coords_new)).astype(np.float32)
    coords_1 = np.floor(coords_new).astype(np.int64)
    coords_2 = np.minimum(coords_1 + 1, length_old - 1)
    return ratios, coords_1, coords_2


def bislerp(samples, width, height):
    orig_dtype = samples.dtype
    samples = samples.astype(np.float32)
    n, c, h, w = samples.shape

    # width pass
    ratios, c1, c2 = _bilinear_data(w, width)
    c1 = np.broadcast_to(c1, (n, c, h, width))
    c2 = np.broadcast_to(c2, (n, c, h, width))
    ratios = np.broadcast_to(ratios, (n, 1, h, width))
    pass_1 = np.take_along_axis(samples, c1, axis=-1)
    pass_2 = np.take_along_axis(samples, c2, axis=-1)
    p1 = pass_1.reshape(-1, c)
    p2 = pass_2.reshape(-1, c)
    r = ratios.reshape(-1, 1)
    result = _slerp(p1, p2, r).reshape(n, c, h, width)

    # height pass
    ratios, c1, c2 = _bilinear_data(h, height)
    c1 = np.broadcast_to(c1.reshape(1, 1, -1, 1), (n, c, height, width))
    c2 = np.broadcast_to(c2.reshape(1, 1, -1, 1), (n, c, height, width))
    ratios = np.broadcast_to(ratios.reshape(1, 1, -1, 1), (n, 1, height, width))
    pass_1 = np.take_along_axis(result, c1, axis=-2)
    pass_2 = np.take_along_axis(result, c2, axis=-2)
    p1 = pass_1.reshape(-1, c)
    p2 = pass_2.reshape(-1, c)
    r = ratios.reshape(-1, 1)
    result = _slerp(p1, p2, r).reshape(n, c, height, width)
    logger.debug("bislerp: %s -> (%d,%d)", samples.shape, height, width)
    return result.astype(orig_dtype)


def _resize_pil(samples, width, height, mode):
    n, c, h, w = samples.shape
    out = np.empty((n, c, height, width), dtype=np.float32)
    resample = {
        "nearest-exact": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "area": Image.Resampling.BOX,
    }.get(mode, Image.Resampling.BILINEAR)
    for i in range(n):
        frame = np.transpose(samples[i], (1, 2, 0))
        if c == 1:
            frame = frame[:, :, 0]
        img = Image.fromarray(_to_uint8(frame))
        img = img.resize((width, height), resample)
        resized = np.array(img).astype(np.float32) / 255.0
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        out[i] = np.transpose(resized, (2, 0, 1))
    return out


def common_upscale(samples, width, height, upscale_method, crop):
    orig_shape = tuple(samples.shape)
    if len(orig_shape) > 4:
        samples = samples.reshape(
            samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1]
        )
        samples = np.transpose(samples, (0, 2, 1, 3, 4))
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
    if crop == "center":
        old_w = samples.shape[-1]
        old_h = samples.shape[-2]
        old_aspect = old_w / old_h
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)
        samples = samples[:, :, y:old_h - y * 2, x:old_w - x * 2]

    if upscale_method == "bislerp":
        out = bislerp(samples, width, height)
    elif upscale_method == "lanczos":
        out = lanczos(samples, width, height)
    else:
        out = _resize_pil(samples, width, height, upscale_method)

    if len(orig_shape) == 4:
        logger.debug("common_upscale: %s -> %s", orig_shape, out.shape)
        return out
    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    out = np.transpose(out, (0, 2, 1, 3, 4)).reshape(orig_shape[:-2] + (height, width))
    logger.debug("common_upscale 5d: %s -> %s", orig_shape, out.shape)
    return out
