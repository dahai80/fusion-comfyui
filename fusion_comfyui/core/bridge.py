import logging

import numpy as np

logger = logging.getLogger("fusion_comfyui.bridge")


def to_mlx_array(data):
    import mlx.core as mx

    if isinstance(data, mx.array):
        return data

    if hasattr(data, "__array_interface__"):
        arr = np.asarray(data)
    elif hasattr(data, "numpy"):
        arr = data.numpy()
    elif hasattr(data, "cpu") and hasattr(data, "numpy"):
        arr = data.cpu().numpy()
    else:
        arr = np.array(data)

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    result = mx.array(arr)
    logger.debug("to_mlx_array: shape=%s dtype=%s", result.shape, result.dtype)
    return result


def to_numpy(data) -> np.ndarray:
    import mlx.core as mx

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, mx.array):
        mx.eval(data)
        return np.asarray(data)

    if hasattr(data, "cpu") and hasattr(data, "numpy"):
        return data.cpu().numpy()

    return np.asarray(data)


def to_image_array(data) -> np.ndarray:
    raw = to_numpy(data)
    if raw.dtype != np.float32:
        raw = raw.astype(np.float32)

    if raw.ndim == 5:
        B, T, C, H, W = raw.shape
        raw = raw.reshape(B * T, C, H, W)

    if raw.ndim == 4 and raw.shape[1] <= 4 and raw.shape[3] > 4:
        raw = raw.transpose(0, 2, 3, 1)

    if raw.ndim == 3 and raw.shape[0] <= 4:
        raw = raw.transpose(1, 2, 0)[np.newaxis, ...]

    if raw.max() > 1.0:
        raw = raw / 255.0

    logger.debug("to_image_array: shape=%s dtype=%s", raw.shape, raw.dtype)
    return raw


def to_image_tensor(data):
    arr = to_image_array(data)
    arr = np.ascontiguousarray(arr)
    logger.debug("to_image_tensor: shape=%s dtype=%s", arr.shape, arr.dtype)
    return arr


to_image_numpy = to_image_tensor


def to_mask_numpy(data):
    raw = to_numpy(data)
    if raw.dtype != np.float32:
        raw = raw.astype(np.float32)
    if raw.ndim == 4 and raw.shape[-1] == 1:
        raw = raw[:, :, :, 0]
    if raw.ndim == 2:
        raw = raw[np.newaxis, ...]
    logger.debug("to_mask_numpy: shape=%s dtype=%s", raw.shape, raw.dtype)
    return raw
