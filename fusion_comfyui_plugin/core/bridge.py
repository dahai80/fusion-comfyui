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
    # ComfyUI v0.28 SaveVideo/CreateVideo/SaveImage expect a torch IMAGE
    # tensor (call .clamp().byte().cpu() on each frame). Our MLX decode path
    # produces numpy; wrap to a CPU torch tensor to satisfy the IMAGE contract.
    # torch is only a CPU dtype wrapper here (inference stays on MLX/Metal).
    import torch

    arr = to_image_array(data)
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
    logger.debug("to_image_tensor: shape=%s dtype=%s", tuple(tensor.shape), tensor.dtype)
    return tensor
