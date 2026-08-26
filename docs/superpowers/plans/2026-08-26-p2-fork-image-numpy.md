# P2: Fork ComfyUI IMAGE Type to numpy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every direct `import torch` from `fusion_comfyui/` and `fusion_comfyui_plugin/`; make ComfyUI `IMAGE`/`MASK` types numpy arrays (NHWC float32, [0,1]).

**Architecture:** `fusion_comfyui.core.bridge` becomes the single numpy IMAGE/MASK conversion seam (drops torch). 6 native pure-transform nodes (ImageScale, ImageScaleBy, ImageBatch, EmptyImage, ImagePadForOutpaint, LoadImageMask) are overridden with numpy/PIL implementations in a new `image_transform.py`. 7 native dead-path nodes (controlnet/inpaint/painter/qwen + ConditioningSetMask) are stubbed to `NotImplementedError` in a new `_deadpath_stubs.py`. Scaling kernels (`common_upscale`/`bislerp`/`lanczos`) ported from `comfy/utils.py` torch to numpy/PIL in a new `_scaling.py`, fixing the already-broken `LatentUpscale` true-latent path. ip_adapter `.bin` loader deleted (safetensors only). All overrides registered in `__init__.py` via the existing monkey-patch mechanism.

**Tech Stack:** numpy, Pillow (PIL), MLX — no torch. pytest, unittest.mock.

**Spec:** `docs/superpowers/specs/2026-08-26-p2-fork-image-type-numpy-design.md`

## Global Constraints

- **No `import torch`** anywhere in `fusion_comfyui/` or `fusion_comfyui_plugin/` after this phase (exit criterion: `grep -rn "import torch" fusion_comfyui/ fusion_comfyui_plugin/` returns 0).
- **IMAGE** = `np.ndarray` float32, layout NHWC `[B,H,W,C]`, range `[0,1]`.
- **MASK** = `np.ndarray` float32, layout `[B,H,W]`, range `[0,1]`.
- Indentation: 4-space multiples. No docstrings. Every function logs (debug or info).
- Match existing style: module-level `logger = logging.getLogger(...)`, bare `from nodes.x import Y` in tests (sys.path holds plugin dir).
- Tests live in `fusion_comfyui_plugin/tests/`. Existing 485 tests must stay green. ruff clean (`ruff check .`).
- Upstream rule: only modify `fusion-comfyui` dir. ComfyUI/ is vendored — do NOT edit it; we override via the plugin monkey-patch mechanism instead.

---

## File Structure

**Create:**
- `fusion_comfyui_plugin/nodes/_scaling.py` — numpy/PIL reimplementation of `common_upscale`, `lanczos`, `bislerp`. Replaces `comfy.utils.common_upscale` calls in plugin code.
- `fusion_comfyui_plugin/nodes/image_transform.py` — 6 pure-numpy IMAGE/MASK node overrides (ImageScale, ImageScaleBy, ImageBatch, EmptyImage, ImagePadForOutpaint, LoadImageMask).
- `fusion_comfyui_plugin/nodes/_deadpath_stubs.py` — 7 NotImplementedError stubs preserving native INPUT_TYPES via inheritance.
- `fusion_comfyui_plugin/tests/test_scaling.py` — scaling kernel unit + parity tests.
- `fusion_comfyui_plugin/tests/test_image_transform.py` — transform node tests.
- `fusion_comfyui_plugin/tests/test_deadpath_stubs.py` — stub tests.

**Modify:**
- `fusion_comfyui/core/bridge.py` — `to_image_tensor` returns numpy (drop torch); add `to_mask_numpy`.
- `fusion_comfyui_plugin/nodes/image.py` — `LoadImage.load_image` mask via `to_mask_numpy` (drop `import torch`).
- `fusion_comfyui_plugin/nodes/ip_adapter.py` — delete `_load_torch_ip_adapter` + `.bin`/`.pt`/`.ckpt` call sites; log safetensors-only message.
- `fusion_comfyui_plugin/nodes/samplers.py` — `LatentUpscale` true-latent path uses `_scaling.common_upscale` instead of `comfy.utils.common_upscale`.
- `fusion_comfyui_plugin/__init__.py` — register 13 new overrides in `NODE_CLASS_MAPPINGS`, `_native_overrides`, `NODE_DISPLAY_NAME_MAPPINGS`.
- `fusion_comfyui_plugin/tests/test_bridge.py` — update `to_image_tensor` assertions (numpy, not torch).
- `fusion_comfyui_plugin/tests/test_image.py` — update `test_load_image` to assert numpy IMAGE/MASK.
- `fusion_comfyui_plugin/tests/test_ip_adapter.py` — drop `.bin` test or assert it returns None with message.

---

## Task Dependency Order

Task 1 (scaling kernels) is the foundation — Task 2 (transforms) and Task 4 (latent path) both import from it. Task 3 (bridge) is independent. Tasks 5-6 (ip_adapter, registry) depend on 2-3. Task 7 (test updates) depends on 3. Task 8 (verification) is last.

---

### Task 1: numpy/PIL scaling kernels (`_scaling.py`)

**Files:**
- Create: `fusion_comfyui_plugin/nodes/_scaling.py`
- Test: `fusion_comfyui_plugin/tests/test_scaling.py`

**Interfaces:**
- Produces:
  - `common_upscale(samples: np.ndarray, width: int, height: int, upscale_method: str, crop: str) -> np.ndarray` — samples is NCHW float32 `[B,C,H,W]` (4D) or `[B,T,C,H,W]` (5D, reshaped internally); returns same layout scaled to `(height, width)` spatial dims.
  - `lanczos(samples: np.ndarray, width: int, height: int) -> np.ndarray` — NCHW `[B,C,H,W]` → scaled NCHW.
  - `bislerp(samples: np.ndarray, width: int, height: int) -> np.ndarray` — NCHW → scaled NCHW.

- [ ] **Step 1: Write the failing tests**

```python
# fusion_comfyui_plugin/tests/test_scaling.py
import numpy as np


def _img(batch=2, c=3, h=64, w=64):
    rng = np.random.default_rng(42)
    return rng.random((batch, c, h, w), dtype=np.float32)


class TestLanczos:
    def test_shape_4d(self):
        from nodes._scaling import lanczos
        out = lanczos(_img(), 128, 96)
        assert out.shape == (2, 3, 96, 128)
        assert out.dtype == np.float32

    def test_range_preserved(self):
        from nodes._scaling import lanczos
        src = np.zeros((1, 3, 32, 32), dtype=np.float32)
        out = lanczos(src, 64, 64)
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestBislerp:
    def test_shape_4d(self):
        from nodes._scaling import bislerp
        out = bislerp(_img(), 128, 96)
        assert out.shape == (2, 3, 96, 128)
        assert out.dtype == np.float32

    def test_identity_upscale(self):
        from nodes._scaling import bislerp
        src = _img(1, 3, 16, 16)
        out = bislerp(src, 16, 16)
        assert np.allclose(out, src, atol=1e-4)


class TestCommonUpscale:
    def test_disabled_crop_grows(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 32, 32), 64, 64, "bilinear", "disabled")
        assert out.shape == (1, 3, 64, 64)

    def test_center_crop_aspect(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 64, 32), 32, 32, "bilinear", "center")
        assert out.shape == (1, 3, 32, 32)

    def test_5d_video(self):
        from nodes._scaling import common_upscale
        src = _img(1, 4, 32, 32)[..., None, :, :]
        src = np.transpose(src, (0, 4, 1, 2, 3))  # B,T,C,H,W
        src = np.ascontiguousarray(src)
        out = common_upscale(src, 64, 64, "bilinear", "disabled")
        assert out.shape == (1, 4, 1, 64, 64)

    def test_method_lanczos(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 32, 32), 64, 64, "lanczos", "disabled")
        assert out.shape == (1, 3, 64, 64)

    def test_method_bislerp(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 32, 32), 64, 64, "bislerp", "disabled")
        assert out.shape == (1, 3, 64, 64)

    def test_nearest_exact(self):
        from nodes._scaling import common_upscale
        out = common_upscale(_img(1, 3, 32, 32), 64, 64, "nearest-exact", "disabled")
        assert out.shape == (1, 3, 64, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_scaling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nodes._scaling'`

- [ ] **Step 3: Implement `_scaling.py`**

```python
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
    c = b1.shape[-1]
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
    coords = np.arange(length_old, dtype=np.float32).reshape(1, 1, 1, -1)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_scaling.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add fusion_comfyui_plugin/nodes/_scaling.py fusion_comfyui_plugin/tests/test_scaling.py
git commit -m "feat: numpy/PIL scaling kernels (common_upscale/lanczos/bislerp) for P2"
```

---

### Task 2: numpy image transforms (`image_transform.py`)

**Files:**
- Create: `fusion_comfyui_plugin/nodes/image_transform.py`
- Test: `fusion_comfyui_plugin/tests/test_image_transform.py`

**Interfaces:**
- Consumes: `from nodes._scaling import common_upscale`
- Produces: 6 node classes — `ImageScale`, `ImageScaleBy`, `ImageBatch`, `EmptyImage`, `ImagePadForOutpaint`, `LoadImageMask`. Each has native-compatible `INPUT_TYPES`/`RETURN_TYPES`/`FUNCTION`/`CATEGORY` and operates on numpy IMAGE `[B,H,W,C]` float32.

- [ ] **Step 1: Write the failing tests**

```python
# fusion_comfyui_plugin/tests/test_image_transform.py
import numpy as np


def _img(batch=1, h=32, w=32, c=3):
    rng = np.random.default_rng(7)
    return rng.random((batch, h, w, c), dtype=np.float32)


class TestImageScale:
    def test_upscale(self):
        from nodes.image_transform import ImageScale
        out = ImageScale().upscale(_img(1, 32, 32), "bilinear", 64, 64, "disabled")
        assert out[0].shape == (1, 64, 64, 3)
        assert out[0].dtype == np.float32

    def test_zero_dims_passthrough(self):
        from nodes.image_transform import ImageScale
        src = _img(1, 32, 32)
        out = ImageScale().upscale(src, "bilinear", 0, 0, "disabled")
        assert out[0].shape == src.shape


class TestImageScaleBy:
    def test_scale_by(self):
        from nodes.image_transform import ImageScaleBy
        out = ImageScaleBy().upscale(_img(1, 20, 20), "bilinear", 2.0)
        assert out[0].shape == (1, 40, 40, 3)


class TestImageBatch:
    def test_same_channels(self):
        from nodes.image_transform import ImageBatch
        a = _img(2, 16, 16, 3)
        b = _img(3, 16, 16, 3)
        out = ImageBatch().batch(a, b)
        assert out[0].shape == (5, 16, 16, 3)

    def test_channel_pad(self):
        from nodes.image_transform import ImageBatch
        a = _img(1, 16, 16, 3)
        b = _img(1, 16, 16, 4)
        out = ImageBatch().batch(a, b)
        assert out[0].shape == (2, 16, 16, 4)


class TestEmptyImage:
    def test_generate(self):
        from nodes.image_transform import EmptyImage
        out = EmptyImage().generate(64, 64, batch_size=2, color=0xFF0000)
        assert out[0].shape == (2, 64, 64, 3)
        assert out[0].dtype == np.float32
        assert np.allclose(out[0][0, 0, 0], [1.0, 0.0, 0.0])


class TestImagePadForOutpaint:
    def test_expand(self):
        from nodes.image_transform import ImagePadForOutpaint
        src = _img(1, 16, 16, 3)
        img, mask = ImagePadForOutpaint().expand_image(src, 4, 4, 4, 4, 0)
        assert img.shape == (1, 24, 24, 3)
        assert mask.shape == (1, 24, 24)


class TestLoadImageMask:
    def test_input_types(self):
        from nodes.image_transform import LoadImageMask
        with __import__("unittest.mock").mock.patch(
            "folder_paths.get_input_directory", return_value="/tmp"
        ):
            inputs = LoadImageMask.INPUT_TYPES()
            assert "channel" in inputs["required"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_image_transform.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nodes.image_transform'`

- [ ] **Step 3: Implement `image_transform.py`**

```python
# fusion_comfyui_plugin/nodes/image_transform.py
import logging

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.image_transform")


class ImageScale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                             "width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1}),
                             "height": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1}),
                             "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            logger.debug("ImageScale: passthrough %s", image.shape)
            return (image,)
        samples = np.transpose(image, (0, 3, 1, 2))
        if width == 0:
            width = max(1, round(samples.shape[3] * height / samples.shape[2]))
        elif height == 0:
            height = max(1, round(samples.shape[2] * width / samples.shape[3]))
        from nodes._scaling import common_upscale
        s = common_upscale(samples, width, height, upscale_method, crop)
        s = np.transpose(s, (0, 2, 3, 1))
        logger.info("ImageScale: %s -> %s", image.shape, s.shape)
        return (s,)


class ImageScaleBy:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                             "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, scale_by):
        samples = np.transpose(image, (0, 3, 1, 2))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        from nodes._scaling import common_upscale
        s = common_upscale(samples, width, height, upscale_method, "disabled")
        s = np.transpose(s, (0, 2, 3, 1))
        logger.info("ImageScaleBy: %s -> %s", image.shape, s.shape)
        return (s,)


class ImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",), "image2": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"
    CATEGORY = "image/batch"
    DEPRECATED = True

    def batch(self, image1, image2):
        if image1.shape[-1] != image2.shape[-1]:
            if image1.shape[-1] > image2.shape[-1]:
                image2 = np.pad(image2, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0)
            else:
                image1 = np.pad(image1, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0)
        if image1.shape[1:] != image2.shape[1:]:
            s2 = np.transpose(image2, (0, 3, 1, 2))
            from nodes._scaling import common_upscale
            s2 = common_upscale(s2, image1.shape[2], image1.shape[1], "bilinear", "center")
            image2 = np.transpose(s2, (0, 2, 3, 1))
        s = np.concatenate((image1, image2), axis=0)
        logger.info("ImageBatch: %s + %s -> %s", image1.shape, image2.shape, s.shape)
        return (s,)


class EmptyImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                             "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image"

    def generate(self, width, height, batch_size=1, color=0):
        r = ((color >> 16) & 0xFF) / 0xFF
        g = ((color >> 8) & 0xFF) / 0xFF
        b = (color & 0xFF) / 0xFF
        img = np.full((batch_size, height, width, 3), [r, g, b], dtype=np.float32)
        logger.info("EmptyImage: %s color=#%06X", img.shape, color)
        return (img,)


class ImagePadForOutpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                             "top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                             "right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                             "bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                             "feathering": ("INT", {"default": 40, "min": 0, "max": 8192, "step": 1, "advanced": True}),}}
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    CATEGORY = "image/transform"

    def expand_image(self, image, left, top, right, bottom, feathering):
        d1, d2, d3, d4 = image.shape
        new_image = np.ones((d1, d2 + top + bottom, d3 + left + right, d4), dtype=np.float32) * 0.5
        new_image[:, top:top + d2, left:left + d3, :] = image
        mask = np.ones((d2 + top + bottom, d3 + left + right), dtype=np.float32)
        t = np.zeros((d2, d3), dtype=np.float32)
        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:
            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2
                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3
                    d = min(dt, db, dl, dr)
                    if d >= feathering:
                        continue
                    v = (feathering - d) / feathering
                    t[i, j] = v * v
        mask[top:top + d2, left:left + d3] = t
        logger.info("ImagePadForOutpaint: %s -> img %s mask %s", image.shape, new_image.shape, mask.shape)
        return (new_image, mask[np.newaxis, ...])


class LoadImageMask:
    _color_channels = ["alpha", "red", "green", "blue"]

    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        import os
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required": {"image": (sorted(files), {"image_upload": True}), "channel": (s._color_channels,)}}

    CATEGORY = "image"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image_mask"

    def load_image_mask(self, image, channel):
        from nodes.image import LoadImage
        image_arr, mask_arr = LoadImage().load_image(image)
        c = channel[0].upper()
        if c == "A":
            return (mask_arr,)
        channel_idx = {"R": 0, "G": 1, "B": 2}.get(c, 0)
        if channel_idx < image_arr.shape[-1]:
            return (np.ascontiguousarray(image_arr[..., channel_idx]).copy(),)
        empty = np.zeros(image_arr.shape[:-1], dtype=np.float32)
        return (empty,)

    @classmethod
    def IS_CHANGED(s, image, channel):
        from nodes.image import LoadImage
        return LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        import folder_paths
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_image_transform.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add fusion_comfyui_plugin/nodes/image_transform.py fusion_comfyui_plugin/tests/test_image_transform.py
git commit -m "feat: numpy image transform overrides (ImageScale/Batch/Empty/Padding/LoadImageMask) for P2"
```

---

### Task 3: bridge.py — return numpy, drop torch

**Files:**
- Modify: `fusion_comfyui/core/bridge.py:69-79`
- Test: `fusion_comfyui_plugin/tests/test_bridge.py` (update assertions)

**Interfaces:**
- Produces:
  - `to_image_tensor(data) -> np.ndarray` — now returns numpy NHWC float32 [0,1] (alias `to_image_numpy`).
  - `to_mask_numpy(data) -> np.ndarray` — numpy float32 `[B,H,W]` [0,1].

- [ ] **Step 1: Write the failing test (update `test_bridge.py`)**

Add/replace these in `fusion_comfyui_plugin/tests/test_bridge.py` (keep existing tests for `to_numpy`/`to_image_array`/`to_mlx_array`):

```python
import numpy as np


def test_to_image_tensor_returns_numpy():
    from fusion_comfyui.core.bridge import to_image_tensor
    src = np.random.default_rng(1).random((2, 8, 8, 3)).astype(np.float32)
    out = to_image_tensor(src)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (2, 8, 8, 3)


def test_to_mask_numpy():
    from fusion_comfyui.core.bridge import to_mask_numpy
    src = np.random.default_rng(2).random((4, 8, 8)).astype(np.float32)
    out = to_mask_numpy(src)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.ndim == 3


def test_to_image_numpy_alias():
    from fusion_comfyui.core.bridge import to_image_numpy, to_image_tensor
    assert to_image_numpy is to_image_tensor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_bridge.py::test_to_image_tensor_returns_numpy -v`
Expected: FAIL — `to_image_tensor` still returns torch tensor (`isinstance(out, np.ndarray)` is False)

- [ ] **Step 3: Modify `bridge.py`**

Replace `to_image_tensor` (lines 69-79) with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_bridge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fusion_comfyui/core/bridge.py fusion_comfyui_plugin/tests/test_bridge.py
git commit -m "refactor: bridge.to_image_tensor returns numpy, add to_mask_numpy (drop torch)"
```

---

### Task 4: samplers.py — LatentUpscale true-latent path uses numpy scaling

**Files:**
- Modify: `fusion_comfyui_plugin/nodes/samplers.py:616-633`

**Interfaces:**
- Consumes: `from nodes._scaling import common_upscale` (numpy, takes NCHW)
- The latent `arr` here is mx/numpy `[B,T,C,H,W]` or `[B,C,H,W]`. `common_upscale` handles both.

- [ ] **Step 1: Write the failing test**

Append to `fusion_comfyui_plugin/tests/test_samplers.py`:

```python
import numpy as np


def test_latent_upscale_true_latent_numpy():
    from nodes.samplers import LatentUpscale
    latent = {
        "samples": np.random.default_rng(3).random((1, 1, 4, 16, 16)).astype(np.float32),
    }
    out = LatentUpscale().upscale(latent, "bilinear", 128, 128, "disabled")
    assert out[0]["samples"].shape == (1, 1, 4, 16, 16)
    assert isinstance(out[0]["samples"], np.ndarray)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_samplers.py::test_latent_upscale_true_latent_numpy -v`
Expected: FAIL — `AttributeError: 'numpy.ndarray' object has no attribute 'movedim'` (the current `comfy.utils.common_upscale` call)

- [ ] **Step 3: Modify `samplers.py`**

Replace lines 616-633 (the `# True latent path` block through the `return (s,)`):

```python
        # True latent path: mirror native behavior (latent space, /8).
        from nodes._scaling import common_upscale

        if width == 0 and height == 0:
            return (s,)
        if width == 0:
            height = max(64, height)
            width = max(64, round(arr.shape[-1] * height / arr.shape[-2]))
        elif height == 0:
            width = max(64, width)
            height = max(64, round(arr.shape[-2] * width / arr.shape[-1]))
        else:
            width = max(64, width)
            height = max(64, height)
        latent_np = np.asarray(arr)
        s["samples"] = common_upscale(
            latent_np, width // 8, height // 8, upscale_method, crop
        )
        logger.info("LatentUpscale: latent path -> %dx%d", width, height)
        return (s,)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_samplers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fusion_comfyui_plugin/nodes/samplers.py fusion_comfyui_plugin/tests/test_samplers.py
git commit -m "fix: LatentUpscale true-latent path uses numpy common_upscale (was AttributeError)"
```

---

### Task 5: ip_adapter — drop torch `.bin` loader

**Files:**
- Modify: `fusion_comfyui_plugin/nodes/ip_adapter.py:395-472`
- Test: `fusion_comfyui_plugin/tests/test_ip_adapter.py`

**Interfaces:**
- `_load_ip_adapter_file(ip_path)` now: safetensors only (file or dir of `*.safetensors`); `.bin`/`.pt`/`.ckpt` → log + return None. `_load_torch_ip_adapter` deleted.

- [ ] **Step 1: Write the failing test**

Append to `fusion_comfyui_plugin/tests/test_ip_adapter.py`:

```python
import numpy as np
from pathlib import Path
import tempfile


def test_load_ip_adapter_bin_returns_none(tmp_path):
    from nodes.ip_adapter import _load_ip_adapter_file
    bin_file = tmp_path / "ip-adapter.bin"
    bin_file.write_bytes(b"not real safetensors")
    out = _load_ip_adapter_file(bin_file)
    assert out is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_ip_adapter.py::test_load_ip_adapter_bin_returns_none -v`
Expected: FAIL — current code tries `torch.load` and may error differently (or import torch)

- [ ] **Step 3: Modify `ip_adapter.py`**

Replace `_load_ip_adapter_file` (lines 395-425) and delete `_load_torch_ip_adapter` (lines 428-472). New `_load_ip_adapter_file`:

```python
def _load_ip_adapter_file(ip_path):
    if ip_path.is_file():
        ext = ip_path.suffix.lower()
        if ext in (".bin", ".pt", ".ckpt"):
            logger.warning(
                "IP-Adapter %s uses .bin which needs torch; download the .safetensors "
                "from HF mirror https://hf-mirror.com (same model repo)", ip_path,
            )
            return None
        try:
            return mx.load(str(ip_path))
        except Exception as e:
            logger.warning("Failed to load %s as safetensors: %s", ip_path, e)
            return None
    elif ip_path.is_dir():
        safetensors = sorted(glob.glob(str(ip_path / "*.safetensors")))
        if not safetensors:
            bins = sorted(glob.glob(str(ip_path / "*.bin")))
            if bins:
                logger.warning(
                    "IP-Adapter dir %s has only .bin (needs torch); download .safetensors "
                    "from https://hf-mirror.com", ip_path,
                )
            else:
                logger.warning("IP-Adapter weights not found at %s", ip_path)
            return None
        raw = {}
        for sf in safetensors:
            raw.update(mx.load(sf))
        logger.info("Loaded %d IP-Adapter tensors from %d safetensors", len(raw), len(safetensors))
        return raw
    else:
        logger.warning("IP-Adapter path not found: %s", ip_path)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_ip_adapter.py -v`
Expected: PASS (existing safetensors tests still pass; new .bin test passes)

- [ ] **Step 5: Commit**

```bash
git add fusion_comfyui_plugin/nodes/ip_adapter.py fusion_comfyui_plugin/tests/test_ip_adapter.py
git commit -m "refactor: ip_adapter safetensors-only, drop torch .bin loader (P2)"
```

---

### Task 6: Dead-path stubs + registry registration

**Files:**
- Create: `fusion_comfyui_plugin/nodes/_deadpath_stubs.py`
- Test: `fusion_comfyui_plugin/tests/test_deadpath_stubs.py`
- Modify: `fusion_comfyui_plugin/__init__.py` (imports + 3 dicts)

**Interfaces:**
- Produces: 7 stub classes (`ConditioningSetMaskStub`, `VAEEncodeForInpaintStub`, `InpaintModelConditioningStub`, `ControlNetApplyStub`, `ControlNetApplyAdvancedStub`, `PainterNodeStub`, `QwenImageDiffsynthControlnetStub`). Each subclasses the native node (inherits `INPUT_TYPES`/`RETURN_TYPES`) but overrides `FUNCTION`'s target to a method that raises `NotImplementedError`.
- Note: stubs must NOT be imported at module top of `__init__.py` in a way that crashes if native class is absent; import inside the patch loop with try/except (existing pattern guards with `_not_found`).

- [ ] **Step 1: Write the failing tests**

```python
# fusion_comfyui_plugin/tests/test_deadpath_stubs.py
import pytest


STUBS = [
    ("ConditioningSetMaskStub", "regional-mask conditioning is not supported"),
    ("VAEEncodeForInpaintStub", "PyTorch model layer"),
    ("InpaintModelConditioningStub", "PyTorch model layer"),
    ("ControlNetApplyStub", "PyTorch model layer"),
    ("ControlNetApplyAdvancedStub", "PyTorch model layer"),
    ("PainterNodeStub", "PyTorch model layer"),
    ("QwenImageDiffsynthControlnetStub", "PyTorch model layer"),
]


@pytest.mark.parametrize("stub_name,frag", STUBS)
def test_stub_raises(stub_name, frag):
    from nodes import _deadpath_stubs
    cls = getattr(_deadpath_stubs, stub_name)
    inst = cls()
    fn = getattr(inst, cls.FUNCTION)
    with pytest.raises(NotImplementedError) as exc:
        fn(*([None] * (cls.INPUT_TYPES().__len__() if False else 0)))
    assert frag in str(exc.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_deadpath_stubs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nodes._deadpath_stubs'`

- [ ] **Step 3: Implement `_deadpath_stubs.py`**

```python
# fusion_comfyui_plugin/nodes/_deadpath_stubs.py
import logging

logger = logging.getLogger("fusion_comfyui.nodes.deadpath_stubs")


def _stub_factory(native_cls_name, native_module, message):
    import importlib
    try:
        mod = importlib.import_module(native_module)
        native = getattr(mod, native_cls_name)
    except Exception as e:
        logger.warning("deadpath stub: native %s not found (%s)", native_cls_name, e)
        native = object

    class _Stub(native):
        pass

    _Stub.FUNCTION = "stub_run"

    def stub_run(self, *args, **kwargs):
        raise NotImplementedError(message)

    _Stub.stub_run = stub_run
    _Stub.__name__ = native_cls_name + "Stub"
    _Stub.__qualname__ = _Stub.__name__
    return _Stub


ConditioningSetMaskStub = _stub_factory(
    "ConditioningSetMask", "nodes",
    "ConditioningSetMask: regional-mask conditioning is not supported on the fusion-mlx "
    "pipeline (engine has no mask hook); use Fusion* nodes or wait for P3 staged conditioning.",
)
VAEEncodeForInpaintStub = _stub_factory(
    "VAEEncodeForInpaint", "nodes",
    "VAEEncodeForInpaint: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
InpaintModelConditioningStub = _stub_factory(
    "InpaintModelConditioning", "nodes",
    "InpaintModelConditioning: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
ControlNetApplyStub = _stub_factory(
    "ControlNetApply", "nodes",
    "ControlNetApply: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
ControlNetApplyAdvancedStub = _stub_factory(
    "ControlNetApplyAdvanced", "nodes",
    "ControlNetApplyAdvanced: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
PainterNodeStub = _stub_factory(
    "PainterNode", "nodes_painter",
    "PainterNode: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
QwenImageDiffsynthControlnetStub = _stub_factory(
    "QwenImageDiffsynthControlnet", "comfy_extras.nodes_model_patch",
    "QwenImageDiffsynthControlnet: routes into a PyTorch model layer not yet ported to MLX (P5); "
    "use the Fusion* equivalent or wait for the comfy/ core fork.",
)
```

- [ ] **Step 4: Fix the stub test to call with proper args**

The Step 1 test used `cls.INPUT_TYPES().__len__()` hack — replace with a clean no-arg call. Replace the test body's call with:

```python
    fn = getattr(inst, cls.FUNCTION)
    with pytest.raises(NotImplementedError) as exc:
        fn()
    assert frag in str(exc.value)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_deadpath_stubs.py -v`
Expected: PASS (7 stubs raise NotImplementedError with the right message fragment)

- [ ] **Step 6: Register overrides in `__init__.py`**

Add imports near the other node imports (after the existing `from .nodes.X import Y` block, around line 43). Add:

```python
from .nodes.image_transform import (
    ImageScale as _ImageScale,
    ImageScaleBy as _ImageScaleBy,
    ImageBatch as _ImageBatch,
    EmptyImage as _EmptyImage,
    ImagePadForOutpaint as _ImagePadForOutpaint,
    LoadImageMask as _LoadImageMask,
)
from .nodes._deadpath_stubs import (
    ConditioningSetMaskStub,
    VAEEncodeForInpaintStub,
    InpaintModelConditioningStub,
    ControlNetApplyStub,
    ControlNetApplyAdvancedStub,
    PainterNodeStub,
    QwenImageDiffsynthControlnetStub,
)
```

In `NODE_CLASS_MAPPINGS` (add near the Image section, after `PreviewImage`):

```python
    "ImageScale": _ImageScale,
    "ImageScaleBy": _ImageScaleBy,
    "ImageBatch": _ImageBatch,
    "EmptyImage": _EmptyImage,
    "ImagePadForOutpaint": _ImagePadForOutpaint,
    "LoadImageMask": _LoadImageMask,
    "ConditioningSetMask": ConditioningSetMaskStub,
    "VAEEncodeForInpaint": VAEEncodeForInpaintStub,
    "InpaintModelConditioning": InpaintModelConditioningStub,
    "ControlNetApply": ControlNetApplyStub,
    "ControlNetApplyAdvanced": ControlNetApplyAdvancedStub,
    "PainterNode": PainterNodeStub,
    "QwenImageDiffsynthControlnet": QwenImageDiffsynthControlnetStub,
```

Add the same 13 keys to `_native_overrides` dict (so they monkey-patch native `NODE_CLASS_MAPPINGS`).

In `NODE_DISPLAY_NAME_MAPPINGS` add:

```python
    "ImageScale": "Image Scale (fusion-mlx)",
    "ImageScaleBy": "Image Scale By (fusion-mlx)",
    "ImageBatch": "Image Batch (fusion-mlx)",
    "EmptyImage": "Empty Image (fusion-mlx)",
    "ImagePadForOutpaint": "Pad Image for Outpaint (fusion-mlx)",
    "LoadImageMask": "Load Image Mask (fusion-mlx)",
    "ConditioningSetMask": "Set Latent Noise Mask (not on MLX)",
    "VAEEncodeForInpaint": "VAE Encode for Inpaint (not on MLX)",
    "InpaintModelConditioning": "Inpaint Model Conditioning (not on MLX)",
    "ControlNetApply": "Apply ControlNet (not on MLX)",
    "ControlNetApplyAdvanced": "Apply ControlNet Advanced (not on MLX)",
    "PainterNode": "Painter (not on MLX)",
    "QwenImageDiffsynthControlnet": "Qwen Image ControlNet (not on MLX)",
```

- [ ] **Step 7: Verify startup + node count**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && python -c "import sys; sys.path.insert(0,'ComfyUI'); sys.path.insert(0,'fusion_comfyui_plugin'); import fusion_comfyui_plugin; print('mappings:', len(fusion_comfyui_plugin.NODE_CLASS_MAPPINGS))"`
Expected: prints a number; no ImportError. (Note: full ComfyUI startup is verified in Task 8.)

- [ ] **Step 8: Commit**

```bash
git add fusion_comfyui_plugin/nodes/_deadpath_stubs.py fusion_comfyui_plugin/tests/test_deadpath_stubs.py fusion_comfyui_plugin/__init__.py
git commit -m "feat: dead-path stubs (7) + register 13 numpy/stub overrides in __init__ (P2)"
```

---

### Task 7: Update LoadImage + existing tests for numpy IMAGE/MASK

**Files:**
- Modify: `fusion_comfyui_plugin/nodes/image.py:65-67`
- Modify: `fusion_comfyui_plugin/tests/test_image.py:15-24`
- Modify: `fusion_comfyui_plugin/tests/test_ip_adapter.py` (if `.bin` test references torch) — already handled Task 5

**Interfaces:**
- `LoadImage.load_image` now returns `(np.ndarray IMAGE, np.ndarray MASK)` via `to_image_tensor` + `to_mask_numpy`.

- [ ] **Step 1: Update the LoadImage test to assert numpy**

Replace `test_load_image` in `fusion_comfyui_plugin/tests/test_image.py`:

```python
    def test_load_image(self):
        from nodes.image import LoadImage
        tmpdir = tempfile.mkdtemp()
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img_path = os.path.join(tmpdir, "test.png")
        img.save(img_path)
        with patch("folder_paths.get_annotated_filepath", return_value=img_path):
            node = LoadImage()
            result = node.load_image("test.png")
            assert result is not None
            image_t, mask_t = result
            assert isinstance(image_t, np.ndarray)
            assert image_t.shape == (1, 64, 64, 3)
            assert image_t.dtype == np.float32
            assert isinstance(mask_t, np.ndarray)
            assert mask_t.ndim == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_image.py::TestLoadImage::test_load_image -v`
Expected: FAIL — `isinstance(image_t, np.ndarray)` False (still torch)

- [ ] **Step 3: Modify `image.py` LoadImage mask wrapping**

Replace lines 65-67 in `fusion_comfyui_plugin/nodes/image.py`:

```python
        image_t = to_image_tensor(output_image)
        from fusion_comfyui.core.bridge import to_mask_numpy
        mask_t = to_mask_numpy(output_mask)
```

Also remove the now-stale comment on lines 63-64 ("Core IMAGE/MASK consumers ... require torch tensors") and replace with:

```python
        # IMAGE/MASK are numpy NHWC float32 [0,1]; fusion-mlx path is torch-free.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest fusion_comfyui_plugin/tests/test_image.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fusion_comfyui_plugin/nodes/image.py fusion_comfyui_plugin/tests/test_image.py
git commit -m "refactor: LoadImage returns numpy IMAGE/MASK via bridge (drop torch, P2)"
```

---

### Task 8: Verification — grep zero torch, full suite, startup, parity

**Files:**
- No new files; verification only.

- [ ] **Step 1: Verify zero `import torch` in plugin + core**

Run: `cd /Users/dahai/fusion/fusion-comfyui && grep -rn "import torch" fusion_comfyui/ fusion_comfyui_plugin/ || echo "ZERO torch imports"`
Expected: `ZERO torch imports`

If any remain, fix that file (likely a test mock or a missed glue site) before proceeding.

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && pytest -q 2>&1 | tail -15`
Expected: 485+ pass, 0 fail (skips OK). If a test imports torch for mocks, that test file may need a mock adjustment — but plugin/core code must not `import torch`.

- [ ] **Step 3: Run ruff**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && ruff check fusion_comfyui/ fusion_comfyui_plugin/`
Expected: clean (no errors). Fix any lint findings.

- [ ] **Step 4: Verify ComfyUI startup + node resolution**

Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && bash startTian.sh &` then check the log for "Patched N native node overrides" and no ImportError, then kill it.

Simpler non-server check:
Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, 'ComfyUI'); sys.path.insert(0, 'fusion_comfyui_plugin')
import nodes as comfy_nodes
import fusion_comfyui_plugin as p
missing = [n for n in ['ImageScale','ImageBatch','EmptyImage','ImagePadForOutpaint','LoadImageMask','ConditioningSetMask','ControlNetApply','PainterNode'] if n not in comfy_nodes.NODE_CLASS_MAPPINGS]
print('missing overrides:', missing)
print('plugin mappings:', len(p.NODE_CLASS_MAPPINGS))
"`
Expected: `missing overrides: []`; a nonzero plugin-mappings count.

- [ ] **Step 5: Scaling parity sanity (corr vs reference, optional torch-compare)**

If torch is importable, verify parity on one image:
Run: `cd /Users/dahai/fusion/fusion-comfyui && source .venv/bin/activate && python -c "
import numpy as np
from PIL import Image
rng = np.random.default_rng(0)
src = rng.random((1,3,32,32), dtype=np.float32)
from nodes._scaling import common_upscale
out_np = common_upscale(src.copy(), 64, 64, 'lanczos', 'disabled')
import torch, comfy.utils
out_t = comfy.utils.common_upscale(torch.from_numpy(src.copy()), 64, 64, 'lanczos', 'disabled').numpy()
corr = float(np.corrcoef(out_np.ravel(), out_t.ravel())[0,1])
print('lanczos corr vs torch:', corr)
assert corr >= 0.999, corr
"`
Expected: corr ≥ 0.999. (If torch import fails in this env, skip with a note — the unit tests already cover shape/range.)

- [ ] **Step 6: Final commit + memory + README**

Update `README.md` with a P2 entry under the version history (numpy IMAGE/MASK, torch-free I/O). Update memory file `p2-fork-image-numpy-2026-08-26.md` with status DONE + exit criteria results.

```bash
git add README.md
git commit -m "docs: P2 — IMAGE/MASK now numpy, torch-free I/O glue"
```

- [ ] **Step 7: Cleanup process data**

Delete any `/tmp` test images or scratch files created during verification. Keep only final outputs + logs per project rule.

---

## Self-Review

**1. Spec coverage:**
- Remove every `import torch` from fusion_comfyui/ + plugin → Task 3 (bridge), Task 5 (ip_adapter), Task 7 (LoadImage), Task 8 grep gate. ✓
- IMAGE/MASK numpy contract → Task 3 (bridge contract), Tasks 2/7 (nodes emit numpy). ✓
- 6 pure transforms forked → Task 2. ✓
- 7 dead paths stubbed → Task 6. ✓
- Scaling kernels ported (common_upscale/bislerp/lanczos) → Task 1. ✓
- samplers.py:630 hidden torch dep → Task 4. ✓
- ip_adapter .bin dropped → Task 5. ✓
- Exit criteria (grep 0, 485+ tests, startup, parity ≥0.999, memory+README+commit) → Task 8. ✓

**2. Placeholder scan:** No TBD/TODO in task bodies. All code blocks complete. Test code present for every task. ✓

**3. Type consistency:** `common_upscale(samples, width, height, method, crop)` signature identical in Task 1 (def) and Task 2/4 (call). `to_image_tensor`/`to_mask_numpy`/`to_image_numpy` consistent across Task 3 def + Task 7 call. Stub `FUNCTION="stub_run"` consistent in Task 6 def + test. ✓

**One risk noted:** Task 6 stub subclasses native classes at import time; if a native module isn't importable in a given env, `_stub_factory` falls back to `object` and logs. The test in Task 6 calls `fn()` with no args — stub ignores args, so it raises cleanly regardless.
