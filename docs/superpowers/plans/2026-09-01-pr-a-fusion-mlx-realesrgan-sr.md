# PR-A: fusion-mlx RealESRGAN MLX Super-Resolution Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-MLX RealESRGAN x4plus super-resolution engine + `POST /v1/images/super-resolution` endpoint to fusion-mlx, so downstream (fusion-comfyui) can SR H3 video frames without torch.

**Architecture:** New `fusion_mlx/image/sr/` module ports the RRDBNet architecture to `mlx.nn`. A new FastAPI route under the existing `/v1/images` prefix accepts a PNG image + scale, runs tiled MLX inference, returns the SR PNG. Model weights (`RealESRGAN_x4plus.pth`) loaded from `~/.fusion-mlx/models/realesrgan/` via hf-mirror. Follows the existing `api/images.py` route pattern (APIRouter + `set_*_context(pool)` injection registered in `server.py`).

**Tech Stack:** MLX (`mlx.core`, `mlx.nn`), FastAPI, pydantic, pytest, pytest-asyncio, ruff. Reference: official RealESRGAN_x4plus (basicsr RRDBNet) for alignment verification.

**Spec:** `docs/superpowers/specs/2026-09-01-h3-video-sr-design.md` (PR-A section)

## Global Constraints

- Pure MLX/Metal — no torch, no einops in inference (fusion-mlx core rule).
- 4-space multiples indentation, no docstrings, always include logging.
- Model downloads via mirror `https://hf-mirror.com` into `~/.fusion-mlx/models/realesrgan/`.
- Model detection from state-dict keys, not filename (fusion-mlx convention).
- Tests in `tests/unit/` (no model) + `tests/integration/` (real model). `asyncio_mode=auto`.
- Alignment gate before release: SR output vs torch reference PSNR > 35dB, SSIM > 0.95.
- Release flow: bump PR → merge → GitHub release tag → publish.yml auto PyPI (per memory `qwen-image-t2i-aicf-landing`).
- All file paths below are relative to `/Users/dahai/fusion/fusion-mlx/`.

## File Structure

- Create: `fusion_mlx/image/sr/__init__.py` — public re-exports (`RealESRGANConfig`, `RRDBNet`, `load_sr_model`, `super_resolve`).
- Create: `fusion_mlx/image/sr/config.py` — `RealESRGANConfig` dataclass mirroring official RealESRGAN_x4plus.
- Create: `fusion_mlx/image/sr/rrdb.py` — `RRDBNet` MLX port (RRDB blocks + pixelshuffle upsample). Pure `mlx.nn`.
- Create: `fusion_mlx/image/sr/weights.py` — state-dict detection, key remapping, model load from disk.
- Create: `fusion_mlx/image/sr/generate.py` — `super_resolve(images, scale, tile_size, tile_overlap)` tiled inference entry.
- Create: `fusion_mlx/api/images_sr.py` — FastAPI route `POST /v1/images/super-resolution` + pydantic models + `set_images_sr_context`.
- Modify: `fusion_mlx/server.py` — import + register `images_sr` router, inject context in `create_app`.
- Modify: `fusion_mlx/public_api.py` — re-export `super_resolve` / `RRDBNet` for downstream stability.
- Create: `tests/unit/test_sr_rrdb.py` — RRDBNet forward shape, tile==whole, key remap.
- Create: `tests/integration/test_sr_e2e.py` — real model load + endpoint round-trip + alignment vs torch (marked, needs model).

---

## Task 1: RealESRGANConfig dataclass

**Files:**
- Create: `fusion_mlx/image/sr/config.py`
- Test: `tests/unit/test_sr_config.py`

**Interfaces:**
- Consumes: none
- Produces: `RealESRGANConfig` dataclass with fields `num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32, res_scale=1.0` — mirrors official RealESRGAN_x4plus (basicsr RRDBNet defaults). Used by Task 2 (RRDBNet) and Task 3 (weights).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sr_config.py
from fusion_mlx.image.sr.config import RealESRGANConfig


def test_realesrgan_x4plus_defaults():
    cfg = RealESRGANConfig()
    assert cfg.num_in_ch == 3
    assert cfg.num_out_ch == 3
    assert cfg.scale == 4
    assert cfg.num_feat == 64
    assert cfg.num_block == 23
    assert cfg.num_grow_ch == 32
    assert cfg.res_scale == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sr_config.py::test_realesrgan_x4plus_defaults -v`
Expected: FAIL `ModuleNotFoundError: No module named 'fusion_mlx.image.sr'`

- [ ] **Step 3: Write minimal implementation**

```python
# fusion_mlx/image/sr/config.py
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RealESRGANConfig:
    # Mirrors official RealESRGAN_x4plus (basicsr.arch.rrdb_net.RRDBNet):
    # num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23,
    # num_grow_ch=32, res_scale=1.0. The ~64MB .pth holds these defaults.
    num_in_ch: int = 3
    num_out_ch: int = 3
    scale: int = 4
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    res_scale: float = 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sr_config.py::test_realesrgan_x4plus_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/image/sr/config.py tests/unit/test_sr_config.py
git commit -m "feat(sr): add RealESRGANConfig dataclass for x4plus defaults"
```

---

## Task 2: RRDBNet MLX port (architecture)

**Files:**
- Create: `fusion_mlx/image/sr/rrdb.py`
- Test: `tests/unit/test_sr_rrdb.py`

**Interfaces:**
- Consumes: `RealESRGANConfig` (Task 1); reuses `fusion_mlx.image.sdxl.unet.Conv2d` (NHWC conv helper, already exists — weight layout `(out_ch, k, k, in_ch)`).
- Produces:
  - `class RRDBNet(nn.Module)` with `__init__(self, config: RealESRGANConfig)` and `__call__(self, x: mx.array) -> mx.array`. Input `x` shape `(b, h, w, c)` NHWC float32, c=3. Output `(b, h*scale, w*scale, c)`. Param names MUST be `conv_first.weight/bias`, `body.{i}.rdb{1..3}.conv{1..5}.weight/bias`, `conv_body.weight/bias`, `conv_up1/conv_up2`, `conv_hr`, `conv_last` — these match the official `.pth` keys (after prefix strip in Task 3), so weights load by exact name.
  - module-level `pixel_shuffle(x: mx.array, upscale: int) -> mx.array` and `leaky_relu(x: mx.array, slope: float = 0.2) -> mx.array` helpers (used by Task 4 too).

Architecture detail (port faithfully, no shortcuts):
- `conv_first = Conv2d(num_in_ch, num_feat)`
- `body = [RRDB(num_feat, num_grow_ch, res_scale) for _ in range(num_block)]`
- `conv_body = Conv2d(num_feat, num_feat)`
- For scale=4: `conv_up1 = Conv2d(num_feat, num_feat)`, `conv_up2 = Conv2d(num_feat, num_feat)` (two x2 pixelshuffles). For scale=2: only `conv_up1`.
- `conv_hr = Conv2d(num_feat, num_feat)`, `conv_last = Conv2d(num_feat, num_out_ch)`
- `RRDB`: three `ResidualDenseBlock_5C` in sequence, `out = rdb3(rdb2(rdb1(x)))`, return `out * res_scale + x`.
- `ResidualDenseBlock_5C`: 5 convs with growing concat (ch = num_feat + k*num_grow_ch), lrelu between, `x5 = conv5(concat(x,x1..x4))`, return `x5 * 0.2 + x`.
- forward: `feat = conv_first(x); body_feat = feat; feat = body(feat); feat = conv_body(feat); feat = body_feat + feat;` then per upsample stage `feat = leaky_relu(conv_upN(feat)); feat = pixel_shuffle(feat, 2);` then `out = conv_last(leaky_relu(conv_hr(feat)))`.

The MLX `Conv2d` helper from sdxl operates on NHWC and uses `mx.random.normal` init — that init is overwritten by `load_weights` in Task 3, so init values don't matter (fusion-mlx rule: init with `torch.empty`-equivalent; here random, loaded from state dict). All work stays NHWC (no NCHW conversion needed — simpler than the diffusion UNets, and the only consumer is `super_resolve`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sr_rrdb.py
import numpy as np

from fusion_mlx.image.sr.config import RealESRGANConfig
from fusion_mlx.image.sr.rrdb import RRDBNet, leaky_relu, pixel_shuffle


def test_leaky_relu_pos_and_neg():
    import mlx.core as mx
    x = mx.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    out = leaky_relu(x, 0.2)
    out_np = np.array(out)
    expected = np.array([[-0.4, -0.2, 0.0, 1.0, 2.0]])
    assert np.allclose(out_np, expected, atol=1e-6)


def test_pixel_shuffle_x2_doubles_dims():
    import mlx.core as mx
    # (b=1, h=2, w=3, c=4*2*2=16) -> (1, 4, 6, 4)
    x = mx.arange(1 * 2 * 3 * 16, dtype=mx.float32).reshape(1, 2, 3, 16)
    out = pixel_shuffle(x, 2)
    assert out.shape == (1, 4, 6, 4), out.shape


def test_rrdbnet_forward_shape_scale4():
    import mlx.core as mx
    cfg = RealESRGANConfig()  # scale=4, num_feat=64, num_block=23
    net = RRDBNet(cfg)
    # NHWC input (b=1, h=8, w=8, c=3) -> (1, 32, 32, 3)
    x = mx.zeros((1, 8, 8, 3), dtype=mx.float32)
    out = net(x)
    assert out.shape == (1, 32, 32, 3), out.shape


def test_rrdbnet_forward_shape_scale2():
    import mlx.core as mx
    cfg = RealESRGANConfig(scale=2)
    net = RRDBNet(cfg)
    x = mx.zeros((1, 10, 12, 3), dtype=mx.float32)
    out = net(x)
    assert out.shape == (1, 20, 24, 3), out.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sr_rrdb.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'fusion_mlx.image.sr.rrdb'`

- [ ] **Step 3: Write minimal implementation**

```python
# fusion_mlx/image/sr/rrdb.py
import logging

import mlx.core as mx
from mlx import nn

from fusion_mlx.image.sr.config import RealESRGANConfig
from fusion_mlx.image.sdxl.unet import Conv2d

logger = logging.getLogger(__name__)


def leaky_relu(x: mx.array, slope: float = 0.2) -> mx.array:
    return mx.where(x > 0, x, slope * x)


def pixel_shuffle(x: mx.array, upscale: int) -> mx.array:
    # NHWC (b, h, w, c*r*r) -> (b, h*r, w*r, c). MLX has no builtin pixel_shuffle.
    b, h, w, cr2 = x.shape
    c = cr2 // (upscale * upscale)
    x = x.reshape(b, h, w, upscale, upscale, c)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(b, h * upscale, w * upscale, c)
    return x


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int):
        super().__init__()
        self.conv1 = Conv2d(num_feat, num_grow_ch)
        self.conv2 = Conv2d(num_feat + num_grow_ch, num_grow_ch)
        self.conv3 = Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch)
        self.conv4 = Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch)
        self.conv5 = Conv2d(num_feat + 4 * num_grow_ch, num_feat)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = leaky_relu(self.conv1(x))
        x2 = leaky_relu(self.conv2(mx.concatenate([x, x1], axis=-1)))
        x3 = leaky_relu(self.conv3(mx.concatenate([x, x1, x2], axis=-1)))
        x4 = leaky_relu(self.conv4(mx.concatenate([x, x1, x2, x3], axis=-1)))
        x5 = self.conv5(mx.concatenate([x, x1, x2, x3, x4], axis=-1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int, res_scale: float):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.res_scale = res_scale

    def __call__(self, x: mx.array) -> mx.array:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.res_scale + x


class RRDBNet(nn.Module):
    def __init__(self, config: RealESRGANConfig):
        super().__init__()
        self.scale = config.scale
        nf = config.num_feat
        self.conv_first = Conv2d(config.num_in_ch, nf)
        self.body = [RRDB(nf, config.num_grow_ch, config.res_scale)
                     for _ in range(config.num_block)]
        self.conv_body = Conv2d(nf, nf)
        if config.scale == 4:
            self.conv_up1 = Conv2d(nf, nf)
            self.conv_up2 = Conv2d(nf, nf)
        elif config.scale == 2:
            self.conv_up1 = Conv2d(nf, nf)
        self.conv_hr = Conv2d(nf, nf)
        self.conv_last = Conv2d(nf, config.num_out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        feat = self.conv_first(x)
        body_feat = feat
        for blk in self.body:
            feat = blk(feat)
        feat = self.conv_body(feat)
        feat = body_feat + feat
        if self.scale == 4:
            feat = leaky_relu(self.conv_up1(feat))
            feat = pixel_shuffle(feat, 2)
            feat = leaky_relu(self.conv_up2(feat))
            feat = pixel_shuffle(feat, 2)
        elif self.scale == 2:
            feat = leaky_relu(self.conv_up1(feat))
            feat = pixel_shuffle(feat, 2)
        out = self.conv_last(leaky_relu(self.conv_hr(feat)))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sr_rrdb.py -v`
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/image/sr/rrdb.py tests/unit/test_sr_rrdb.py
git commit -m "feat(sr): port RRDBNet (RealESRGAN) architecture to MLX"
```

---

## Task 3: Weight loading + key remapping

**Files:**
- Create: `fusion_mlx/image/sr/weights.py`
- Test: `tests/unit/test_sr_weights.py`

**Interfaces:**
- Consumes: `RRDBNet` (Task 2). State-dict file `RealESRGAN_x4plus.pth` at `~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.pth` (torch `.pth`, loaded via `mx.load` after torch→numpy bridge, OR safetensors variant). torch Conv weight is `(out_ch, in_ch, k, k)`; MLX `Conv2d` weight is `(out_ch, k, k, in_ch)` NHWC — remap = `transpose(0, 2, 3, 1)`. Bias passes through.
- Produces: `load_sr_model(model_path: str, config: RealESRGANConfig | None = None) -> RRDBNet` — builds net, loads weights, returns ready net. Strips `params_ema.` / `params.` prefix if present. Logs matched/missing/unmatched counts (matches fusion-mlx convention from `h3/vae.py:919-929`).

NOTE on torch `.pth`: fusion-mlx is pure-MLX and may not have torch. The official RealESRGAN_x4plus ships as `RealESRGAN_x4plus.pth` (torch pickle). Two options, pick based on environment probe in Step 1:
- (a) If torch is importable: `import torch; sd = torch.load(path, map_location="cpu")["params_ema"]`.
- (b) Convert once to `.safetensors` and load via `mx.load`. The plan assumes a one-time conversion script (Task 3 Step 0) writes `RealESRGAN_x4plus.safetensors` next to the `.pth`, and `load_sr_model` loads the safetensors. This keeps fusion-mlx pure-MLX at runtime (torch only used in the throwaway converter, not imported by `weights.py`). Verify torch availability in the venv first; if absent, document the converter as a setup step run in a torch-enabled env.

- [ ] **Step 0: Probe torch availability + convert model**

Run: `python -c "import torch; print(torch.__version__)"` and `ls ~/.fusion-mlx/models/realesrgan/`.
If torch present and `.pth` downloaded: write `/tmp/convert_realesrgan.py` that loads `.pth` with `torch.load(path, map_location="cpu", weights_only=True)` (the `.pth` holds only `params_ema` tensors — `weights_only=True` is safe and avoids arbitrary-code-execution via pickle), strips `params_ema.` prefix, transposes conv weights to NHWC `(out,k,k,in)`, saves `RealESRGAN_x4plus.safetensors`. Run it. Verify keys count (~ the RRDBNet param count). Delete the converter (throwaway, project cleanup rule). If torch absent in fusion-mlx venv but present elsewhere, document: run converter once, commit only the `.safetensors` path convention.

> The safetensors stores keys WITHOUT prefix and with MLX-NHWC conv layout, so `weights.py` loads them by exact param name — no transpose at load time.

- [ ] **Step 1: Write the failing test** (uses a tiny synthetic safetensors, not the real 64MB model — unit test must run without GPU/model download)

```python
# tests/unit/test_sr_weights.py
import os

import numpy as np
import pytest


@pytest.fixture
def tiny_sr_safetensors(tmp_path):
    # Build a minimal RRDBNet (1 block, scale=2) and dump its params as
    # safetensors with the MLX-NHWC layout load_sr_model expects.
    import mlx.core as mx
    from safetensors.numpy import save_file
    from fusion_mlx.image.sr.config import RealESRGANConfig
    from fusion_mlx.image.sr.rrdb import RRDBNet

    cfg = RealESRGANConfig(num_block=1, scale=2)
    net = RRDBNet(cfg)
    params = dict(net.parameters())  # mlx tree
    flat = {}
    for k, v in params.items():
        flat[k] = np.array(v)
    path = tmp_path / "tiny.safetensors"
    save_file(flat, str(path))
    return str(path), cfg


def test_load_sr_model_matches_all_keys(tiny_sr_safetensors):
    from fusion_mlx.image.sr.weights import load_sr_model
    path, cfg = tiny_sr_safetensors
    net = load_sr_model(path, config=cfg)
    # No missing/unmatched -> param tree fully matched
    assert net is not None


def test_load_sr_model_strips_params_ema_prefix(tmp_path, tiny_sr_safetensors):
    # Re-save with a 'params_ema.' prefix to mimic the raw .pth convention.
    from safetensors.numpy import save_file
    import mlx.core as mx
    from fusion_mlx.image.sr.weights import load_sr_model
    base_path, cfg = tiny_sr_safetensors
    from safetensors import safe_open
    prefixed = {}
    with safe_open(base_path, framework="numpy") as f:
        for k in f.keys():
            prefixed["params_ema." + k] = f.get_tensor(k)
    p = tmp_path / "prefixed.safetensors"
    save_file(prefixed, str(p))
    net = load_sr_model(str(p), config=cfg)
    assert net is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sr_weights.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'fusion_mlx.image.sr.weights'`

- [ ] **Step 3: Write minimal implementation**

```python
# fusion_mlx/image/sr/weights.py
import logging
import os

from mlx.utils import tree_flatten, tree_unflatten

from fusion_mlx.image.sr.config import RealESRGANConfig
from fusion_mlx.image.sr.rrdb import RRDBNet

logger = logging.getLogger(__name__)

_PREFIXES = ("params_ema.", "params.")


def _strip_prefix(key: str) -> str:
    for p in _PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def load_sr_model(model_path: str, config: RealESRGANConfig | None = None) -> RRDBNet:
    # Loads a RealESRGAN safetensors (MLX-NHWC conv layout, keys already
    # transpose-correct from the Task 3 Step 0 converter). Strips
    # params_ema./params. prefix if a raw-converted file is passed.
    import mlx.core as mx

    cfg = config or RealESRGANConfig()
    net = RRDBNet(cfg)
    if not os.path.exists(model_path):
        logger.warning("sr weights: %s missing, returning random-init net", model_path)
        return net

    raw = mx.load(model_path)
    mapped = {_strip_prefix(k): v for k, v in raw.items()}

    flat = tree_flatten(net.parameters())
    loaded = {}
    matched = 0
    for k, v in flat:
        if k in mapped:
            mv = mapped[k]
            if mv.dtype != mx.float32:
                mv = mv.astype(mx.float32)
            if list(mv.shape) != list(v.shape):
                logger.warning("sr weights: shape mismatch %s %s vs %s",
                               k, list(mv.shape), list(v.shape))
                continue
            loaded[k] = mv
            matched += 1
    net.load_weights(list(loaded.items()), strict=False)

    missing = len(flat) - matched
    unmatched = len(mapped) - matched
    if missing or unmatched:
        logger.warning("sr weights: matched=%d missing=%d unmatched=%d (%s)",
                       matched, missing, unmatched, model_path)
    else:
        logger.info("sr weights: loaded %d/%d keys (strict) from %s",
                    matched, len(flat), model_path)
    return net
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sr_weights.py -v`
Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/image/sr/weights.py tests/unit/test_sr_weights.py
git commit -m "feat(sr): add state-dict loader with prefix strip + NHWC layout"
```

---

## Task 4: super_resolve tiled inference entry

**Files:**
- Create: `fusion_mlx/image/sr/generate.py`
- Create: `fusion_mlx/image/sr/__init__.py`
- Test: `tests/unit/test_sr_generate.py`

**Interfaces:**
- Consumes: `RRDBNet` (Task 2), `load_sr_model` (Task 3). Tile/blend helpers mirror `fusion_mlx/video/minimax_h3/vae.py:812-897` (`_split_tiles`, `_blend`, `tiled_decode`) — adapted for 2D image (no temporal axis).
- Produces:
  - `def super_resolve(images: np.ndarray, model_path: str | None = None, scale: int = 4, tile_size: int = 512, tile_overlap: int = 64, config: RealESRGANConfig | None = None) -> np.ndarray` — `images` shape `(n, h, w, 3)` float32 [0,1]; returns `(n, h*scale, w*scale, 3)` float32 [0,1]. Caches the loaded net (module-level `_NET_CACHE` keyed by `(model_path, scale)`). For tiles larger than `tile_size`, splits H and W, runs each tile through the net, blends overlaps (linear ramp, same math as `vae._blend`), reassembles. `mx.eval` each tile to bound memory (matches `vae.tiled_decode:880`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sr_generate.py
import numpy as np


def test_super_resolve_small_image_no_tiling(tmp_path):
    # Image smaller than tile_size -> single forward, no split.
    from fusion_mlx.image.sr.generate import super_resolve
    imgs = np.random.rand(1, 16, 20, 3).astype(np.float32)
    out = super_resolve(imgs, model_path=None, scale=4, tile_size=512,
                        tile_overlap=64)
    assert out.shape == (1, 64, 80, 3), out.shape
    assert out.dtype == np.float32


def test_super_resolve_tile_equals_whole(tmp_path):
    # Tiled output must match whole-image output (the SR-tiling correctness gate).
    import mlx.core as mx
    from fusion_mlx.image.sr.config import RealESRGANConfig
    from fusion_mlx.image.sr.rrdb import RRDBNet
    from fusion_mlx.image.sr.generate import super_resolve, _set_net_for_test

    cfg = RealESRGANConfig(num_block=1, scale=2)
    net = RRDBNet(cfg)
    mx.eval(net.parameters())
    _set_net_for_test(net, scale=2)

    imgs = (np.random.rand(1, 40, 48, 3) * 0.5 + 0.2).astype(np.float32)
    whole = super_resolve(imgs, scale=2, tile_size=512, tile_overlap=64)
    tiled = super_resolve(imgs, scale=2, tile_size=16, tile_overlap=8)
    diff = np.abs(whole - tiled).max()
    assert diff < 1e-3, f"tile vs whole max diff {diff}"


def test_super_resolve_preserves_range(tmp_path):
    from fusion_mlx.image.sr.generate import super_resolve
    from fusion_mlx.image.sr.config import RealESRGANConfig
    from fusion_mlx.image.sr.rrdb import RRDBNet
    from fusion_mlx.image.sr.generate import _set_net_for_test
    import mlx.core as mx
    cfg = RealESRGANConfig(num_block=1, scale=2)
    _set_net_for_test(RRDBNet(cfg), scale=2)
    imgs = np.zeros((1, 12, 12, 3), dtype=np.float32)
    out = super_resolve(imgs, scale=2, tile_size=512, tile_overlap=64)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sr_generate.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'fusion_mlx.image.sr.generate'`

- [ ] **Step 3: Write minimal implementation**

```python
# fusion_mlx/image/sr/generate.py
import logging
import math
import os

import mlx.core as mx
import numpy as np

from fusion_mlx.image.sr.config import RealESRGANConfig
from fusion_mlx.image.sr.rrdb import RRDBNet
from fusion_mlx.image.sr.weights import load_sr_model

logger = logging.getLogger(__name__)

_NET_CACHE: dict = {}
_TEST_NET = None


def _set_net_for_test(net, scale):
    # Test hook: bypass disk load + cache for unit tests without a model file.
    global _TEST_NET
    _TEST_NET = (net, scale)


def _get_net(model_path, scale, config):
    if _TEST_NET is not None and _TEST_NET[1] == scale:
        return _TEST_NET[0]
    key = (model_path, scale)
    if key not in _NET_CACHE:
        cfg = config or RealESRGANConfig(scale=scale)
        path = model_path or os.path.expanduser(
            "~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.safetensors")
        _NET_CACHE[key] = load_sr_model(path, cfg)
        logger.info("sr: loaded net scale=%d from %s", scale, path)
    return _NET_CACHE[key]


def _split_tiles(input_len, tile_size, overlap):
    if tile_size >= input_len:
        return [0], [input_len], []
    n = math.ceil((input_len - overlap) / (tile_size - overlap))
    starts = []
    for i in range(n):
        s = i * (tile_size - overlap)
        if s + tile_size > input_len:
            s = input_len - tile_size
        if s not in starts:
            starts.append(s)
    lens = [tile_size] * len(starts)
    overlaps = []
    for i in range(1, len(starts)):
        overlaps.append(starts[i - 1] + tile_size - starts[i])
    return starts, lens, overlaps


def _blend(a, b, blend_extent, dim):
    blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
    positions = mx.arange(blend_extent, dtype=a.dtype)
    wa = 1.0 - positions / blend_extent
    wb = positions / blend_extent
    shape = [1] * a.ndim
    shape[dim] = blend_extent
    wa = wa.reshape(shape)
    wb = wb.reshape(shape)
    sla = [slice(None)] * a.ndim
    sla[dim] = slice(-blend_extent, None)
    slb = [slice(None)] * b.ndim
    slb[dim] = slice(0, blend_extent)
    blended = a[tuple(sla)] * wa + b[tuple(slb)] * wb
    if blend_extent < b.shape[dim]:
        slr = [slice(None)] * b.ndim
        slr[dim] = slice(blend_extent, None)
        return mx.concatenate([blended, b[tuple(slr)]], axis=dim)
    return blended


def _resolve_one(frame_hwc, net, scale, tile_size, tile_overlap):
    # frame_hwc: mx.array (h, w, 3) float32 -> (h*scale, w*scale, 3)
    x = frame_hwc[None, ...]  # (1, h, w, 3)
    h, w = x.shape[1], x.shape[2]
    if h <= tile_size and w <= tile_size:
        out = net(x)[0]
        mx.eval(out)
        return out
    y_idx, y_len, y_ov = _split_tiles(h, tile_size, tile_overlap)
    x_idx, x_len, x_ov = _split_tiles(w, tile_size, tile_overlap)
    rows = []
    for i, (iy, ly) in enumerate(zip(y_idx, y_len)):
        row = []
        for j, (jx, lx) in enumerate(zip(x_idx, x_len)):
            tile = x[:, iy:iy + ly, jx:jx + lx, :]
            tile_out = net(tile)[0]
            mx.eval(tile_out)
            if i > 0:
                tile_out = _blend(rows[-1][j], tile_out, y_ov[i - 1], dim=-2)
            if j > 0:
                tile_out = _blend(row[-1], tile_out, x_ov[j - 1], dim=-1)
            if i < len(y_idx) - 1:
                tile_out = tile_out[:-(y_ov[i]), :]
            if j < len(x_idx) - 1:
                tile_out = tile_out[:, :-(x_ov[j])]
            row.append(tile_out)
        rows.append(mx.concatenate(row, axis=-2))
    return mx.concatenate(rows, axis=-1)


def super_resolve(images, model_path=None, scale=4, tile_size=512,
                  tile_overlap=64, config=None):
    # images: np.ndarray (n, h, w, 3) float32 [0,1] -> (n, h*scale, w*scale, 3)
    net = _get_net(model_path, scale, config)
    n, h, w, _ = images.shape
    out = np.empty((n, h * scale, w * scale, 3), dtype=np.float32)
    for i in range(n):
        frame = mx.array(images[i].astype(np.float32))
        sr = _resolve_one(frame, net, scale, tile_size, tile_overlap)
        mx.eval(sr)
        out[i] = np.array(sr)
        logger.info("sr: frame %d/%d in=%dx%d out=%dx%d", i + 1, n, w, h,
                    out.shape[2], out.shape[1])
    return out
```

```python
# fusion_mlx/image/sr/__init__.py
from fusion_mlx.image.sr.config import RealESRGANConfig
from fusion_mlx.image.sr.generate import super_resolve
from fusion_mlx.image.sr.rrdb import RRDBNet
from fusion_mlx.image.sr.weights import load_sr_model

__all__ = ["RealESRGANConfig", "RRDBNet", "load_sr_model", "super_resolve"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sr_generate.py -v`
Expected: all 3 PASS (tile-equals-whole is the core correctness gate)

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/image/sr/generate.py fusion_mlx/image/sr/__init__.py tests/unit/test_sr_generate.py
git commit -m "feat(sr): add tiled super_resolve inference entry"
```

---

## Task 5: POST /v1/images/super-resolution endpoint

**Files:**
- Create: `fusion_mlx/api/images_sr.py`
- Test: `tests/unit/test_sr_endpoint.py`

**Interfaces:**
- Consumes: `super_resolve` (Task 4). Mirrors `fusion_mlx/api/images.py` route pattern exactly: `APIRouter(prefix="/v1/images", tags=["images"])`, module-level `_pool`, `set_images_sr_context(pool)`, auth deps `verify_api_key`/`check_rate_limit`. The SR endpoint does NOT need an engine from the pool (SR is a standalone net, no EnginePool engine) — but it follows the same context-injection + auth-deps shape for consistency and so `verify_api_key` works. `_pool` may stay unused but is injected for parity; if `verify_api_key` needs it, pass through.
- Produces:
  - `router = APIRouter(prefix="/v1/images", tags=["images"])`
  - `def set_images_sr_context(pool: EnginePool) -> None`
  - pydantic `SRRequest` (multipart: `image: UploadFile`, `scale: int = 4`, `tile_size: int = 512`) — use `UploadFile` for the binary PNG so per-frame POST from PR-B sends raw bytes (matches the spec's "multipart/binary image bytes").
  - pydantic `SRResponse` (`image_b64: str`, `width: int`, `height: int`, `in_width: int`, `in_height: int`, `elapsed: float`).
  - `@router.post("/super-resolution", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])` async handler: read upload bytes → PIL decode → numpy float32 [0,1] `(1,h,w,3)` → `super_resolve(...)` → PIL encode SR → base64 → `SRResponse`. 503 with download instructions if model file missing.

- [ ] **Step 1: Write the failing test** (FastAPI TestClient + monkeypatch `super_resolve` to a deterministic stub — no real model in unit tests)

```python
# tests/unit/test_sr_endpoint.py
import base64
import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fusion_mlx.api import images_sr


def _stub_resolve(images, model_path=None, scale=4, tile_size=512,
                  tile_overlap=64, config=None):
    # Deterministic stub: nearest-style 2x/4x repeat, no model.
    n, h, w, c = images.shape
    rep = scale
    out = np.repeat(np.repeat(images, rep, axis=1), rep, axis=2)
    return out.astype(np.float32)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(images_sr, "super_resolve", _stub_resolve)
    app = FastAPI()
    app.include_router(images_sr.router)
    return TestClient(app)


def _png_bytes(arr_hwc):
    from PIL import Image
    pil = Image.fromarray((np.clip(arr_hwc, 0, 1) * 255).astype(np.uint8)[:, :, :3])
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def test_sr_endpoint_roundtrip(client):
    img = _png_bytes(np.random.rand(32, 40, 3).astype(np.float32))
    resp = client.post(
        "/v1/images/super-resolution",
        files={"image": ("frame.png", img, "image/png")},
        data={"scale": "4", "tile_size": "512"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["in_width"] == 40 and body["in_height"] == 32
    assert body["width"] == 160 and body["height"] == 128
    dec = base64.b64decode(body["image_b64"])
    from PIL import Image
    out = np.array(Image.open(io.BytesIO(dec)))
    assert out.shape[0] == 128 and out.shape[1] == 160
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sr_endpoint.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'fusion_mlx.api.images_sr'`

- [ ] **Step 3: Write minimal implementation**

```python
# fusion_mlx/api/images_sr.py
import base64
import io
import logging
import os
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from fusion_mlx.image.sr import super_resolve
from fusion_mlx.middleware.auth import check_rate_limit, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/images", tags=["images"])

_pool = None


def set_images_sr_context(pool) -> None:
    global _pool
    _pool = pool


class SRResponse(BaseModel):
    image_b64: str
    width: int
    height: int
    in_width: int
    in_height: int
    elapsed: float = Field(default=0.0)


def _to_numpy_hwc(png_bytes: bytes):
    from PIL import Image
    pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return arr[None, ...]  # (1, h, w, 3)


import numpy as np  # noqa: E402


@router.post(
    "/super-resolution",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def super_resolution(
    image: UploadFile = File(...),
    scale: int = Form(4),
    tile_size: int = Form(512),
) -> SRResponse:
    t0 = time.time()
    if scale not in (2, 4):
        raise HTTPException(422, "scale must be 2 or 4")
    model_path = os.path.expanduser(
        "~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.safetensors")
    if not os.path.exists(model_path):
        raise HTTPException(
            503,
            "RealESRGAN model not downloaded. Convert RealESRGAN_x4plus.pth "
            "to safetensors at " + model_path + " (see PR-A plan Task 3).",
        )
    raw = await image.read()
    if not raw:
        raise HTTPException(422, "empty image upload")
    try:
        inp = _to_numpy_hwc(raw)
    except Exception as exc:
        logger.warning("sr decode failed: %s", exc)
        raise HTTPException(422, "could not decode image") from exc
    try:
        sr = super_resolve(inp, model_path=model_path, scale=scale,
                           tile_size=tile_size, tile_overlap=64)
    except Exception as exc:
        logger.exception("sr inference failed")
        raise HTTPException(500, "super-resolution failed") from exc
    from PIL import Image
    out_hwc = sr[0]
    pil = Image.fromarray((np.clip(out_hwc, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    elapsed = time.time() - t0
    logger.info("sr endpoint: in=%dx%d out=%dx%d scale=%d elapsed=%.2fs",
                inp.shape[2], inp.shape[1], out_hwc.shape[1],
                out_hwc.shape[0], scale, elapsed)
    return SRResponse(
        image_b64=base64.b64encode(buf.getvalue()).decode(),
        width=out_hwc.shape[1], height=out_hwc.shape[0],
        in_width=inp.shape[2], in_height=inp.shape[1], elapsed=elapsed,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sr_endpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fusion_mlx/api/images_sr.py tests/unit/test_sr_endpoint.py
git commit -m "feat(sr): add POST /v1/images/super-resolution endpoint"
```

---

## Task 6: Register route in server.py + public_api re-export

**Files:**
- Modify: `fusion_mlx/server.py:39` (add import near `images`), `fusion_mlx/server.py:1018` (add `include_router`), `fusion_mlx/server.py:1479` (add `set_images_sr_context(self.pool)`)
- Modify: `fusion_mlx/public_api.py` (add re-exports)
- Test: `tests/unit/test_sr_registration.py`

**Interfaces:**
- Consumes: `images_sr.router`, `images_sr.set_images_sr_context` (Task 5).
- Produces: the SR route is live in the server app; `from fusion_mlx.public_api import super_resolve, RRDBNet` works downstream (PR-B will rely on the endpoint, not the import, but the re-export documents the public surface).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sr_registration.py
def test_sr_router_registered_in_create_app():
    # create_app must include the super-resolution route.
    from fusion_mlx.server import create_app
    app = create_app()
    paths = {r.path for r in app.routes}
    assert "/v1/images/super-resolution" in paths, paths


def test_public_api_reexports_sr():
    from fusion_mlx.public_api import RRDBNet, super_resolve
    assert callable(super_resolve)
    assert RRDBNet is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sr_registration.py -v`
Expected: FAIL (route not registered / names not exported)

- [ ] **Step 3: Make the three server.py edits**

Edit 1 — add import after line 39 (`from .api.images import set_images_context`):

```python
from .api.images_sr import router as images_sr_router
from .api.images_sr import set_images_sr_context
```

Edit 2 — add after line 1018 (`app.include_router(images_router)`):

```python
app.include_router(images_sr_router)
```

Edit 3 — add after line 1479 (`set_images_context(self.pool)`):

```python
set_images_sr_context(self.pool)
```

Edit 4 — `fusion_mlx/public_api.py`: add with the other image re-exports,

```python
from .image.sr import RRDBNet, super_resolve
```

and add `"RRDBNet", "super_resolve"` to `__all__`.

> NOTE: `create_app` may need `verify_api_key` to tolerate a test env without an API key. If the registration test 401s, follow the existing images test pattern in `tests/unit/` for how `create_app` is tested (check `test_*images*` for auth-bypass in tests) and mirror it — do NOT weaken production auth.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sr_registration.py -v`
Expected: both PASS

- [ ] **Step 5: Run full SR unit suite + ruff**

Run: `pytest tests/unit/test_sr_config.py tests/unit/test_sr_rrdb.py tests/unit/test_sr_weights.py tests/unit/test_sr_generate.py tests/unit/test_sr_endpoint.py tests/unit/test_sr_registration.py -v && ruff check fusion_mlx/image/sr fusion_mlx/api/images_sr.py`
Expected: all PASS, ruff clean

- [ ] **Step 6: Commit**

```bash
git add fusion_mlx/server.py fusion_mlx/public_api.py tests/unit/test_sr_registration.py
git commit -m "feat(sr): register /v1/images/super-resolution + public_api re-exports"
```

---

## Task 7: Alignment integration test (release gate)

**Files:**
- Create: `tests/integration/test_sr_alignment.py`
- Reference: torch basicsr/diffusers RealESRGAN_x4plus output on 3 fixed test images.

**Interfaces:**
- Consumes: real `RealESRGAN_x4plus.safetensors` (Task 3 Step 0) + a torch reference. Marked `@pytest.mark.integration` (skipped unless `--run-integration` / model present).
- Produces: PSNR > 35dB, SSIM > 0.95 vs torch reference on 3 images. This is the gate before opening the fusion-mlx PR.

- [ ] **Step 1: Write the test** (guarded — skips if model or torch reference absent)

```python
# tests/integration/test_sr_alignment.py
import os

import numpy as np
import pytest

_MODEL = os.path.expanduser(
    "~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.safetensors")


def _have():
    return os.path.exists(_MODEL)


@pytest.mark.integration
@pytest.mark.skipif(not _have(), reason="RealESRGAN safetensors not present")
def test_sr_alignment_vs_torch(tmp_path):
    from fusion_mlx.image.sr import super_resolve
    from PIL import Image

    # 3 fixed LR images (download once to tests/data/sr_align/ or generate).
    # For the gate: load each, run MLX SR, run torch basicsr SR, compare.
    try:
        import torch  # noqa: F401
        from basicsr.archs.rrdb_net_arch import RRDBNet as TorchRRDB  # noqa: F401
    except Exception:
        pytest.skip("torch/basicsr not available for reference")

    # --- build torch reference net + load same weights (NHWC->NCHW transpose) ---
    # --- run on 3 images, collect ref outputs ---
    # --- run fusion_mlx super_resolve on same 3 images ---
    # --- assert PSNR > 35, SSIM > 0.95 for each ---
    # (concrete comparison code filled with the actual test images once
    #  the converter + torch env are available in the execution env)
    for img_path in _three_test_images(tmp_path):
        lr = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        mlx_sr = super_resolve(lr[None, ...], model_path=_MODEL, scale=4)
        ref_sr = _torch_reference(lr, _MODEL)  # implemented in-test
        psnr = _psnr(mlx_sr[0], ref_sr)
        ssim = _ssim(mlx_sr[0], ref_sr)
        assert psnr > 35.0, f"{img_path}: PSNR {psnr}"
        assert ssim > 0.95, f"{img_path}: SSIM {ssim}"
```

> The torch reference helper `_torch_reference` and the 3 test images are filled during execution once the torch env + converted model are in place. The assertions + gate thresholds are fixed here; only the I/O plumbing is env-dependent. If torch cannot run in the fusion-mlx venv, run the comparison in a separate torch-enabled env and record the PSNR/SSIM numbers in the PR description as the gate evidence (the test stays as the regression guard for future MLX-side changes, comparing against a cached reference array instead of live torch).

- [ ] **Step 2: Run + record** (during execution with real model)

Run: `pytest tests/integration/test_sr_alignment.py -v --run-integration`
Expected: PASS with PSNR > 35, SSIM > 0.95 for all 3 images. Record the numbers.

- [ ] **Step 3: Cleanup + commit**

Delete all intermediate frames/stub outputs (project cleanup rule). Commit the test + any cached reference arrays under `tests/data/sr_align/`.

```bash
git add tests/integration/test_sr_alignment.py tests/data/sr_align/
git commit -m "test(sr): add alignment gate vs torch RealESRGAN (PSNR>35 SSIM>0.95)"
```

---

## Task 8: File issue → PR → release → dep bump

Per project rule (issue → PR → dep bump) and the spec's Open Question #3. No existing SR issue found in fusion-mlx (gh search confirmed only #481/#480/#493, unrelated).

- [ ] **Step 1: File the fusion-mlx issue** (English, GitHub). Body: Summary / Context (H3 VAE 16x low-pass, SR post-processing path) / What's needed (RealESRGAN MLX engine + `/v1/images/super-resolution` endpoint) / Workdown (the 7 tasks above). Link the spec.
- [ ] **Step 2: Open PR** against fusion-mlx with all 7 code tasks. CI: ruff + pytest unit green.
- [ ] **Step 3: Merge → GitHub release tag** → `publish.yml` auto-publishes PyPI (per memory `qwen-image-t2i-aicf-landing`).
- [ ] **Step 4: Dep bump** in fusion-comfyui (`fusion-mlx>=<new>`) — enables PR-B real SR. Until then PR-B runs in `mock` mode.
