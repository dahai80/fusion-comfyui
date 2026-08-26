# P3 Transparent Staged Default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route native ComfyUI `KSampler` through fusion-mlx's staged API (text-encode → denoise → vae-decode, strict offload between) for video-T2V and image-txt2img; auto-fallback to monolith for I2V/VACE/cascade/img2img.

**Architecture:** Add a `_should_use_staged` predicate (auto-detect from 4 wired latent/cond keys) + `_staged_pixels_to_numpy` normalizer (samplers.py), a `_run_staged_pipeline` orchestration helper on `FusionEngineWrapper` (engine_wrapper.py, calls 10 stage methods with `purge_memory` between each), and a `_generate_staged` async sibling to `_generate_monolithic`. `KSampler.sample` gains a 3-line dispatch: staged path → `_generate_staged`, else → `_generate_monolithic` (byte-for-byte unchanged). Both paths return pixel-frame numpy, so the existing `sample` result-wrap branches (samplers.py:330-354) stay untouched.

**Tech Stack:** Python 3.12, MLX, numpy, pytest + pytest-asyncio, unittest.mock. fusion-mlx `public_api` stage methods (async). venv: `/Users/dahai/fusion/.venv/bin/{pytest,ruff,python}`.

**Spec:** `docs/superpowers/specs/2026-08-26-p3-transparent-staged-default-design.md` (commit `175eabd`) — the plan argues from the spec; executors read both.

## Global Constraints

- Zero `import torch` in non-test code (P2 invariant — do not regress; grep-verify each task).
- 4-space indent, no docstrings, logging in every new function (user rule).
- Surgical: `_generate_monolithic` body stays byte-for-byte unchanged; only `sample` gets a 3-line dispatch added.
- Stage calls go through `FusionEngineWrapper` methods (engine_wrapper.py), NOT directly to `engine._engine`.
- `FusionMemoryGuardian.purge_memory()` between every stage (strict offload). Log each stage transition.
- No autograd wrappers, no model freeze (ComfyUI AGENTS.md).
- All tests: clean up process data after verification, keep only final outputs + logs (user rule).
- Real-model tests: load real model via `~/claude-home/fusion-mlx/start.sh start|stop`; mirror `https://hf-mirror.com` for downloads (user rule).
- **Ruling (architecture):** Route in `KSampler.sample` (3-line dispatch), NOT inside `_generate_monolithic`. Spec over-claimed "sample needs no change"; corrected here. `_generate_monolithic` stays pure monolith; `_generate_staged` is the new sibling. Both return pixel-frame numpy → same `sample` result-wrap branch (samplers.py:330-345) unchanged beyond the dispatch. Cost if wrong: rename + move 3 lines — trivial.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `fusion_comfyui_plugin/nodes/samplers.py` | `_should_use_staged` predicate, `_staged_pixels_to_numpy` normalizer, `_generate_staged` async fn, `sample` 3-line dispatch | Modify |
| `fusion_comfyui/core/engine_wrapper.py` | `_run_staged_pipeline` orchestration helper (10 stage calls + purge between) | Modify |
| `fusion_comfyui_plugin/tests/test_staged_routing.py` | Routing matrix + staged happy-path + fallback regression + normalizer tests | Create |
| `fusion_comfyui_plugin/tests/test_engine_wrapper.py` | `_run_staged_pipeline` orchestration tests (stage order, purge, cfg-skip-neg) | Modify (append) |
| `tests/test_e2e_p3_staged.py` | Real-model staged T2V e2e (inference-gated, auto-skip in CI) | Create |
| `README.md` | P3 entry | Modify |
| `CONSTRUCTION_PLAN.md` | P3 checkmark | Modify |

---

## Task 1: `_should_use_staged` predicate + routing-matrix tests

**Files:**
- Modify: `fusion_comfyui_plugin/nodes/samplers.py` (add `_should_use_staged` near top, after imports/before `_generate_monolithic`)
- Test: `fusion_comfyui_plugin/tests/test_staged_routing.py` (create)

**Interfaces:**
- Consumes: `model_wrapper.model_type` (str "video"|"image"), `positive`/`negative` (dict or None, may carry `stable_cascade_prior`), `latent_image` (dict, may carry `_i2v_image_path`, `_vace_control_video`, `_vace_control_mask`, `_vace_reference_images`, `_image_init_path`), `denoise` (float).
- Produces: `_should_use_staged(model_wrapper, positive, negative, latent_image, denoise) -> bool`.

- [ ] **Step 1: Write failing routing-matrix tests**

Create `fusion_comfyui_plugin/tests/test_staged_routing.py`:

```python
import numpy as np
import pytest
from unittest.mock import MagicMock


def _model(model_type):
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    m = MagicMock(spec=FusionModelWrapper)
    m.model_type = model_type
    return m


def _latent(**extra):
    base = {"samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32)}
    base.update(extra)
    return base


@pytest.mark.parametrize("model_type,latent_extra,denoise,positive_extra,expected", [
    ("video", {}, 1.0, {}, True),                                   # T2V pure text -> staged
    ("video", {"_i2v_image_path": "/tmp/x.png"}, 1.0, {}, False),   # I2V -> monolith
    ("video", {"_vace_control_video": "/tmp/v.mp4"}, 1.0, {}, False),  # VACE ctrl -> monolith
    ("video", {"_vace_control_mask": "/tmp/m.png"}, 1.0, {}, False),   # VACE mask -> monolith
    ("video", {"_vace_reference_images": "/tmp/r.png"}, 1.0, {}, False),  # VACE ref -> monolith
    ("image", {"_image_init_path": "/tmp/i.png"}, 0.5, {}, False),  # img2img (denoise<1) -> monolith
    ("image", {"_image_init_path": "/tmp/i.png"}, 1.0, {}, True),   # txt2img w/ stale init key, denoise=1 -> staged
    ("image", {}, 1.0, {}, True),                                   # txt2img FLUX.2 -> staged
    ("image", {}, 1.0, {"stable_cascade_prior": np.zeros((64, 64, 3))}, False),  # cascade stage_b -> pass-through
])
def test_should_use_staged_matrix(model_type, latent_extra, denoise, positive_extra, expected):
    from nodes.samplers import _should_use_staged
    model = _model(model_type)
    positive = {"prompt": "p", **positive_extra}
    negative = {"prompt": "n"}
    latent = _latent(**latent_extra)
    assert _should_use_staged(model, positive, negative, latent, denoise) is expected


def test_should_use_staged_negative_none():
    from nodes.samplers import _should_use_staged
    model = _model("video")
    assert _should_use_staged(model, {"prompt": "p"}, None, _latent(), 1.0) is True


def test_should_use_staged_unknown_type():
    from nodes.samplers import _should_use_staged
    model = _model("audio")  # neither video nor image
    assert _should_use_staged(model, {"prompt": "p"}, None, _latent(), 1.0) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/test_staged_routing.py -v`
Expected: FAIL with `ImportError: cannot import name '_should_use_staged'`

- [ ] **Step 3: Implement `_should_use_staged`**

Add to `fusion_comfyui_plugin/nodes/samplers.py` after the `_upscale_init_image` function (before `async def _generate_monolithic`):

```python
def _should_use_staged(model_wrapper, positive, negative, latent_image, denoise):
    # Auto-detect: route to staged API only for video T2V (no i2v/vace inputs)
    # and image txt2img (no cascade prior, not img2img denoise<1). Everything
    # else (I2V, VACE, cascade stage_b, img2img) falls back to monolith.
    model_type = getattr(model_wrapper, "model_type", None)
    has_i2v = bool(latent_image.get("_i2v_image_path"))
    has_vace = bool(
        latent_image.get("_vace_control_video")
        or latent_image.get("_vace_control_mask")
        or latent_image.get("_vace_reference_images")
    )
    has_init = bool(latent_image.get("_image_init_path"))
    has_cascade_prior = False
    for cond in (positive, negative):
        if isinstance(cond, dict) and cond.get("stable_cascade_prior") is not None:
            has_cascade_prior = True
            break
    if model_type == "video":
        if has_i2v or has_vace:
            logger.info("_should_use_staged: video monolith (i2v=%s vace=%s)", has_i2v, has_vace)
            return False
        logger.info("_should_use_staged: video T2V -> staged")
        return True
    if model_type == "image":
        if has_cascade_prior:
            logger.info("_should_use_staged: image cascade stage_b -> pass-through (monolith)")
            return False
        if has_init and denoise is not None and denoise < 1.0:
            logger.info("_should_use_staged: image img2img denoise=%.2f -> monolith", denoise)
            return False
        logger.info("_should_use_staged: image txt2img -> staged")
        return True
    logger.info("_should_use_staged: unknown model_type=%s -> monolith", model_type)
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/test_staged_routing.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint + commit**

Run: `/Users/dahai/fusion/.venv/bin/ruff check fusion_comfyui_plugin/nodes/samplers.py fusion_comfyui_plugin/tests/test_staged_routing.py`
Expected: clean

```bash
git add fusion_comfyui_plugin/nodes/samplers.py fusion_comfyui_plugin/tests/test_staged_routing.py
git commit -m "feat(p3): add _should_use_staged routing predicate + matrix tests"
```

---

## Task 2: `_staged_pixels_to_numpy` normalizer + tests

**Files:**
- Modify: `fusion_comfyui_plugin/nodes/samplers.py` (add `_staged_pixels_to_numpy` after `_should_use_staged`)
- Test: `fusion_comfyui_plugin/tests/test_staged_routing.py` (append normalizer tests)

**Interfaces:**
- Consumes: `pixels` (mx.array OR np.ndarray, VAE-decode output float [0,1] OR uint8 fallback), `model_type` (str).
- Produces: `_staged_pixels_to_numpy(pixels, model_type) -> np.ndarray` — video `[T,H,W,3]` float32 [0,1]; image `[H,W,3]` float32 [0,1].

- [ ] **Step 1: Write failing normalizer tests**

Append to `fusion_comfyui_plugin/tests/test_staged_routing.py`:

```python
def test_staged_pixels_video_mx_array():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    # VAE decode returns float [0,1]; THWC
    pixels = mx.array(np.random.rand(4, 512, 768, 3).astype(np.float32))
    out = _staged_pixels_to_numpy(pixels, "video")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (4, 512, 768, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_staged_pixels_video_ndim5_squeezes():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    pixels = mx.array(np.random.rand(1, 4, 512, 768, 3).astype(np.float32))
    out = _staged_pixels_to_numpy(pixels, "video")
    assert out.shape == (4, 512, 768, 3)


def test_staged_pixels_image_nchw_to_hwc():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    # Image decode returns [batch,c,h,w]
    pixels = mx.array(np.random.rand(1, 3, 512, 512).astype(np.float32))
    out = _staged_pixels_to_numpy(pixels, "image")
    assert out.shape == (512, 512, 3)
    assert out.dtype == np.float32


def test_staged_pixels_image_4ch_slices_to_3():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    pixels = mx.array(np.random.rand(1, 4, 64, 64).astype(np.float32))
    out = _staged_pixels_to_numpy(pixels, "image")
    assert out.shape == (64, 64, 3)


def test_staged_pixels_clamps_out_of_range():
    import mlx.core as mx
    from nodes.samplers import _staged_pixels_to_numpy
    pixels = mx.array(np.full((2, 8, 8, 3), 2.0, dtype=np.float32))
    out = _staged_pixels_to_numpy(pixels, "video")
    assert out.max() <= 1.0


def test_staged_pixels_uint8_fallback_divides():
    from nodes.samplers import _staged_pixels_to_numpy
    # Defensive: if decode ever returns uint8 numpy, divide like monolith
    pixels = np.full((2, 8, 8, 3), 255, dtype=np.uint8)
    out = _staged_pixels_to_numpy(pixels, "video")
    assert out.dtype == np.float32
    assert out.max() <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/test_staged_routing.py -k staged_pixels -v`
Expected: FAIL with `ImportError: cannot import name '_staged_pixels_to_numpy'`

- [ ] **Step 3: Implement `_staged_pixels_to_numpy`**

Add to `fusion_comfyui_plugin/nodes/samplers.py` after `_should_use_staged`:

```python
def _staged_pixels_to_numpy(pixels, model_type):
    # Staged decode returns mx.array float [0,1] (NOT uint8 like monolith raw).
    # Normalize to the same numpy contract the monolith produces:
    #   video -> [T,H,W,3] float32 [0,1]
    #   image -> [H,W,3] float32 [0,1]
    if isinstance(pixels, mx.array):
        arr = np.array(pixels)
        mx.eval(pixels)
    else:
        arr = np.asarray(pixels)
    is_uint8 = arr.dtype != np.float32 and np.issubdtype(arr.dtype, np.integer)
    if is_uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    if model_type == "video":
        while arr.ndim > 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 5:
            arr = arr[0]
        logger.info("_staged_pixels_to_numpy: video out shape=%s", arr.shape)
        return arr
    # image: decode is [batch,c,h,w] -> [H,W,3]
    if arr.ndim == 4:
        if arr.shape[1] == 4:
            arr = arr[:, :3]
        arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.shape[0] == 1:
            arr = arr[0]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        if arr.shape[0] == 4:
            arr = arr[:3]
        arr = np.transpose(arr, (1, 2, 0))
    logger.info("_staged_pixels_to_numpy: image out shape=%s", arr.shape)
    return arr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/test_staged_routing.py -k staged_pixels -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint + commit**

Run: `/Users/dahai/fusion/.venv/bin/ruff check fusion_comfyui_plugin/nodes/samplers.py fusion_comfyui_plugin/tests/test_staged_routing.py`
Expected: clean

```bash
git add fusion_comfyui_plugin/nodes/samplers.py fusion_comfyui_plugin/tests/test_staged_routing.py
git commit -m "feat(p3): add _staged_pixels_to_numpy normalizer + tests"
```

---

## Task 3: `_run_staged_pipeline` orchestration on FusionEngineWrapper + tests

**Files:**
- Modify: `fusion_comfyui/core/engine_wrapper.py` (add `_run_staged_pipeline` method after `unload_vae`, ~line 287)
- Test: `fusion_comfyui_plugin/tests/test_engine_wrapper.py` (append `TestRunStagedPipeline` class)

**Interfaces:**
- Consumes: `self.load_text_encoder/encode_text/unload_text_encoder/load_dit/denoise/unload_dit/load_vae/decode/unload_vae` (existing wrapper methods, all async), `FusionMemoryGuardian.purge_memory()` (existing), `self.model_type` (str).
- Produces: `async def _run_staged_pipeline(self, latent, prompt, neg_prompt, steps, cfg, seed, num_frames=None) -> mx.array` — returns the raw decoded pixels mx.array (caller normalizes via `_staged_pixels_to_numpy`).

- [ ] **Step 1: Write failing orchestration tests**

Append to `fusion_comfyui_plugin/tests/test_engine_wrapper.py`:

```python
class TestRunStagedPipeline:
    def _make_wrapper(self, model_type="video"):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        w = MagicMock(spec=FusionEngineWrapper)
        w.model_type = model_type
        w.load_text_encoder = AsyncMock()
        w.encode_text = AsyncMock(side_effect=lambda p, neg="": {"embed": mx.array(np.zeros((1, 256), dtype=np.float32))})
        w.unload_text_encoder = AsyncMock()
        w.load_dit = AsyncMock()
        w.denoise = AsyncMock(return_value=mx.array(np.zeros((1, 16, 5, 32, 32), dtype=np.float32)))
        w.unload_dit = AsyncMock()
        w.load_vae = AsyncMock()
        w.decode = AsyncMock(return_value=mx.array(np.zeros((4, 512, 768, 3), dtype=np.float32)))
        w.unload_vae = AsyncMock()
        return w

    def test_video_full_stage_order_and_purge(self):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        w = self._make_wrapper("video")
        import asyncio
        with patch("fusion_comfyui.core.engine_wrapper.FusionMemoryGuardian.purge_memory") as purge:
            result = asyncio.run(w._run_staged_pipeline(
                mx.array(np.zeros((1, 16, 5, 32, 32))), "cat", "dog", 20, 6.0, 42, num_frames=41))
        assert isinstance(result, mx.array)
        # Stage call order
        w.load_text_encoder.assert_awaited_once()
        assert w.encode_text.await_count == 2  # pos + neg (cfg>1)
        w.unload_text_encoder.assert_awaited_once()
        w.load_dit.assert_awaited_once()
        w.denoise.assert_awaited_once()
        w.unload_dit.assert_awaited_once()
        w.load_vae.assert_awaited_once()
        w.decode.assert_awaited_once()
        w.unload_vae.assert_awaited_once()
        # purge between each of the 3 stages
        assert purge.call_count == 3

    def test_cfg_le_1_skips_negative_encode(self):
        w = self._make_wrapper("video")
        import asyncio
        asyncio.run(w._run_staged_pipeline(
            mx.array(np.zeros((1, 16, 5, 32, 32))), "cat", "dog", 20, 1.0, 42, num_frames=41))
        assert w.encode_text.await_count == 1  # positive only
        # denoise called with neg_cond=None
        denoise_args = w.denoise.await_args
        assert denoise_args.args[2] is None or denoise_args.kwargs.get("negative") is None

    def test_image_pipeline_no_num_frames(self):
        w = self._make_wrapper("image")
        import asyncio
        result = asyncio.run(w._run_staged_pipeline(
            mx.array(np.zeros((1, 16, 32, 32))), "cat", "dog", 20, 6.0, 42))
        w.denoise.assert_awaited_once()
        w.decode.assert_awaited_once()
        assert isinstance(result, mx.array)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/test_engine_wrapper.py::TestRunStagedPipeline -v`
Expected: FAIL with `AttributeError: '_run_staged_pipeline'` or mock spec missing it.

- [ ] **Step 3: Implement `_run_staged_pipeline`**

Add to `fusion_comfyui/core/engine_wrapper.py` after `unload_vae` (line ~287), inside the `FusionEngineWrapper` class:

```python
    async def _run_staged_pipeline(self, latent, prompt, neg_prompt, steps, cfg, seed, num_frames=None):
        # Staged: text-encode -> denoise -> vae-decode, strict offload between.
        # Returns decoded pixels mx.array (float [0,1]); caller normalizes to numpy.
        logger.info("_run_staged_pipeline: start type=%s steps=%d cfg=%.1f seed=%d", self.model_type, steps, cfg, seed)
        await self.load_text_encoder()
        try:
            pos_cond = await self.encode_text(prompt)
            neg_cond = await self.encode_text(neg_prompt) if (cfg > 1.0 and neg_prompt) else None
        finally:
            await self.unload_text_encoder()
        FusionMemoryGuardian.purge_memory()
        logger.info("_run_staged_pipeline: text stage done, pos+neg encoded=%s", neg_cond is not None)

        await self.load_dit()
        try:
            if self.model_type == "video":
                latent = await self.denoise(latent, pos_cond, neg_cond, steps=steps, cfg=cfg, seed=seed, num_frames=num_frames)
            else:
                latent = await self.denoise(latent, pos_cond, neg_cond, steps=steps, cfg=cfg, seed=seed)
        finally:
            await self.unload_dit()
        FusionMemoryGuardian.purge_memory()
        logger.info("_run_staged_pipeline: denoise stage done, latent shape=%s", tuple(latent.shape))

        await self.load_vae()
        try:
            pixels = await self.decode(latent)
        finally:
            await self.unload_vae()
        FusionMemoryGuardian.purge_memory()
        logger.info("_run_staged_pipeline: vae stage done, pixels shape=%s", tuple(pixels.shape))
        return pixels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/test_engine_wrapper.py::TestRunStagedPipeline -v`
Expected: PASS (3 tests). If the mock-spec test fails because MagicMock(spec=...) blocks the new method, drop `spec=FusionEngineWrapper` in `_make_wrapper` and assert via the real class import instead.

- [ ] **Step 5: Lint + commit**

Run: `/Users/dahai/fusion/.venv/bin/ruff check fusion_comfyui/core/engine_wrapper.py fusion_comfyui_plugin/tests/test_engine_wrapper.py`
Expected: clean

```bash
git add fusion_comfyui/core/engine_wrapper.py fusion_comfyui_plugin/tests/test_engine_wrapper.py
git commit -m "feat(p3): add _run_staged_pipeline orchestration on FusionEngineWrapper"
```

---

## Task 4: `_generate_staged` async fn + `sample` dispatch + integration/fallback tests

**Files:**
- Modify: `fusion_comfyui_plugin/nodes/samplers.py` (add `_generate_staged` after `_generate_monolithic`; add 3-line dispatch in `KSampler.sample` before `run_async`)
- Test: `fusion_comfyui_plugin/tests/test_staged_routing.py` (append staged-happy-path + fallback-regression tests)

**Interfaces:**
- Consumes: `_should_use_staged` (Task 1), `_staged_pixels_to_numpy` (Task 2), `FusionEngineWrapper._run_staged_pipeline` (Task 3), `model_wrapper.get_engine()`, `model_wrapper.model_type`, `positive.get("prompt")`/`negative.get("prompt")`, `fusion_comfyui.core.async_utils.run_async`.
- Produces: `async def _generate_staged(model_wrapper, positive, negative, latent_image, steps, cfg, seed, width, height, num_frames, denoise=1.0, sampler_name="euler") -> np.ndarray` (pixel frames, same contract as monolith image/video output).

- [ ] **Step 1: Write failing integration tests**

Append to `fusion_comfyui_plugin/tests/test_staged_routing.py`:

```python
def _staged_mock_model(model_type="video"):
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    mock = MagicMock(spec=FusionModelWrapper)
    mock.model_type = model_type
    mock.model_name = f"staged-{model_type}"
    engine = MagicMock()
    engine.ensure_started = AsyncMock()
    engine._run_staged_pipeline = AsyncMock(
        return_value=mx.array(np.random.rand(4, 512, 768, 3).astype(np.float32))
    )
    mock.get_engine = MagicMock(return_value=engine)
    return mock


class TestGenerateStaged:
    def test_video_staged_calls_pipeline(self):
        from nodes.samplers import _generate_staged
        model = _staged_mock_model("video")
        positive = {"prompt": "cat playing"}
        negative = {"prompt": "blurry"}
        latent = {"samples": np.zeros((1, 16, 5, 32, 32), dtype=np.float32), "num_frames": 41, "width": 768, "height": 512}
        import asyncio
        result = asyncio.run(_generate_staged(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41))
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 3 and result.shape[-1] == 3
        engine = model.get_engine.return_value
        engine._run_staged_pipeline.assert_awaited_once()
        call = engine._run_staged_pipeline.await_args
        assert call.kwargs["prompt"] == "cat playing"
        assert call.kwargs["neg_prompt"] == "blurry"
        assert call.kwargs["num_frames"] == 41

    def test_image_staged_calls_pipeline(self):
        from nodes.samplers import _generate_staged
        model = _staged_mock_model("image")
        model.get_engine.return_value._run_staged_pipeline = AsyncMock(
            return_value=mx.array(np.random.rand(1, 3, 512, 512).astype(np.float32))
        )
        positive = {"prompt": "a cat"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32), "width": 512, "height": 512}
        import asyncio
        result = asyncio.run(_generate_staged(model, positive, negative, latent, 20, 6.0, 42, 512, 512, 1))
        assert isinstance(result, np.ndarray)
        assert result.shape == (512, 512, 3)

    def test_staged_negative_none(self):
        from nodes.samplers import _generate_staged
        model = _staged_mock_model("video")
        positive = {"prompt": "cat"}
        negative = None
        latent = {"samples": np.zeros((1, 16, 5, 32, 32), dtype=np.float32), "num_frames": 41, "width": 768, "height": 512}
        import asyncio
        asyncio.run(_generate_staged(model, positive, negative, latent, 20, 6.0, 42, 768, 512, 41))
        call = model.get_engine.return_value._run_staged_pipeline.await_args
        assert call.kwargs["neg_prompt"] == ""


class TestSampleDispatch:
    def test_t2v_routes_to_staged(self):
        from nodes.samplers import KSampler
        model = _staged_mock_model("video")
        positive = {"prompt": "cat"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32), "num_frames": 41, "width": 768, "height": 512}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"):
            result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        model.get_engine.return_value._run_staged_pipeline.assert_awaited_once()
        assert "_decoded_frames_key" in result[0]

    def test_i2v_routes_to_monolith(self):
        from nodes.samplers import KSampler
        model = _make_mock_model_monolith("video")  # helper below
        positive = {"prompt": "cat"}
        negative = {"prompt": "bad"}
        latent = {"samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32), "_i2v_image_path": "/tmp/x.png"}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=np.zeros((4, 512, 768, 3), dtype=np.float32)) as ra:
            node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        # monolith path: run_async called with _generate_monolithic coroutine
        assert ra.called

    def test_cascade_routes_to_monolith(self):
        from nodes.samplers import KSampler
        model = _make_mock_model_monolith("image")
        prior = np.zeros((64, 64, 3), dtype=np.float32)
        positive = {"prompt": "p", "stable_cascade_prior": prior}
        negative = {"prompt": "n"}
        latent = {"samples": np.zeros((1, 4, 64, 64), dtype=np.float32)}
        node = KSampler()
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=prior) as ra:
            node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
        assert ra.called
```

Add the monolith mock helper near top of test file:

```python
def _make_mock_model_monolith(model_type="image"):
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    mock = MagicMock(spec=FusionModelWrapper)
    mock.model_type = model_type
    mock.model_name = f"mono-{model_type}"
    engine = MagicMock()
    engine.ensure_started = AsyncMock()
    mock.get_engine = MagicMock(return_value=engine)
    return mock
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/test_staged_routing.py -k "GenerateStaged or SampleDispatch" -v`
Expected: FAIL with `ImportError: cannot import name '_generate_staged'`

- [ ] **Step 3: Implement `_generate_staged`**

Add to `fusion_comfyui_plugin/nodes/samplers.py` after `async def _generate_monolithic` (after line 245):

```python
async def _generate_staged(model_wrapper, positive, negative, latent_image,
                           steps, cfg, seed, width, height, num_frames, denoise=1.0,
                           sampler_name="euler"):
    engine = model_wrapper.get_engine()
    await engine.ensure_started()
    prompt = positive.get("prompt", "")
    neg_prompt = negative.get("prompt", "") if negative else ""
    mlx_latent = latent_image["samples"]
    if not isinstance(mlx_latent, mx.array):
        from fusion_comfyui.core.bridge import to_mlx_array
        mlx_latent = to_mlx_array(mlx_latent)
    logger.info(
        "_generate_staged: model=%s steps=%d cfg=%.1f seed=%d frames=%d %dx%d",
        model_wrapper.model_name, steps, cfg, seed, num_frames, width, height,
    )
    pixels = await engine._run_staged_pipeline(
        mlx_latent, prompt, neg_prompt, steps, cfg, seed, num_frames=num_frames,
    )
    return _staged_pixels_to_numpy(pixels, model_wrapper.model_type)
```

- [ ] **Step 4: Add 3-line dispatch in `KSampler.sample`**

In `fusion_comfyui_plugin/nodes/samplers.py`, find the `run_async(_generate_monolithic(...))` call (line ~316). Replace it with a dispatch:

```python
        if _should_use_staged(model, positive, negative, latent_image, denoise):
            logger.info("KSampler: staged path selected for %s", model.model_name)
            generate_coro = _generate_staged(
                model, positive, negative, latent_image,
                steps, cfg, seed, width, height, num_frames, denoise=denoise,
                sampler_name=sampler_name,
            )
        else:
            generate_coro = _generate_monolithic(
                model, positive, negative, latent_image,
                steps, cfg, seed, width, height, num_frames, denoise=denoise,
                sampler_name=sampler_name,
            )
        result = fusion_comfyui.core.async_utils.run_async(generate_coro, timeout=3600)
```

(This replaces the existing `result = fusion_comfyui.core.async_utils.run_async(_generate_monolithic(...), timeout=3600)` block — lines 316-323.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/test_staged_routing.py -v`
Expected: PASS (all staged-routing tests)

- [ ] **Step 6: Run full sampler suite to verify no regressions**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/test_samplers.py fusion_comfyui_plugin/tests/test_staged_routing.py -v`
Expected: PASS (existing monolith tests still green — `_generate_monolithic` byte-for-byte, dispatch falls through to monolith for their inputs)

- [ ] **Step 7: Lint + commit**

Run: `/Users/dahai/fusion/.venv/bin/ruff check fusion_comfyui_plugin/nodes/samplers.py fusion_comfyui_plugin/tests/test_staged_routing.py`
Expected: clean

```bash
git add fusion_comfyui_plugin/nodes/samplers.py fusion_comfyui_plugin/tests/test_staged_routing.py
git commit -m "feat(p3): add _generate_staged + sample dispatch, keep monolith fallback"
```

---

## Task 5: Full plugin suite + zero-torch grep + cleanup

**Files:** none (verification only)

- [ ] **Step 1: Run full plugin test suite**

Run: `/Users/dahai/fusion/.venv/bin/pytest fusion_comfyui_plugin/tests/ -v`
Expected: all PASS (449 prior + new staged tests, 0 failures). If a prior test breaks, the dispatch mis-routed — fix `_should_use_staged`, do not touch `_generate_monolithic`.

- [ ] **Step 2: Verify zero `import torch` in non-test code**

Run: `cd /Users/dahai/fusion/fusion-comfyui && grep -rn "import torch" fusion_comfyui/ fusion_comfyui_plugin/ --include="*.py" | grep -v "/tests/" | grep -v test_`
Expected: empty (P2 invariant holds)

- [ ] **Step 3: Ruff clean across touched files**

Run: `/Users/dahai/fusion/.venv/bin/ruff check fusion_comfyui/core/engine_wrapper.py fusion_comfyui_plugin/nodes/samplers.py fusion_comfyui_plugin/tests/`
Expected: clean

- [ ] **Step 4: Commit any fixups (if none, skip)**

```bash
git status --porcelain
# if clean, no commit; if fixups:
git add -A && git commit -m "fix(p3): suite-green fixups"
```

---

## Task 6: Real-model staged T2V e2e (inference-gated)

**Files:**
- Create: `tests/test_e2e_p3_staged.py`

**Context:** User rule — real-model tests must load real model. Start fusion-mlx via `~/claude-home/fusion-mlx/start.sh start`. Use mirror `https://hf-mirror.com` if model download needed. This test is `pytest.mark.inference` — auto-skip in CI (no GPU/model), but MUST run locally for exit-criteria #6.

- [ ] **Step 1: Check fusion-mlx staged e2e pattern**

Read `tests/test_e2e_wan2_staged.py` for the existing staged e2e harness (model load, KSampler or FusionKSampler call, video assertion, cleanup). Mirror its structure.

- [ ] **Step 2: Write the e2e test**

Create `tests/test_e2e_p3_staged.py`:

```python
import logging
import os
import pytest
import numpy as np

logger = logging.getLogger("fusion_comfyui.e2e.p3_staged")

pytestmark = pytest.mark.inference


@pytest.mark.asyncio
async def test_staged_t2v_real_model(tmp_path):
    # Exit criteria #6: native KSampler transparent staged path produces valid
    # video T2V; memory sawtooth visible in logs (load/unload between stages).
    from fusion_comfyui_plugin.nodes.samplers import KSampler
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    # Load a small T2V model (wan2 1.3B) via the wrapper — mirror test_e2e_wan2_staged
    model = FusionModelWrapper(model_path="wan2.1-t2v-1.3B", model_type="video")
    positive = {"prompt": "a cat walking on a beach, cinematic"}
    negative = {"prompt": "blurry, low quality"}
    latent = {
        "samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32),
        "num_frames": 41, "width": 768, "height": 512,
    }
    node = KSampler()
    result = node.sample(model, 42, 20, 6.0, "euler", "normal", positive, negative, latent, denoise=1.0)
    out = result[0]
    assert "_decoded_frames_key" in out
    frames = out["samples"]
    assert frames.ndim >= 4 and frames.shape[-1] == 3
    # Valid video: not all-black, not all-NaN
    assert not np.isnan(frames).any()
    assert frames.std() > 1e-3
    logger.info("test_staged_t2v_real_model: frames shape=%s std=%.4f", frames.shape, frames.std())
    # Cleanup: clear cache, stop models
    import mlx.core as mx
    mx.metal.clear_cache()
```

- [ ] **Step 3: Start fusion-mlx + run the e2e**

```bash
~/claude-home/fusion-mlx/start.sh start
/Users/dahai/fusion/.venv/bin/pytest tests/test_e2e_p3_staged.py -v -s
```

Expected: PASS. Check logs for the sawtooth: `stage loaded: text_encoder` → `stage unloaded: text_encoder` → `stage loaded: dit` → ... → `stage unloaded: vae`. If model download needed, set `HF_MIRROR=https://hf-mirror.com`.

If the test fails because the 1.3B model isn't present, download via mirror first, then re-run. Do NOT skip — user rule requires real model.

- [ ] **Step 4: Verify memory sawtooth in logs**

Run: `~/claude-home/fusion-mlx/start.sh log | grep -E "stage (loaded|unloaded)"`
Expected: 6 transitions (load+unload for text_encoder, dit, vae). This is the Phase-1 sawtooth exit criterion.

- [ ] **Step 5: Cleanup process data + commit**

```bash
~/claude-home/fusion-mlx/start.sh stop
# Remove temp outputs, keep only logs
rm -rf /tmp/fusion_i2i_* /tmp/fusion_p3_* 2>/dev/null || true
git add tests/test_e2e_p3_staged.py
git commit -m "test(p3): real-model staged T2V e2e (inference-gated)"
```

---

## Task 7: Docs (README P3 entry + CONSTRUCTION_PLAN checkmark) + fusion-mlx I2V/VACE issue

**Files:**
- Modify: `README.md`
- Modify: `CONSTRUCTION_PLAN.md`

- [ ] **Step 1: Add P3 entry to README**

In `README.md`, find the version/changelog section. Add (match surrounding style):

```markdown
### P3 — Transparent Staged Default (2026-08-26)

Native `KSampler` now routes video-T2V and image-txt2img through the fusion-mlx
staged API (text-encode → denoise → vae-decode) with strict model offload between
stages — memory sawtooth, peak = largest single stage. I2V/VACE/cascade/img2img
auto-fallback to the monolith path unchanged. Explicit `FusionKSampler`/
`TextEncoder`/`VAEDecoder` nodes remain for cacheable explicit-stage graphs.

- Auto-detect keys: `_i2v_image_path`, `_vace_control_video`/`_mask`/`_ref`, `stable_cascade_prior`, `_image_init_path`+denoise<1.0.
- I2V/VACE staged gap filed upstream (fusion-mlx issue <N>).
```

- [ ] **Step 2: Mark P3 in CONSTRUCTION_PLAN**

In `CONSTRUCTION_PLAN.md`, find the phase roadmap. Add a P3 completion marker referencing this plan + spec. Match existing checkmark style.

- [ ] **Step 3: File fusion-mlx issue for I2V/VACE staged gap**

```bash
gh issue create --repo dahai80/fusion-mlx \
  --title "Staged API for I2V/VACE/camera (currently monolith-only)" \
  --body "The staged API (load_text_encoder/encode_text/denoise/decode) is implemented for Wan2 T2V, SkyReels, and FLUX.2 image. Video I2V, VACE, and camera paths have no staged implementation (wan2.py:350, stage.py:7) — fusion-comfyui P3 must route these to monolith. Request: staged I2V/VACE so the transparent staged default covers all video paths."
```

Record the issue number in the README P3 entry (`<N>` placeholder) and commit.

- [ ] **Step 4: Commit docs**

```bash
git add README.md CONSTRUCTION_PLAN.md
git commit -m "docs(p3): README P3 entry + CONSTRUCTION_PLAN checkmark + upstream I2V/VACE issue"
```

---

## Self-Review

**1. Spec coverage:**
- Decision 1 (transparent staged default) → Task 4 (`sample` dispatch, no new node). ✓
- Decision 2 (auto-detect fallback) → Task 1 (`_should_use_staged`, 4 wired keys). ✓
- Decision 3 (all stages inside KSampler) → Task 3 (`_run_staged_pipeline` called from `_generate_staged` called from `sample`). ✓
- Decision 4 (strict sequential offload) → Task 3 (`purge_memory` between each of 3 stages, `unload_*` in `finally`). ✓
- Decision 5 (keep explicit nodes) → no task modifies `FusionKSamplerNode`/`TextEncoder`/`VAEDecoder`. ✓
- Auto-detect matrix (6 rows) → Task 1 parametrized tests. ✓
- Staged pipeline sequence → Task 3 + verified in Task 4. ✓
- Result normalization → Task 2. ✓
- I2V/VACE upstream issue → Task 7. ✓
- Exit criteria (1-8) → Tasks 1-7. ✓

**2. Placeholder scan:** `<N>` in README (Task 7) is filled at run-time from `gh issue create` output — acceptable, not a plan placeholder. No TBD/TODO in tasks.

**3. Type consistency:**
- `_should_use_staged(model_wrapper, positive, negative, latent_image, denoise) -> bool` — same in Task 1 + Task 4 dispatch. ✓
- `_staged_pixels_to_numpy(pixels, model_type) -> np.ndarray` — same Task 2 + Task 4. ✓
- `_run_staged_pipeline(self, latent, prompt, neg_prompt, steps, cfg, seed, num_frames=None) -> mx.array` — same Task 3 + Task 4 call. ✓
- `_generate_staged(model_wrapper, positive, negative, latent_image, steps, cfg, seed, width, height, num_frames, denoise=1.0, sampler_name="euler")` — matches `_generate_monolithic` signature (Task 4). ✓
- `neg_cond=None` for cfg<=1.0 — Task 3 encode-skip + Task 1 matrix + spec. ✓

**4. Risk check:** Both spec open risks RESOLVED pre-plan (SkyReels parity, cfg negative). No remaining plan-task placeholders.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-26-p3-transparent-staged-default.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session, batch with checkpoints.

Which approach?
