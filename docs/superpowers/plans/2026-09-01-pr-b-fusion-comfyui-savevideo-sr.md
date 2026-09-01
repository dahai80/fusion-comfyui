# PR-B: fusion-comfyui SR-aware SaveVideo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the H3 `SaveVideo` node to super-resolve each frame (via fusion-mlx `POST /v1/images/super-resolution`, PR-A) before re-encoding the mp4, toggled by environment variables — zero changes to the AICF orchestration shell (off-limits).

**Architecture:** Inside `SaveVideo.save()` (`fusion_comfyui_plugin/nodes/h3.py:262`), after the H3 frames arrive as `video["images"]` (numpy float32), check `FUSION_H3_SR`. `0` = unchanged. `1` = per-frame HTTP POST (multipart PNG) to fusion-mlx, decode returned `image_b64`, assemble SR frame array, re-encode via the existing `FusionSaveVideoNode._encode_video_av`. `mock` = PIL lanczos resize (no HTTP) for env-trigger + AICF-chain verification before PR-A ships. Fail-loud: any SR failure raises `RuntimeError` (no silent fallback to the blurry original).

**Tech Stack:** numpy, PIL, httpx (sync client, mirrors `nodes/voice.py:279`), PyAV (existing `_encode_video_av`), pytest, pytest-asyncio, ruff.

**Spec:** `docs/superpowers/specs/2026-09-01-h3-video-sr-design.md` (PR-B section)

## Global Constraints

- AICF source/JSON OFF-LIMITS — SR is an env var on the fusion-comfyui node, invisible to AICF. AICF downloads the same `/history` output (now SR'd).
- Fail-loud (project Rule 12): SR opt-in (`FUSION_H3_SR=1`); on failure raise, do NOT hand back the blurry original.
- 4-space multiples indentation, no docstrings, always include logging.
- httpx sync `Client` pattern (mirror `nodes/voice.py:279` — `with httpx.Client(timeout=...) as client: client.post(...)`), retry transient 5xx, raise `RuntimeError` on non-transient/timeout.
- Real-model e2e (Task 5) needs fusion-mlx running with the SR model — use `~/claude-home/fusion-mlx/start.sh start|stop`, mirror download via `https://hf-mirror.com`.
- All file paths relative to `/Users/dahai/fusion/fusion-comfyui/`.
- Cleanup: delete test intermediates after verification; keep only final mp4 + logs.

## File Structure

- Modify: `fusion_comfyui_plugin/nodes/h3.py` — add SR helpers (`_sr_mode`, `_sr_frames_real`, `_sr_frames_mock`) + extend `SaveVideo.save()`.
- Create: `fusion_comfyui_plugin/nodes/sr.py` — SR frame processing (real HTTP + mock lanczos), kept out of `h3.py` to keep that file focused. `h3.py` imports `super_resolve_frames` from here.
- Test: `tests/test_h3_savevideo_sr.py` — TDD: env-trigger modes, mock dims, real-endpoint stub, fail-loud on 5xx/timeout.

---

## Task 1: sr.py — mock lanczos mode (no HTTP)

**Files:**
- Create: `fusion_comfyui_plugin/nodes/sr.py`
- Test: `tests/test_h3_savevideo_sr.py`

**Interfaces:**
- Consumes: numpy frames `(n, h, w, 3)` float32 [0,1] (the `video["images"]` contract from `h3.py:269`).
- Produces:
  - `def super_resolve_frames(images: np.ndarray, mode: str, scale: int = 4, endpoint: str = "http://127.0.0.1:11434/v1/images/super-resolution") -> np.ndarray` — dispatch entry. `mode` in `{"0","1","mock"}`. `mode=="0"` returns `images` unchanged. `mode=="mock"` → `_mock_resize`. `mode=="1"` → `_real_sr`. Returns `(n, h*scale, w*scale, 3)` float32.
  - `def _mock_resize(images, scale) -> np.ndarray` — PIL lanczos resize each frame by `scale`, no HTTP.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_h3_savevideo_sr.py
import numpy as np

from fusion_comfyui_plugin.nodes.sr import super_resolve_frames


def test_sr_mode_0_passthrough():
    imgs = np.random.rand(3, 16, 20, 3).astype(np.float32)
    out = super_resolve_frames(imgs, mode="0", scale=4)
    assert out.shape == (3, 16, 20, 3)
    assert np.array_equal(out, imgs)


def test_sr_mock_resize_4x_dims():
    imgs = np.random.rand(2, 16, 20, 3).astype(np.float32)
    out = super_resolve_frames(imgs, mode="mock", scale=4)
    assert out.shape == (2, 64, 80, 3), out.shape
    assert out.dtype == np.float32


def test_sr_mock_resize_2x_dims():
    imgs = np.zeros((1, 32, 40, 3), dtype=np.float32)
    out = super_resolve_frames(imgs, mode="mock", scale=2)
    assert out.shape == (1, 64, 80, 3), out.shape


def test_sr_mock_preserves_range():
    imgs = np.zeros((1, 12, 12, 3), dtype=np.float32)
    out = super_resolve_frames(imgs, mode="mock", scale=4)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_h3_savevideo_sr.py::test_sr_mode_0_passthrough tests/test_h3_savevideo_sr.py::test_sr_mock_resize_4x_dims -v`
Expected: FAIL `ModuleNotFoundError: No module named 'fusion_comfyui_plugin.nodes.sr'`

- [ ] **Step 3: Write minimal implementation**

```python
# fusion_comfyui_plugin/nodes/sr.py
import io
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
    # mode: "0" passthrough | "1" real fusion-mlx SR | "mock" lanczos placeholder.
    if mode == "0" or not mode:
        return images
    if mode == "mock":
        return _mock_resize(images, scale)
    if mode == "1":
        return _real_sr(images, scale, endpoint)
    raise ValueError(f"unknown FUSION_H3_SR mode: {mode!r}")


def _real_sr(images, scale, endpoint):
    raise NotImplementedError("real SR implemented in Task 2")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_h3_savevideo_sr.py -v`
Expected: all 4 PASS (the `_real_sr` NotImplementedError is not exercised yet)

- [ ] **Step 5: Commit**

```bash
git add fusion_comfyui_plugin/nodes/sr.py tests/test_h3_savevideo_sr.py
git commit -m "feat(sr): add sr.py mock lanczos mode + mode dispatch"
```

---

## Task 2: sr.py — real HTTP SR (per-frame POST, fail-loud)

**Files:**
- Modify: `fusion_comfyui_plugin/nodes/sr.py` (replace `_real_sr` stub)
- Test: `tests/test_h3_savevideo_sr.py` (add real-endpoint tests with httpx stub)

**Interfaces:**
- Consumes: PR-A endpoint `POST /v1/images/super-resolution` — multipart `files={"image": (name, png_bytes, "image/png")}`, `data={"scale": str, "tile_size": "512"}`; response JSON `{image_b64, width, height, in_width, in_height, elapsed}`. httpx sync `Client` (mirror `nodes/voice.py:279`).
- Produces: `def _real_sr(images, scale, endpoint) -> np.ndarray` — per-frame: numpy→PNG bytes (PIL), POST, decode `image_b64`→numpy, assemble. Fail-loud `RuntimeError` with frame index + endpoint + HTTP status on 5xx/timeout/empty. Retry transient 5xx (mirror voice.py, 3 attempts). Single-frame timeout default 60s (5376x3072 frame inference is heavy). Log per-frame + total elapsed.

- [ ] **Step 1: Write the failing test** (stub httpx.Client — no real endpoint in unit tests)

```python
# appended to tests/test_h3_savevideo_sr.py
import base64
from unittest.mock import patch


def _fake_sr_response(out_h, out_w):
    from PIL import Image
    arr = (np.random.rand(out_h, out_w, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode(),
            "width": out_w, "height": out_h,
            "in_width": out_w // 4, "in_height": out_h // 4, "elapsed": 0.1}


def test_sr_real_per_frame_post_called_n_times():
    import httpx
    from fusion_comfyui_plugin.nodes.sr import super_resolve_frames

    calls = []

    class FakeResp:
        def __init__(self, body):
            self.status_code = 200
            self._body = body
            self.text = ""

        def json(self):
            return self._body

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **k):
            calls.append((url, k.get("data", {}).get("scale")))
            return FakeResp(_fake_sr_response(8, 8))

    with patch("httpx.Client", FakeClient):
        imgs = np.random.rand(3, 8, 8, 3).astype(np.float32)
        out = super_resolve_frames(imgs, mode="1", scale=4)
    assert len(calls) == 3, calls
    assert out.shape == (3, 8, 8, 3), out.shape


def test_sr_real_5xx_raises_runtime_error_no_fallback():
    import httpx
    from fusion_comfyui_plugin.nodes.sr import super_resolve_frames

    class FakeResp:
        status_code = 500
        text = "boom"

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **k):
            return FakeResp()

    with patch("httpx.Client", FakeClient):
        imgs = np.zeros((2, 8, 8, 3), dtype=np.float32)
        with pytest.raises(RuntimeError):
            super_resolve_frames(imgs, mode="1", scale=4)


def test_sr_real_timeout_raises():
    import httpx
    from fusion_comfyui_plugin.nodes.sr import super_resolve_frames

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **k):
            raise httpx.TimeoutException("timed out")

    with patch("httpx.Client", FakeClient):
        imgs = np.zeros((1, 8, 8, 3), dtype=np.float32)
        with pytest.raises(RuntimeError):
            super_resolve_frames(imgs, mode="1", scale=4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_h3_savevideo_sr.py::test_sr_real_per_frame_post_called_n_times -v`
Expected: FAIL — `_real_sr` raises `NotImplementedError`

- [ ] **Step 3: Replace `_real_sr` stub**

```python
# in fusion_comfyui_plugin/nodes/sr.py — replace the stub _real_sr
def _real_sr(images, scale, endpoint):
    import httpx
    import time
    from PIL import Image

    n, h, w, _ = images.shape
    out = np.empty((n, h * scale, w * scale, 3), dtype=np.float32)
    url = f"{endpoint}" if endpoint.endswith("/super-resolution") \
        else f"{endpoint.rstrip('/')}/v1/images/super-resolution"
    logger.info("sr real: %d frames scale=%d endpoint=%s", n, scale, url)
    max_retries = 3
    t0 = time.time()
    for i in range(n):
        frame = (np.clip(images[i], 0, 1) * 255).astype(np.uint8)[:, :, :3]
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="PNG")
        png = buf.getvalue()
        ft0 = time.time()
        resp = None
        for attempt in range(1, max_retries + 1):
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        url,
                        files={"image": (f"frame_{i}.png", png, "image/png")},
                        data={"scale": str(scale), "tile_size": "512"},
                    )
            except Exception as e:
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
            raise RuntimeError(f"SR frame {i}: no successful response from {endpoint}")
        body = resp.json()
        sr_bytes = base64.b64decode(body["image_b64"])
        if not sr_bytes:
            raise RuntimeError(f"SR frame {i}: empty image_b64 from {endpoint}")
        sr_arr = np.array(Image.open(io.BytesIO(sr_bytes)).convert("RGB"),
                          dtype=np.float32) / 255.0
        out[i] = sr_arr
        logger.info("sr real: frame %d/%d in=%dx%d out=%dx%d elapsed=%.2fs",
                    i + 1, n, w, h, sr_arr.shape[1], sr_arr.shape[0],
                    time.time() - ft0)
    logger.info("sr real: done %d frames total elapsed=%.2fs", n, time.time() - t0)
    return out
```

Add `import base64` to the top imports of `sr.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_h3_savevideo_sr.py -v`
Expected: all 7 PASS (4 mock/passthrough + 3 real-endpoint stubs)

- [ ] **Step 5: Commit**

```bash
git add fusion_comfyui_plugin/nodes/sr.py tests/test_h3_savevideo_sr.py
git commit -m "feat(sr): add real per-frame HTTP SR with fail-loud + retry"
```

---

## Task 3: Wire SR into SaveVideo.save()

**Files:**
- Modify: `fusion_comfyui_plugin/nodes/h3.py:262-280` (`SaveVideo.save()`)
- Test: `tests/test_h3_savevideo_sr.py` (add SaveVideo integration tests)

**Interfaces:**
- Consumes: `super_resolve_frames` from `sr.py` (Tasks 1-2). Env vars: `FUSION_H3_SR` (`0`|`1`|`mock`), `FUSION_H3_SR_SCALE` (`4`|`2`, default 4), `FUSION_H3_SR_ENDPOINT` (default `http://127.0.0.1:11434/v1/images/super-resolution`).
- Produces: `SaveVideo.save()` runs SR on `images` before `_encode_video_av`. Output contract unchanged (`{"ui": {"videos": [...]}}`), so AICF downloads via `/history` unchanged. Dimensions now `h*scale x w*scale`.

- [ ] **Step 1: Write the failing test**

```python
# appended to tests/test_h3_savevideo_sr.py
class TestSaveVideoSR:
    def _setup_savevideo(self, monkeypatch, tmp_path, encoded):
        from fusion_comfyui_plugin.nodes import video_io as vio
        monkeypatch.setattr(
            vio.FusionSaveVideoNode, "_encode_video_av",
            lambda self, images, output_path, fps, codec, crf:
                (encoded.append((output_path, images.shape)), None)[1],
        )
        fake_fp = type(sys)("folder_paths")
        fake_fp.get_output_directory = lambda: str(tmp_path)
        fake_fp.get_save_image_path = lambda prefix, outdir, w, h: (
            str(tmp_path), prefix, 1, "", prefix)
        monkeypatch.setitem(sys.modules, "folder_paths", fake_fp)
        return encoded

    def test_savevideo_sr_off_unchanged_dims(self, monkeypatch, tmp_path):
        encoded = []
        self._setup_savevideo(monkeypatch, tmp_path, encoded)
        from fusion_comfyui_plugin.nodes.h3 import SaveVideo
        monkeypatch.delenv("FUSION_H3_SR", raising=False)
        video = {"images": np.zeros((2, 16, 20, 3), dtype=np.float32), "fps": 24.0}
        SaveVideo().save(video=video, filename_prefix="h3_sr")
        assert encoded
        _, shape = encoded[0]
        assert shape[1] == 16 and shape[2] == 20  # unchanged

    def test_savevideo_sr_mock_4x_dims(self, monkeypatch, tmp_path):
        encoded = []
        self._setup_savevideo(monkeypatch, tmp_path, encoded)
        monkeypatch.setenv("FUSION_H3_SR", "mock")
        from fusion_comfyui_plugin.nodes.h3 import SaveVideo
        video = {"images": np.zeros((2, 16, 20, 3), dtype=np.float32), "fps": 24.0}
        SaveVideo().save(video=video, filename_prefix="h3_sr")
        _, shape = encoded[0]
        assert shape[1] == 64 and shape[2] == 80  # 4x

    def test_savevideo_sr_mock_scale2_env(self, monkeypatch, tmp_path):
        encoded = []
        self._setup_savevideo(monkeypatch, tmp_path, encoded)
        monkeypatch.setenv("FUSION_H3_SR", "mock")
        monkeypatch.setenv("FUSION_H3_SR_SCALE", "2")
        from fusion_comfyui_plugin.nodes.h3 import SaveVideo
        video = {"images": np.zeros((2, 16, 20, 3), dtype=np.float32), "fps": 24.0}
        SaveVideo().save(video=video, filename_prefix="h3_sr")
        _, shape = encoded[0]
        assert shape[1] == 32 and shape[2] == 40  # 2x
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_h3_savevideo_sr.py::TestSaveVideoSR -v`
Expected: FAIL — SaveVideo ignores env, dims unchanged (mock test fails)

- [ ] **Step 3: Modify SaveVideo.save()**

In `fusion_comfyui_plugin/nodes/h3.py`, replace the body of `save()` (lines 262-280) — insert SR between reading `images` and calling `_encode_video_av`:

```python
    def save(self, video, filename_prefix="h3_t2v", format="auto", codec="auto"):
        import folder_paths
        from fusion_comfyui_plugin.nodes.sr import super_resolve_frames
        from fusion_comfyui_plugin.nodes.video_io import FusionSaveVideoNode
        images = video["images"] if isinstance(video, dict) else video
        fps = video.get("fps", 24.0) if isinstance(video, dict) else 24.0
        out_codec = "libx265" if codec == "libx265" else "libx264"
        # SR post-processing (env-triggered, AICF-invisible). 0=off, 1=real
        # fusion-mlx SR, mock=lanczos placeholder. Fail-loud: raises on SR
        # error, no silent fallback to the blurry original (project Rule 12).
        sr_mode = os.environ.get("FUSION_H3_SR", "0")
        sr_scale = int(os.environ.get("FUSION_H3_SR_SCALE", "4"))
        sr_endpoint = os.environ.get(
            "FUSION_H3_SR_ENDPOINT",
            "http://127.0.0.1:11434/v1/images/super-resolution")
        if sr_mode in ("1", "mock"):
            logger.info("SaveVideo(H3): SR mode=%s scale=%d endpoint=%s",
                        sr_mode, sr_scale, sr_endpoint)
            images = super_resolve_frames(images, mode=sr_mode, scale=sr_scale,
                                          endpoint=sr_endpoint)
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _prefix = \
            folder_paths.get_save_image_path(filename_prefix, output_dir, images.shape[2], images.shape[1])
        file = f"{filename}_{counter:05d}_.mp4"
        filepath = os.path.join(full_output_folder, file)
        helper = FusionSaveVideoNode()
        helper._encode_video_av(images, filepath, int(fps), out_codec, 18)
        logger.info("SaveVideo(H3): saved %s frames=%s fps=%.1f sr=%s",
                    filepath, images.shape, fps, sr_mode)
        return {"ui": {"videos": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_h3_savevideo_sr.py -v`
Expected: all PASS (existing `TestSaveVideo::test_writes_mp4_ui_output` still passes — `FUSION_H3_SR` unset = "0" = unchanged)

Also run the existing h3 suite to confirm no regression:
Run: `pytest tests/test_h3_nodes.py -v`
Expected: all PASS

- [ ] **Step 5: Lint**

Run: `ruff check fusion_comfyui_plugin/nodes/sr.py fusion_comfyui_plugin/nodes/h3.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add fusion_comfyui_plugin/nodes/h3.py tests/test_h3_savevideo_sr.py
git commit -m "feat(sr): wire env-triggered SR into H3 SaveVideo (mock+real)"
```

---

## Task 4: AICF chain verification (mock mode, live ComfyUI)

**Files:**
- No new source files. Run live ComfyUI:11445 + AICF h3-i2v workflow with `FUSION_H3_SR=mock`.
- Test: manual e2e + measurement (document outcome in PR).

**Goal:** Verify the env trigger fires inside the real AICF→ComfyUI chain, the 4x mp4 is produced and served from `/history`, and AICF downloads it — with zero AICF source/JSON changes. Mock mode proves the wiring without needing PR-A's real SR.

- [ ] **Step 1: Start fusion-mlx + ComfyUI**

Run: `~/claude-home/fusion-mlx/start.sh start` then start ComfyUI on :11445 with `FUSION_H3_SR=mock FUSION_H3_SR_SCALE=4` in the environment. Verify `:11445` responds.
- [ ] **Step 2: Run AICF h3-i2v workflow**

Trigger AICF's h3-i2v job (the static JSON, untouched). Watch ComfyUI logs for `SaveVideo(H3): SR mode=mock scale=4` + the 4x output dims. Confirm no AICF error.
- [ ] **Step 3: Verify AICF download + dims**

Confirm AICF downloads the mp4 from `/history` and the file is 4x resolution (e.g. 5376x3072 from 1344x768). Record file size.
- [ ] **Step 4: Cleanup**

Delete intermediate artifacts (logs excepted). Keep the final mp4 + the SR-mode log line as evidence.
- [ ] **Step 5: Document outcome**

Record pass/fail + dims + timing in the PR description. This task gates the "AICF chain zero-change" success criterion.

---

## Task 5: Real-model e2e + sharpness measurement (after PR-A release)

**Files:**
- No new source files. Needs PR-A merged + fusion-mlx dep bumped + SR model downloaded (hf-mirror).
- Test: manual e2e + lap_var measurement.

**Goal:** With `FUSION_H3_SR=1` (real fusion-mlx RealESRGAN), produce the SR'd H3 mp4 and measure sharpness gain. Target: SR-after lap_var approaches the keyframe baseline (213), not just 2x of 47.

- [ ] **Step 1: Dep bump + model**

Bump `fusion-mlx>=<PR-A version>` in fusion-comfyui. Download `RealESRGAN_x4plus.pth` via `https://hf-mirror.com`, convert to safetensors (PR-A Task 3 Step 0), place at `~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.safetensors`. Restart fusion-mlx + ComfyUI with `FUSION_H3_SR=1 FUSION_H3_SR_SCALE=4`.
- [ ] **Step 2: Run + measure**

Run the H3 i2v pipeline (same keyframe/seed/768p/dur2 baseline as the s12 reference, `/tmp/h3_captured.mp4`). Measure 3-frame (first/mid/last) lap_var of the SR mp4 vs the s12 baseline (46.98). Log per-frame SR elapsed + total.
- [ ] **Step 3: Time-budget check**

If 4x total SR time + H3 generate > AICF 30min deadline, set `FUSION_H3_SR_SCALE=2` and re-measure. Document the shippable scale explicitly (never a silent downgrade — log it). 2x is still a major gain over the VAE low-pass.
- [ ] **Step 4: Verify + cleanup**

Confirm SR mp4 is measurably sharper (lap_var SR-after > lap_var before, ideally toward 213). Delete intermediates; keep final mp4 + measurement log. Record numbers in PR.
- [ ] **Step 5: Notify user**

Per the standing constraint ("完整的端到端验证后再来通知我"), only notify once the full AICF e2e with real SR is verified — sharper mp4 produced, AICF downloads it, no errors, time within budget.

---

## Task 6: README/docs update

Per project rule (update README when code changes need it).

- [ ] **Step 1: Document the SR env vars** in `README.md` (English, env-var reference table) and `README_CN.md` (Chinese): `FUSION_H3_SR`, `FUSION_H3_SR_SCALE`, `FUSION_H3_SR_ENDPOINT`, the model download (hf-mirror) + safetensors conversion step, and the PR-A endpoint dependency.
- [ ] **Step 2: Commit**

```bash
git add README.md README_CN.md
git commit -m "docs(sr): document H3 SR env vars + model setup"
```
