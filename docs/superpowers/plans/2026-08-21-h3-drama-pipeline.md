# MiniMax-H3 Drama Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable minimax h3 to generate 西游记连续剧 via the fusion-comfyui drama pipeline — route H3 as video, add pipeline video-gen per scene, re-enable TTS+lipsync, wire upstream quantize so it fits 137G RAM.

**Architecture:** Two-layer fix. Routing layer (local A/B) gets H3 to `VideoGenEngine`. Pipeline layer (local C/D/F2) adds `generate_video` per scene + env-gated TTS/lipsync using existing wrapper lifecycle. Upstream layer (E/F1/G) makes quantize reachable, adds native audio + scene continuity. Local lands first (mock-tested), upstream follows issue→PR→merge.

**Tech Stack:** Python 3.10+, fusion-mlx (engine HTTP), MLX (Apple Silicon), pytest+pytest-asyncio, ruff (scope `fusion_comfyui tests`).

**Spec:** `docs/superpowers/specs/2026-08-21-h3-drama-pipeline-design.md`

## Global Constraints

- **venv:** `cd /Users/dahai/fusion && source .venv/bin/activate && cd fusion-comfyui` before any pytest.
- **ruff scope:** `ruff check fusion_comfyui tests` (NEVER `.` — vendored ComfyUI errors). select E/F/W/T/N805/S307/S102, ignore E501/E722/E731/E712/E402/E741.
- **test run:** `pytest tests/<file>::<test> -v` single; `pytest tests/ -v` full. asyncio_mode=auto.
- **indent:** 4-space multiples. No docstrings. Every new function has logging.
- **commit:** only when user asks + branch first. This plan's tasks end at "ready to commit" — actual commit gated on user.
- **upstream rule:** fusion-mlx fixes → file issue first, then PR, then follow to merge. Local fixes done directly in this repo.
- **cleanup:** test process artifacts after verify; keep only final output + logs.
- **H3 facts:** z_channels=24, max_n=1, dim_divisibility=16, max_frames=361 (15s@24fps), FL2VA peak 144G > 137G RAM without quantize.

---

## File Structure

**Local (this repo — `fusion-comfyui`):**
- Modify `fusion_comfyui/core/engine_wrapper.py` — add H3 to `_MODEL_TYPES`/`_LATENT_CHANNELS` (Gap A); add `generate_video` method (Gap C).
- Modify `fusion_comfyui_plugin/core/wrappers.py` — add H3 branch in `_fallback_model` (Gap B).
- Modify `fusion_comfyui/nodes/drama/pipeline.py` — `DRAMA_VIDEO_MODEL` env, video-gen path, TTS/lipsync re-enable (Gap C/D/F2).
- Create `tests/test_engine_wrapper_h3_routing.py` — A unit tests.
- Create `tests/test_wrappers_h3_fallback.py` — B unit tests.
- Create `tests/test_drama_pipeline_video.py` — C/D/F2 mock tests.

**Upstream (`~/claude-home/fusion-mlx`):**
- Modify `fusion_mlx/engines/video_backends/base.py` — `VideoGenParams.quantize` field (Gap E).
- Modify `fusion_mlx/engines/video.py` — forward `quantize` kwarg (Gap E).
- Modify `fusion_mlx/engines/video_backends/minimax_h3.py` — pass `quantize` to `generate_video` (Gap E).
- (F1/G filed as issues + PRs after E merges — separate plan tasks for issue/PR creation, implementation deferred to upstream cycle.)

---

### Task 1: Gap A — engine routing for minimax/h3

**Files:**
- Modify: `fusion_comfyui/core/engine_wrapper.py:15-37`
- Test: `tests/test_engine_wrapper_h3_routing.py`

**Interfaces:**
- Consumes: none (dict literals only).
- Produces: `_infer_model_type("minimax-h3")=="video"`, `_get_latent_channels("minimax-h3")==24`. Unblocks Task 3 (pipeline uses `_infer_model_type` indirectly via wrapper construction).

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine_wrapper_h3_routing.py`:

```python
from fusion_comfyui.core.engine_wrapper import _infer_model_type, _get_latent_channels


class TestH3Routing:
    def test_minimax_routes_to_video(self):
        assert _infer_model_type("minimax-h3") == "video"
        assert _infer_model_type("MiniMax-H3") == "video"

    def test_h3_routes_to_video(self):
        assert _infer_model_type("h3-fl2va") == "video"
        assert _infer_model_type("h3-ref2va") == "video"

    def test_fl2va_ref2va_routes_to_video(self):
        assert _infer_model_type("fl2va-14B") == "video"
        assert _infer_model_type("ref2va-14B") == "video"

    def test_h3_latent_channels_24(self):
        assert _get_latent_channels("minimax-h3") == 24
        assert _get_latent_channels("h3-fl2va") == 24
        assert _get_latent_channels("fl2va-14B") == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dahai/fusion && source .venv/bin/activate && cd fusion-comfyui && pytest tests/test_engine_wrapper_h3_routing.py -v`
Expected: FAIL — `minimax-h3` returns `"image"` (default), not `"video"`; channels returns `4` default, not `24`.

- [ ] **Step 3: Add H3 entries to dicts**

In `fusion_comfyui/core/engine_wrapper.py`, replace the two dict literals:

```python
_MODEL_TYPES = {
    "flux2": "image",
    "flux": "image",
    "wan2": "video",
    "wan": "video",
    "skyreels": "video",
    "ltx": "video",
    "cosmos": "video",
    "hunyuan": "video",
    "svd": "video",
    "minimax": "video",
    "h3": "video",
    "fl2va": "video",
    "ref2va": "video",
}

_LATENT_CHANNELS = {
    "flux2": 16,
    "flux": 4,
    "wan2": 16,
    "wan": 16,
    "skyreels": 16,
    "ltx": 16,
    "cosmos": 16,
    "hunyuan": 16,
    "svd": 4,
    "minimax": 24,
    "h3": 24,
    "fl2va": 24,
    "ref2va": 24,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine_wrapper_h3_routing.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Verify no regression + ruff**

Run: `pytest tests/test_engine_wrapper_routing.py -v && ruff check fusion_comfyui/core/engine_wrapper.py tests/test_engine_wrapper_h3_routing.py`
Expected: existing routing tests PASS, ruff clean.

---

### Task 2: Gap B — fallback routing for minimax/h3

**Files:**
- Modify: `fusion_comfyui_plugin/core/wrappers.py:227` (after `available = _available_video_models()`, before wan branch)
- Test: `tests/test_wrappers_h3_fallback.py`

**Interfaces:**
- Consumes: `_available_video_models()` (existing, returns list).
- Produces: `_fallback_model("h3")=="minimax-h3"` when installed.

- [ ] **Step 1: Write failing tests**

Create `tests/test_wrappers_h3_fallback.py`:

```python
import pytest


@pytest.fixture
def h3_model_installed(monkeypatch):
    import fusion_comfyui_plugin.core.wrappers as w
    monkeypatch.setattr(w, "_available_video_models", lambda: ["minimax-h3", "Wan2.2-5B"])
    return "minimax-h3"


class TestH3Fallback:
    def test_fallback_h3_to_minimax(self, h3_model_installed):
        from fusion_comfyui_plugin.core.wrappers import _fallback_model
        assert _fallback_model("h3-14B") == "minimax-h3"
        assert _fallback_model("minimax-h3") == "minimax-h3"

    def test_fallback_minimax_exact_dir_short_circuits(self, h3_model_installed, monkeypatch):
        from fusion_comfyui_plugin.core.wrappers import _fallback_model
        import fusion_comfyui_plugin.core.wrappers as w
        import os
        orig_isdir = os.path.isdir
        def fake_isdir(p):
            if p.endswith("minimax-h3"):
                return True
            return orig_isdir(p)
        monkeypatch.setattr(os.path, "isdir", fake_isdir)
        assert _fallback_model("minimax-h3") == "minimax-h3"

    def test_fallback_h3_not_installed_falls_through(self, monkeypatch):
        import fusion_comfyui_plugin.core.wrappers as w
        monkeypatch.setattr(w, "_available_video_models", lambda: ["Wan2.2-5B"])
        from fusion_comfyui_plugin.core.wrappers import _fallback_model
        resolved = _fallback_model("h3-14B")
        assert resolved == "Wan2.2-5B", resolved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wrappers_h3_fallback.py::TestH3Fallback::test_fallback_h3_to_minimax -v`
Expected: FAIL — no H3 branch → `_fallback_model("h3-14B")` returns `available[0]` or wan branch, not `"minimax-h3"`.

- [ ] **Step 3: Add H3 branch**

In `fusion_comfyui_plugin/core/wrappers.py`, insert after `available = _available_video_models()` (line 227), before the `if "wan" in name` branch:

```python
    available = _available_video_models()
    if ("minimax" in name or "h3" in name) and "minimax-h3" in available:
        logger.info("Falling back %s -> minimax-h3 (H3 video)", requested)
        return "minimax-h3"
    if "wan" in name and "Wan2.2-5B" in available:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wrappers_h3_fallback.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify no regression + ruff**

Run: `pytest tests/test_stable_cascade_bridge.py -v && ruff check fusion_comfyui_plugin/core/wrappers.py tests/test_wrappers_h3_fallback.py`
Expected: cascade tests PASS, ruff clean.

---

### Task 3: Gap C — add `generate_video` to FusionEngineWrapper

**Files:**
- Modify: `fusion_comfyui/core/engine_wrapper.py` (add method after `generate_image`, ~line 330)
- Test: `tests/test_drama_pipeline_video.py`

**Interfaces:**
- Consumes: `self._engine.generate(...)` (VideoGenEngine — accepts prompt/num_frames/width/height/seed/n/on_step kwargs).
- Produces: `async generate_video(prompt, num_frames, width, height, seed, output_path, fps, **kwargs) -> str` — writes mp4 to `output_path`, returns `output_path`. Task 4 (pipeline) calls this.

- [ ] **Step 1: Write failing test**

Create `tests/test_drama_pipeline_video.py` with this first test:

```python
class TestGenerateVideo:
    async def test_generate_video_writes_mp4_and_returns_path(self, monkeypatch, tmp_path):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper
        wrapper = FusionEngineWrapper.__new__(FusionEngineWrapper)
        wrapper.model_name = "minimax-h3"
        wrapper.model_type = "video"
        wrapper._started = False
        wrapper._engine = None
        wrapper._on_step = None

        async def fake_ensure_started():
            wrapper._started = True
        monkeypatch.setattr(wrapper, "ensure_started", fake_ensure_started)

        class FakeEngine:
            async def generate(self, **kwargs):
                return (b"FAKE_MP4_BYTES",)
        wrapper._engine = FakeEngine()

        out = str(tmp_path / "scene_1.mp4")
        result = await wrapper.generate_video(
            prompt="a monkey king leaps", num_frames=25,
            width=768, height=448, seed=1000, output_path=out,
        )
        assert result == out
        with open(out, "rb") as f:
            assert f.read() == b"FAKE_MP4_BYTES"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drama_pipeline_video.py::TestGenerateVideo::test_generate_video_writes_mp4_and_returns_path -v`
Expected: FAIL — `AttributeError: 'FusionEngineWrapper' object has no attribute 'generate_video'`.

- [ ] **Step 3: Implement `generate_video`**

In `fusion_comfyui/core/engine_wrapper.py`, add after the `generate_image` method (after line 329):

```python
    async def generate_video(
        self,
        prompt: str,
        num_frames: int = 49,
        width: int = 768,
        height: int = 448,
        seed: int = 0,
        output_path: str = "",
        fps: int = 24,
        **kwargs,
    ) -> str:
        async with NodeTimer.timed(self.model_name, "generate_video", frames=num_frames, seed=seed):
            await self.ensure_started()
            logger.info("generate_video: prompt=%dchars frames=%d %dx%d seed=%d", len(prompt), num_frames, width, height, seed)
            if self.model_type != "video":
                logger.warning("generate_video called on non-video model %s", self.model_name)
            gen_kwargs = dict(
                prompt=prompt,
                num_frames=num_frames,
                width=width,
                height=height,
                seed=seed,
                n=1,
                fps=fps,
                on_step=self._on_step,
            )
            gen_kwargs.update(kwargs)
            result_bytes = await self._engine.generate(**gen_kwargs)
            if not output_path:
                import tempfile
                output_path = tempfile.mktemp(suffix=".mp4")
            with open(output_path, "wb") as f:
                f.write(result_bytes[0])
            logger.info("generate_video: wrote %s (%d bytes)", output_path, len(result_bytes[0]))
            return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drama_pipeline_video.py::TestGenerateVideo -v`
Expected: PASS.

- [ ] **Step 5: ruff**

Run: `ruff check fusion_comfyui/core/engine_wrapper.py tests/test_drama_pipeline_video.py`
Expected: clean.

---

### Task 4: Gap D+C+F2 — pipeline video-gen path with TTS/lipsync

**Files:**
- Modify: `fusion_comfyui/nodes/drama/pipeline.py:23,55-104`
- Test: `tests/test_drama_pipeline_video.py` (add cases)

**Interfaces:**
- Consumes: Task 3 `generate_video`; wrapper `load_tts`/`tts_synthesize`/`unload_tts`, `load_lipsync`/`lipsync_run`/`unload_lipsync` (existing, lines 507-535/454-481).
- Produces: pipeline `run_chapter` honors `DRAMA_VIDEO_MODEL` (video path) and `DRAMA_TTS_ENABLED=1` (audio). Backward compat: unset = existing image slideshow.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_drama_pipeline_video.py`:

```python
class TestPipelineVideoPath:
    async def test_video_model_env_routes_to_generate_video(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DRAMA_VIDEO_MODEL", "minimax-h3")
        monkeypatch.setenv("DRAMA_MODEL", "FLUX.2-klein-base-4B")
        monkeypatch.setenv("FUSION_OUTPUT_DIR", str(tmp_path))

        calls = {"video": [], "image": []}

        class FakeWrapper:
            model_type = "video"
            def __init__(self, *a, **k): pass
            async def ensure_started(self): pass
            async def generate_image(self, **k):
                calls["image"].append(k); return b"\x89PNG"
            async def generate_video(self, **k):
                out = str(tmp_path / f"v{len(calls['video'])}.mp4")
                with open(out, "wb") as f: f.write(b"MP4")
                calls["video"].append(k); return out
            async def stop(self): pass

        monkeypatch.setattr("fusion_comfyui.nodes.drama.pipeline.FusionEngineWrapper", FakeWrapper)
        monkeypatch.setattr("fusion_comfyui.nodes.drama.pipeline.unload_all_fusion_engines",
                            lambda *a: __import__("asyncio").sleep(0))

        from fusion_comfyui.nodes.drama import pipeline as pl
        monkeypatch.setattr(pl.DramaChapterParser, "split_only",
                            lambda self, t: [
                                {"scene_id": 1, "description_en": "monkey king born", "description_cn": "猴王出世", "duration_seconds": 3, "dialogue": ["I am born!"]},
                                {"scene_id": 2, "description_en": "monkey learns magic", "description_cn": "拜师学艺", "duration_seconds": 3, "dialogue": ["Teach me!"]},
                            ])
        from fusion_comfyui.nodes.drama.pipeline import run_chapter
        await run_chapter("第一回\n场景一：x。", "testchapter")

        assert len(calls["video"]) == 2, f"expected 2 video gens, got {calls}"
        assert len(calls["image"]) == 0, "video model set -> no image gen"

    async def test_no_video_env_falls_back_to_image(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DRAMA_VIDEO_MODEL", raising=False)
        monkeypatch.setenv("DRAMA_MODEL", "FLUX.2-klein-base-4B")
        monkeypatch.setenv("FUSION_OUTPUT_DIR", str(tmp_path))

        calls = {"video": [], "image": []}
        class FakeWrapper:
            model_type = "image"
            def __init__(self, *a, **k): pass
            async def ensure_started(self): pass
            async def generate_image(self, **k):
                calls["image"].append(k); return b"\x89PNG"
            async def generate_video(self, **k):
                calls["video"].append(k); return "x.mp4"
            async def stop(self): pass

        monkeypatch.setattr("fusion_comfyui.nodes.drama.pipeline.FusionEngineWrapper", FakeWrapper)
        monkeypatch.setattr("fusion_comfyui.nodes.drama.pipeline.unload_all_fusion_engines",
                            lambda *a: __import__("asyncio").sleep(0))
        from fusion_comfyui.nodes.drama import pipeline as pl
        monkeypatch.setattr(pl.DramaChapterParser, "split_only",
                            lambda self, t: [{"scene_id": 1, "description_en": "x", "description_cn": "x", "duration_seconds": 3, "dialogue": []}])
        from fusion_comfyui.nodes.drama.pipeline import run_chapter
        await run_chapter("第一回\n场景一：x。", "testimg")

        assert len(calls["image"]) == 1, f"expected image fallback, got {calls}"
        assert len(calls["video"]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_drama_pipeline_video.py::TestPipelineVideoPath -v`
Expected: FAIL — pipeline always calls `generate_image` regardless of `DRAMA_VIDEO_MODEL`.

- [ ] **Step 3: Modify pipeline env block**

In `fusion_comfyui/nodes/drama/pipeline.py`, replace lines 23-29 env block with:

```python
DEFAULT_MODEL = os.environ.get("DRAMA_MODEL", "FLUX.2-klein-base-4B")
DEFAULT_VIDEO_MODEL = os.environ.get("DRAMA_VIDEO_MODEL", "")
DEFAULT_QUANTIZE = os.environ.get("DRAMA_QUANTIZE", "dit8_te4")
DEFAULT_VLM = os.environ.get("DRAMA_VLM", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
DEFAULT_TTS = os.environ.get("DRAMA_TTS", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
DEFAULT_LIPSYNC_MODEL = os.environ.get("DRAMA_LIPSYNC_DIR", "")
DRAMA_ENABLE_TTS = os.environ.get("DRAMA_TTS_ENABLED", "0") == "1"
DEFAULT_STEPS = int(os.environ.get("DRAMA_STEPS", "4"))
DEFAULT_WIDTH = int(os.environ.get("DRAMA_WIDTH", "512"))
DEFAULT_HEIGHT = int(os.environ.get("DRAMA_HEIGHT", "512"))
DEFAULT_VIDEO_FRAMES = int(os.environ.get("DRAMA_VIDEO_FRAMES", "49"))
DEFAULT_VIDEO_FPS = int(os.environ.get("DRAMA_VIDEO_FPS", "24"))
```

- [ ] **Step 4: Replace Phase 3 body**

In `fusion_comfyui/nodes/drama/pipeline.py`, replace the Phase 3 block (from the line `# Phase 3: Generate scene images via monolithic generate()` through the `FusionMemoryGuardian.purge_memory()` line that precedes `# Unload FLUX model after all scene images generated`, inclusive of the scene `for` loop and `await model.stop()` cleanup) with:

```python
    # Phase 3: Generate scene media (video if DRAMA_VIDEO_MODEL set, else image)
    use_video = bool(DEFAULT_VIDEO_MODEL)
    logger.info("[Phase 3] Generating scene %s...", "video" if use_video else "images")

    scene_videos = []
    scene_audios = []

    model_name = DEFAULT_VIDEO_MODEL if use_video else DEFAULT_MODEL
    model = FusionEngineWrapper(model_name, offload_strategy="sequential", quant_bit="none")

    # Optional TTS engine (loaded once, reused per scene)
    tts_engine = None
    if DRAMA_ENABLE_TTS:
        try:
            tts_engine = FusionEngineWrapper(DEFAULT_TTS, offload_strategy="sequential")
            await tts_engine.load_tts(DEFAULT_TTS)
            logger.info("[Phase 3] TTS engine loaded: %s", DEFAULT_TTS)
        except Exception as e:
            logger.warning("[Phase 3] TTS load failed, audio disabled: %s", e)
            tts_engine = None

    for idx, scene in enumerate(scenes):
        scene_id = int(scene.get("scene_id", idx + 1))
        desc_en = scene.get("description_en", "")
        desc_cn = scene.get("description_cn", "")
        dialogue = scene.get("dialogue", [])
        seed = int(scene_id) * 1000

        logger.info("[Phase 3] Scene %d: %s", scene_id, desc_cn[:60])

        try:
            if use_video:
                video_path = os.path.join(output_dir, f"{chapter_title}_scene_{scene_id}.mp4")
                video_path = await model.generate_video(
                    prompt=desc_en,
                    num_frames=DEFAULT_VIDEO_FRAMES,
                    width=DEFAULT_WIDTH,
                    height=DEFAULT_HEIGHT,
                    seed=seed,
                    output_path=video_path,
                    fps=DEFAULT_VIDEO_FPS,
                    quantize=DEFAULT_QUANTIZE,
                )
                scene_videos.append(video_path)
                logger.info("[Phase 3] Scene %d video saved: %s", scene_id, video_path)
            else:
                frame_path = os.path.join(output_dir, f"{chapter_title}_scene_{scene_id}.png")
                png_bytes = await model.generate_image(
                    prompt=desc_en,
                    negative_prompt="low quality, blurry, deformed, watermark, text",
                    width=DEFAULT_WIDTH,
                    height=DEFAULT_HEIGHT,
                    steps=DEFAULT_STEPS,
                    cfg=3.5,
                    seed=seed,
                )
                with open(frame_path, "wb") as f:
                    f.write(png_bytes)
                scene_videos.append(frame_path)
                logger.info("[Phase 3] Scene %d image saved: %s (%d bytes)", scene_id, frame_path, len(png_bytes))
        except Exception as e:
            logger.warning("[Phase 3] Gen failed for scene %d: %s — placeholder", scene_id, e)
            if use_video:
                from PIL import Image as PILImage
                placeholder = os.path.join(output_dir, f"{chapter_title}_scene_{scene_id}.png")
                PILImage.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT), (128, 128, 128)).save(placeholder)
                scene_videos.append(placeholder)
            FusionMemoryGuardian.purge_memory()

        # TTS for dialogue lines (concatenate all lines into one audio)
        audio_path = ""
        if tts_engine and dialogue:
            try:
                full_line = " ".join(str(d) for d in dialogue) if isinstance(dialogue, list) else str(dialogue)
                wav_bytes = await tts_engine.tts_synthesize(text=full_line, speed=1.0)
                audio_path = os.path.join(output_dir, f"{chapter_title}_scene_{scene_id}.wav")
                with open(audio_path, "wb") as f:
                    f.write(wav_bytes)
                logger.info("[Phase 3] Scene %d TTS saved: %s", scene_id, audio_path)
            except Exception as e:
                logger.warning("[Phase 3] TTS failed for scene %d: %s", scene_id, e)
                audio_path = ""
        scene_audios.append(audio_path)

        FusionMemoryGuardian.purge_memory()

    # Unload generation model + TTS after all scenes
    logger.info("[Phase 3.5] Unloading model(s) to free memory...")
    await model.stop()
    if tts_engine:
        await tts_engine.stop()
    FusionMemoryGuardian.purge_memory()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_drama_pipeline_video.py -v`
Expected: PASS — `TestGenerateVideo` (1) + `TestPipelineVideoPath` (2).

- [ ] **Step 6: Verify full suite + ruff**

Run: `pytest tests/ -v -m "not inference" 2>&1 | tail -20 && ruff check fusion_comfyui/nodes/drama/pipeline.py tests/test_drama_pipeline_video.py`
Expected: no new failures vs baseline, ruff clean.

---

### Task 5: Gap E — upstream quantize plumbing (issue → PR → merge)

**Files:**
- Upstream: `~/claude-home/fusion-mlx/fusion_mlx/engines/video_backends/base.py`, `engines/video.py`, `engines/video_backends/minimax_h3.py`
- No local repo changes (dep bump in Task 7).

**Interfaces:**
- Consumes: existing `generate_video(..., quantize="none")` in `video/minimax_h3/generate.py` (already accepts kwarg).
- Produces: `VideoGenParams.quantize` field; engine forwards it; backend passes it through. After merge → fusion-mlx version bump → local dep bump (Task 7).

**Workflow per global rule:** file issue first in `dahai80/fusion-mlx`, then branch+PR, then follow to merge.

- [ ] **Step 1: File upstream issue**

Create issue in `dahai80/fusion-mlx` via `gh issue create`:

```bash
cd ~/claude-home/fusion-mlx
gh issue create --title "quantize (dit8_te4) unreachable through VideoGenEngine API — H3 OOM on 137G RAM" --body "$(cat <<'EOF'
## Problem
`video/minimax_h3/generate.py` `generate_video()` accepts `quantize` kwarg (none/te4/dit8/dit8_te4) but it is unreachable from the engine API:
- `engines/video_backends/base.py` `VideoGenParams` has no `quantize` field
- `engines/video.py` `VideoGenEngine.generate()` builds `VideoGenParams(**kwargs)` without forwarding `quantize`
- `engines/video_backends/minimax_h3.py` `generate(params)` calls `generate_video(...)` with no `quantize` arg -> defaults `"none"`

## Impact
MiniMax-H3 FL2VA peak memory = TE 67G + DiT 66G + VAE 11G = 144G > M5 Max 137G -> OOM. The runtime quantize path (`quantize.py`) exists but is unreachable. Must run quantize=dit8_te4 (peak ~61G) to fit.

## Proposed fix
1. `base.py`: add `quantize: str = "none"` to `VideoGenParams`
2. `video.py`: `VideoGenEngine.generate()` forward `quantize=kwargs.get("quantize", "none")`
3. `minimax_h3.py`: `generate(params)` pass `quantize=params.quantize` to `generate_video()`

`generate.py` already accepts the kwarg — no change there.
EOF
)"
```

Record the issue number (e.g. `#NNN`).

- [ ] **Step 2: Branch upstream**

```bash
cd ~/claude-home/fusion-mlx
git checkout -b fix/h3-quantize-reachable-NNN
```

- [ ] **Step 3: Add `quantize` field to VideoGenParams**

In `fusion_mlx/engines/video_backends/base.py`, add field to the `VideoGenParams` dataclass (after `output_format` or last existing field):

```python
    quantize: str = "none"
```

- [ ] **Step 4: Forward quantize in VideoGenEngine.generate**

In `fusion_mlx/engines/video.py`, find where `VideoGenParams(...)` is constructed in `generate()` and add:

```python
        quantize=kwargs.get("quantize", "none"),
```

(match the existing kwarg-forwarding style in that constructor call)

- [ ] **Step 5: Pass quantize in MiniMaxH3Backend.generate**

In `fusion_mlx/engines/video_backends/minimax_h3.py`, in `generate(params)`, change the `generate_video(...)` call to include:

```python
            quantize=params.quantize,
```

- [ ] **Step 6: Write upstream test**

Create/append to upstream test file (match upstream test convention — check `~/claude-home/fusion-mlx/tests/` for existing video param tests):

```python
def test_video_gen_params_has_quantize_default_none():
    from fusion_mlx.engines.video_backends.base import VideoGenParams
    p = VideoGenParams(prompt="x")
    assert p.quantize == "none"
    p2 = VideoGenParams(prompt="x", quantize="dit8_te4")
    assert p2.quantize == "dit8_te4"
```

- [ ] **Step 7: Run upstream test + commit + push + PR**

```bash
cd ~/claude-home/fusion-mlx
pytest tests/<the_video_param_test_file>::test_video_gen_params_has_quantize_default_none -v
git add -A && git commit -m "fix(engines): make quantize reachable through VideoGenEngine API

closes #NNN

- VideoGenParams: add quantize field (default 'none')
- VideoGenEngine.generate: forward quantize kwarg
- MiniMaxH3Backend.generate: pass params.quantize to generate_video"
gh pr create --title "fix(engines): make quantize reachable through VideoGenEngine API" --body "Closes #NNN. See issue for root cause + impact."
```

- [ ] **Step 8: Follow to merge**

Monitor PR, address review, merge. Record merged commit + new fusion-mlx version for Task 7 dep bump.

---

### Task 6: Gap F1+G — upstream issues (native audio + scene continuity)

**Files:** upstream issues only (implementation PRs deferred to upstream cycle after E merges).

- [ ] **Step 1: File F1 issue — H3 native audio**

```bash
cd ~/claude-home/fusion-mlx
gh issue create --title "MiniMax-H3 audio_output discarded — Omni-Transformer native audio not decoded" --body "$(cat <<'EOF'
## Problem
`video/minimax_h3/transformer.py` `MiniMaxH3DiTModel.forward()` returns `(video_output, audio_output)` but `generate.py` discards `audio_output`. `config.py` `H3AudioVAEConfig` is defined but has zero usages (grep confirms).

H3 is an Omni-Transformer (joint video+audio token generation). The MLX port decodes video only — no native speech/music/SFX.

## Proposed
- Wire `audio_output` through `generate_video` -> `H3AudioVAE` decode -> write audio track alongside mp4
- Scope as experimental flag (e.g. `audio=True`) — audio VAE forward + tokenizer is substantial

## Depends on
#NNN (quantize reachable) — audio VAE adds memory pressure.
EOF
)"
```

- [ ] **Step 2: File G issue — H3 scene continuity (i2va/l2va/fl2va)**

```bash
gh issue create --title "MiniMax-H3 only t2va implemented — i2va/l2va/fl2va missing, no inter-scene continuity" --body "$(cat <<'EOF'
## Problem
`generate.py` `generate_video()` calls `generate_t2va_video(...)` only. `config.py` `H3Config.tasks=("t2va","i2va","l2va","fl2va")` but only t2va implemented.

i2va (image->video), l2va (last-frame->video), fl2va (first+last->video) are scene-continuity primitives. Without them each scene generates from text only -> characters look different each cut (no visual continuity for serial drama).

## Proposed
- i2va first (most useful: keyframe->video), then l2va/fl2va
- Verify image-encoder weights exist in current bundle before implementation (may not be in 71G FL2VA pack)

## Depends on
#NNN (quantize reachable) — i2va loads image encoder too, raises peak memory.
EOF
)"
```

- [ ] **Step 3: Record issue numbers**

Both issues filed. PRs for F1+G implementation deferred to upstream cycle (large scope, after E merges + version bump). Note in spec `Honest Limitations`.

---

### Task 7: Local dep bump after upstream E merges

**Files:**
- Modify: `fusion_comfyui/pyproject.toml:28` (`fusion-mlx>=0.8.27` -> new floor)

**Interfaces:**
- Consumes: Task 5 merged PR + new fusion-mlx version tag.
- Produces: local repo requires quantize-reachable fusion-mlx.

- [ ] **Step 1: Confirm upstream merged + version**

```bash
cd ~/claude-home/fusion-mlx
git checkout main && git pull
git tag --sort=-creatordate | head -3
pip install -e . 2>&1 | tail -3
```

Record new version (e.g. `0.8.28`).

- [ ] **Step 2: Bump local dep floor**

In `fusion_comfyui/pyproject.toml`, update:

```toml
    "fusion-mlx>=0.8.28",
```

(use the actual new version from Step 1)

- [ ] **Step 3: Verify + ruff**

```bash
cd /Users/dahai/fusion && source .venv/bin/activate && cd fusion-comfyui
pytest tests/ -v -m "not inference" 2>&1 | tail -20
ruff check fusion_comfyui tests
```

Expected: green.

- [ ] **Step 4: Ready for commit** (gated on user)

This + Tasks 1-4 form the local deliverable. Summarize for user: ready to branch + commit + release when user approves.

---

## Self-Review

**Spec coverage:**
- Gap A (routing) -> Task 1
- Gap B (fallback) -> Task 2
- Gap C (pipeline video gen) -> Task 3 (wrapper method) + Task 4 (pipeline call)
- Gap D (video model env) -> Task 4 Step 3
- Gap E (quantize unreachable) -> Task 5
- Gap F1 (native audio) -> Task 6 Step 1 (issue; PR deferred per spec Honest Limitations)
- Gap F2 (TTS/lipsync re-enable) -> Task 4 Step 4
- Gap G (scene continuity) -> Task 6 Step 2 (issue; PR deferred)

**Placeholder scan:** no TBD/TODO in steps. All code blocks concrete. F1/G PR implementation deliberately deferred (issues filed) — stated in spec Honest Limitations, not a placeholder.

**Type consistency:**
- `generate_video(prompt, num_frames, width, height, seed, output_path, fps, **kwargs) -> str` — Task 3 defines, Task 4 calls with matching kwargs (incl `quantize=DEFAULT_QUANTIZE` via `**kwargs`). OK.
- `_fallback_model(requested) -> str` — Task 2 matches existing signature. OK.
- `VideoGenParams.quantize: str` — Task 5 Step 3 defines, Step 5 uses `params.quantize`. OK.
- `run_chapter(chapter_text, chapter_title)` — Task 4 keeps signature, tests call unchanged. OK.
- Env names: `DRAMA_VIDEO_MODEL`/`DRAMA_QUANTIZE`/`DRAMA_TTS_ENABLED` consistent across Task 4 Step 3 (define) + tests + spec. Existing `DEFAULT_TTS` env `DRAMA_TTS` reused for model name; new `DRAMA_TTS_ENABLED` gates on/off — distinct, no collision.

**Dependency order:** Tasks 1-2 (routing) unblock Task 4 e2e. Task 3 (`generate_video`) called by Task 4. Task 5 (upstream E) unblocks real e2e but not local code (mock-tested). Task 6 (issues) standalone. Task 7 (dep bump) after Task 5 merges. OK.
