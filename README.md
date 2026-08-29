# fusion-comfyui

**English** | [中文](README_CN.md)

Pure MLX ComfyUI server on Apple Silicon. Zero PyTorch at runtime.

Built on [fusion-mlx](https://github.com/dahai80/fusion-mlx) engine — FLUX.2, Wan2.2, SkyReels-V3, LTX-2 image/video generation via Metal/MLX.

## Quick Start

```bash
# Install
pip install -e .

# Run standalone server (pure MLX, no PyTorch)
fusion-comfyui serve --port 11443

# Or run as ComfyUI custom nodes (Phase 1, requires PyTorch host)
cd ComfyUI && python main.py
```

Open `http://localhost:11443` — the ComfyUI frontend connects directly.

## Architecture

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | ComfyUI custom nodes (PyTorch host + MLX compute) |
| Phase 2 | ✅ Complete | Standalone FastAPI server, ComfyUI protocol, zero PyTorch |
| Phase 3 | 🔧 In Progress | Spec denoise machinery ✅ (landed, default-off, accel falsified), Radix cache ❌ (FALSIFIED — T5 embeddings not prefix-reusable; compile cache covers latency), stats node ✅, NVFP4 blocked (mlx#2962), async dispatch needs fusion-mlx |
| Phase 4 | ✅ Done | Swift app ✅, first-run setup assistant ✅, DMG+icon ✅, exit criteria verified (launch 0.15s, hf-mirror pull, no-terminal e2e T2V) |

See [CONSTRUCTION_PLAN.md](CONSTRUCTION_PLAN.md) for full details.

## Native Node Bridge Overrides (Phase 1)

The `ComfyUI/custom_nodes/ComfyUI-Fusion-MLX/` plugin overrides selected native ComfyUI nodes so ZHO/example workflows route to the fusion-mlx engine instead of PyTorch. Override classes are registered in `__init__.py` (`NODE_CLASS_MAPPINGS` + `_native_overrides`) and injected into ComfyUI's native `NODE_CLASS_MAPPINGS` at load time.

**Stable Cascade** (v0.2.6, issue #13): the 4 native `comfy_extras/nodes_stable_cascade.py` nodes (`StableCascade_EmptyLatentImage`, `StableCascade_StageC_VAEEncode`, `StableCascade_StageB_Conditioning`, `StableCascade_SuperResolutionControlnet`) are overridden with numpy latents (no `torch`). `CheckpointLoaderSimple`/`_fallback_model` route any `cascade`/`wuerstchen` checkpoint to the self-contained `models--stabilityai--stable-cascade-prior` pipeline (never a video model), and `FusionVAEWrapper` exposes `downscale_ratio` (4) + `encode()` so the native VAE-encode nodes no longer crash. The fusion-mlx `stable_cascade` variant runs end-to-end (prior→decoder→vqgan); txt2img Cascade workflows run fully, image-conditioned Cascade degrades to txt2img (tracked separately for an upstream engine change).

**KSampler cfg/sampler forwarding + Wan routing** (v0.2.8, issue #315): the `KSampler` node now forwards its `cfg` to the engine (`guide_scale` + `cfg_scale`, so wan2 and the other backends both honor it) and `sampler_name` to the engine `scheduler` (via `normalize_sampler`, mapping ComfyUI spellings like `uni_pc`→`unipc`). `shift` is intentionally not forwarded (model-tuned). `_map_unet_name_to_model_name` gained early branches so `t2v`+`14b` checkpoints route to `Wan2.1-T2V-14B` (t2v weights, `in_dim=16`) and `fun`+`camera` checkpoints route to `Wan2.1-Fun-Camera-1.3B` (`add_control_adapter`), each before the generic 14b/i2v branches that previously loaded the wrong weights or dropped the camera control adapter. `to_numpy` uses `np.asarray` (NumPy 2.x-safe against MLX `__array__`).

**Frontend init alignment + toolbar testids + ImageGen width-collapse guard** (v0.2.9, issues #5, #37): all API handlers are now mounted under the `/api` prefix (in addition to the legacy flat routes) so the bundled Vue frontend v1.45.20 initializes and renders its topbar instead of 404-ing out; `/jobs` pagination returns `has_more` (snake_case) to satisfy the frontend's zod schema; `/templates/*` and `/userdata/*` JSON fetches return `[]` instead of 404. `object_info` now includes the `output_name`/`name`/`display_name`/`output_node`/`deprecated` fields the frontend expects, and the `/ws` status message carries `sid` + `queue_remaining`. `index.html` injects a bootstrap that stamps 13 toolbar buttons (`Queue Prompt`, `Queue Front`, `View Queue`, `View History`, the gear ⚙️ `Settings`, `Save`, `Load`, `Refresh`, `Clear`, `Load Default`, `Reset View`, `Clipspace`, `Save (API Format)`) with stable `data-testid` + `aria-label` and dispatches a `comfy:ui-ready` CustomEvent (via `MutationObserver` for async Vue mount), fixing Playwright button_smoke timeouts (#5). `FusionImageGenNode.generate` now detects upstream raw width-collapse (degenerate `(H,3,3)` shape) via `_raw_width_collapsed` and retries with a fresh seed up to 3 times (#37); upstream root cause (cross-thread MLX lazy-eval race in fusion-mlx `image_gen`) tracked in fusion-mlx issue #575 / PR #576.

**drama rename + CI + build fixes** (v0.2.10): the `xiyouji` package was renamed to `drama` (package, node category, display names, `XIYOUJI_*` env vars → `DRAMA_*`, loggers, VLM prompt generalized away from 西游记-specific wording) so the GUI is not tied to one series. The first GitHub Actions CI workflow (`.github/workflows/ci.yml`, macOS-14 ruff lint + pytest, e2e auto-skip) was added with a `.gitignore` secret guard. Three fresh-install blockers were fixed: standard `setuptools.build_meta` build backend (was a broken `_legacy` alias), dropped PEP639-redundant license classifier, and a cascade routing test monkeypatch hole. Dependency floor raised to `fusion-mlx>=0.8.27` (upstream #575 cross-thread lazy-eval race fix).

**H3 drama video pipeline + t2va dark-output fix** (v0.2.11, fusion-mlx v0.8.34): the MiniMax-H3 drama pipeline is wired end-to-end — engine routing for H3 (t2va) in `_MODEL_TYPES`/`_LATENT_CHANNELS`, a `_fallback_model` H3 branch, scene-loop video generation (`generate_video`), a `DRAMA_VIDEO_MODEL` env knob (default image fallback), TTS+lipsync via `DRAMA_TTS_ENABLED`→`tts_engine.tts_synthesize`, and the upstream `quantize` knob threaded through `/v1/videos/generate` (fusion-mlx PR #587, issue #586). A `supports_i2v=False` guard blocks image/reference inputs that H3 cannot yet consume (fusion-mlx PR #590, issue #589 — full i2va/l2va/fl2va still open). The t2va dark-output bug was fixed in two upstream rounds: timestep assignment (fusion-mlx #602 → PR #603 → v0.8.33) then the DiT RoPE position grid (fusion-mlx #605 → PR #606 → v0.8.34) — the MLX `video_position_grid` used raw `arange(n)` (spatial range scaled with resolution) instead of the official aspect-normalized `_spatial_position_grid` with `sqrt_area=sqrt(h*w)` (patch-pre, constant [0,32) regardless of resolution) plus non-uniform temporal spacing and text-row `arange(n_text)`; real-model verify on 256×256 t2va raised mean YAVG 9.9 → 29.06. A test-flake (random-order `test_single_node` status=interrupted) was fixed at its root cause — `_interrupt_flag` module global leaked by two interrupt tests not consuming the one-shot flag — by consuming it (`check_interrupt() is True`) at the end of each. Dependency floor raised to `fusion-mlx>=0.8.34`. Env knobs (call-time read in `pipeline.py`): `DRAMA_VIDEO_MODEL`, `DRAMA_QUANTIZE` (default `dit8_te4`), `DRAMA_TTS_ENABLED` (default `0`), `DRAMA_VIDEO_FRAMES` (default 49), `DRAMA_VIDEO_FPS` (default 24).

**H3 native audio joint generation** (F1, fusion-mlx PR #611 — engine-level landed, pipeline wiring landed): the drama pipeline now wires H3's native t2va A/V joint generation (environment sound + score, generated alongside the video) through the `DRAMA_NATIVE_AUDIO` env knob (default `0`, call-time read in `pipeline.py`). When on, `generate_video` is called with `audio=True`, which the engine forwards to the H3 `generate_t2va_av` joint denoise (dual scheduler — video shift=12.0, audio shift=3.0, shared step count) + AudioVAE decode (DAC+BigVGAN, 32kHz mono) + ffmpeg mux, producing an MP4 with its own audio track. `SceneVideoAssembler` gained a `native_audio` input: when the video already carries a native track AND a F2 TTS dialogue wav is present, an ffmpeg `amix` filter mixes them (voice weight 1.0 over environment weight 0.5, `duration=longest` so the environment bed fills the scene) instead of one replacing the other; native track alone is preserved; the legacy single-TTS path is unchanged. FFT spectral verify on synthetic fixtures confirmed the mix sums both sources (300Hz environment + 898Hz voice both present in the output, vs legacy replace which keeps only the video's first audio input). Without the knob the pipeline behaves exactly as before (F2 TTS-only, no native audio).

**P2 — torch-free IMAGE/MASK I/O glue (fork to numpy)** (v0.2.13+): the ComfyUI `IMAGE`/`MASK` types are forked to `np.ndarray` float32 NHWC `[B,H,W,C]` / `[B,H,W]` in `[0,1]`, and the last `import torch` was removed from `fusion_comfyui/` + `fusion_comfyui_plugin/` non-test code (the plugin is now torch-free in its I/O path). Six pure image transforms (`ImageScale`, `ImageScaleBy`, `ImageBatch`, `EmptyImage`, `ImagePadForOutpaint`, `LoadImageMask`) were rewritten on a numpy/PIL scaling kernel (`_scaling.common_upscale`/`lanczos`/`bislerp`, handles 4D NCHW + 5D video), overriding the native torch nodes via the existing `_native_overrides` monkey-patch. Seven dead-path nodes with no MLX route (`ConditioningSetMask`, `VAEEncodeForInpaint`, `InpaintModelConditioning`, `ControlNetApply`, `ControlNetApplyAdvanced`, `PainterNode`, `QwenImageDiffsynthControlnet`) are stubbed via `_stub_factory` — each subclasses the native (inherits `INPUT_TYPES`) but raises `NotImplementedError` from its `stub_run`, so the node shows in the UI but fails loudly if used. `LoadImage.load_image` returns numpy IMAGE/MASK via `to_image_tensor`/`to_mask_numpy` (bridge). `LatentUpscale`'s true-latent path uses the numpy `common_upscale` (was `comfy.utils`); `ip_adapter` is safetensors-only (`.bin`/`.pt`/`.ckpt` → log + None, the torch `_load_torch_ip_adapter` deleted). Exit criteria: `grep "import torch"` 0 in both packages (non-test); 449 plugin tests pass + 3 skipped; ruff clean; ComfyUI startup 96 mappings, 13/13 P2 overrides registered; scaling `lanczos` parity vs torch reference corr = 1.0.

**P3 — Transparent Staged Default** (fork-comfyui phase 3): the native ComfyUI `KSampler` now routes video-T2V and image-txt2img through the fusion-mlx **staged API** (text-encode → denoise → vae-decode) with strict model offload between stages — memory sawtooth, peak = the largest single stage, not the sum. A `_should_use_staged(model_wrapper, positive, negative, latent_image, denoise) -> bool` predicate auto-detects from 4 wired latent/conditioning keys which path to take; `_generate_staged` is a new async sibling to `_generate_monolithic`, and `KSampler.sample` gains a 3-line dispatch (`_generate_staged` vs `_generate_monolithic`). `_generate_monolithic` stays byte-for-byte unchanged. Auto-fallback to monolith: video I2V (`_i2v_image_path`), VACE (`_vace_control_video`/`_mask`/`_ref`), image cascade stage_b (`stable_cascade_prior` pass-through), image img2img (`_image_init_path` + `denoise < 1.0`). The orchestration helper `FusionEngineWrapper._run_staged_pipeline` chains the 10 stage methods with `FusionMemoryGuardian.purge_memory()` between each (load_text_encoder → encode pos + neg (when `cfg > 1.0`) → unload → load_dit → denoise → unload → load_vae → decode → unload); `neg_cond=None` when `cfg <= 1.0`. `_staged_pixels_to_numpy` normalizes the staged `decode` mx.array float [0,1] output (NO `/255.0`, unlike the monolith uint8 path) to numpy — video `[T,H,W,3]`, image NCHW→NHWC squeeze `[H,W,3]` (4ch→3 slice), defensive clamp. Explicit `FusionKSampler`/`TextEncoder`/`VAEDecoder` nodes remain for cacheable explicit-stage graphs (Decision 5, coexist). Verified end-to-end on real Wan2.1-T2V-1.3B (17 frames, 832×480, 8 steps, 88.99s): 6-stage sawtooth confirmed (text_encoder/dit/vae load+unload, purge between each), output `(1,17,480,832,3)` std 0.0812 no NaN. Exit criteria: 475 plugin tests pass + 3 skipped; ruff clean; 0 `import torch` non-test; real-model e2e sawtooth confirmed. I2V/VACE/camera staged gap filed upstream (fusion-mlx issue #652). Plan: `docs/superpowers/plans/2026-08-26-p3-transparent-staged-default.md`; spec: `docs/superpowers/specs/2026-08-26-p3-transparent-staged-default-design.md`.

**Reliability fixes — IP-Adapter cross-thread, lifecycle cache, drama scene continuity** (v0.2.12): three production-hardening fixes landed together. (1) IP-Adapter weights loaded on the ComfyUI main thread were lazy `mx.arrays`; consumed inside the patched transformer `__call__` on the image-executor worker thread, cross-stream lazy access raised `There is no Stream(gpu, 0) in current thread` at `to_k_ip`/`to_v_ip` time. Fix: `_materialize_weights()` evals all params (SigLIP + proj + attn processors) into concrete stream-independent GPU buffers right after `from_pretrained()` — `nn.Module.parameters()` returns a nested DICT pytree (not a flat list), so each dict is passed directly to `mx.eval` which walks it recursively. (2) `purge_memory()` called the deprecated `mx.metal.clear_cache()` which corrupts live Metal command buffers mid-generation, aborting the next `mx.eval` with `Invalid Resource (kIOGPUCommandBufferCallbackErrorInvalidResource)` — switched to the non-deprecated `mx.clear_cache()` with a getattr fallback; regression guard asserts the deprecated path is never called. (3) Drama scene continuity: each scene's final frame is extracted (`_extract_last_frame`, ffprobe frame-count + `ffmpeg select=eq(n,N-1)`) and forwarded as the next scene's first-frame `image=` kwarg, wiring H3 fl2va keyframe conditioning (upstream fusion-mlx PR #616) for continuous narrative video — the chain is `pipeline.py image=prev_last_frame` → `engine_wrapper.generate_video` → `engines/video.py VideoGenParams.image` → H3 `generate_video(image=...)` → `generate_fl2va_video(condition_image_paths=[image], keyframe_anchors=("first",))`; `DEFAULT_QUANTIZE="dit8_te4"` is forwarded to avoid 33B jetsam OOM; continuity temp frames are cleaned in a new Phase 3.5. Regression guards: `test_set_image_embeds_materializes_to_concrete_mx_array`, `test_purge_uses_non_deprecated_clear_cache`, `TestExtractLastFrame`, `TestSceneContinuity`. Plugin unit tests 48 pass, drama tests 11 pass, e2e verified on clean server (IP-Adapter + Flux + scene-continuity chain).

**H3 sampling-pipe nodes for AICF** (v0.2.13, B2, PR #66): 10 MiniMax-H3 alias nodes under `Fusion-MLX/H3` so AIComicWorkstation (AICF) style "script→video" workflows route through the fusion-comfyui sampler pipe instead of the HTTP video route. The nodes stage H3 conditioning into the LATENT dict as `_h3_*` keys, which the sampler path (`SamplerCustomAdvanced` → `_generate_monolithic` → `engine.generate`) forwards to the engine — a new H3 block in `samplers.py` threads `_h3_quantize`/`_h3_audio`/`_h3_first_frame_path`/`_h3_last_frame_path`/`_h3_ref_images` into the `gen_kwargs`. Node list: `MiniMaxH3SigmaShift` (model passthrough, MLX uses its own shift defaults), `EmptyMiniMaxH3LatentAV` (latent `(B,24,(L-1)//4+1,H//16,W//16)` — z=24, spatial /16, temporal /4 causal, audio default on), `MiniMaxH3ImageToVideo` (i2v, `first_frame`/`last_frame` → `/tmp` png, audio forced off since fl2va image path is video-only), `MiniMaxH3ReferenceToVideo` (r2v, `ref_images` → `/tmp` png list, handles both dict-batch and plain-batch), `VAEDecodeAudio` (silent dummy — audio is muxed into the MLX mp4 already), `CreateVideo` + `SaveVideo` (override the core new-style video nodes so `/history` serves the file AICF downloads), and 3 aux nodes (`ImageScaleToTotalPixels`, `PrimitiveFloat`, `ComfyMathExpression` — sandboxed `eval`, `__builtins__` stripped). Temp pngs are written under `/tmp` explicitly (not `$TMPDIR`) because the H3 backend `_ALLOWED_READ_DIRS` blocks macOS `$TMPDIR`. 6 nodes collide with core (`CreateVideo`/`SaveVideo`/`VAEDecodeAudio`/`ImageScaleToTotalPixels`/`PrimitiveFloat`/`ComfyMathExpression`) and override via `_native_overrides`; total node registry grew to 106 mappings. Spec: `docs/superpowers/specs/2026-08-28-h3-sampling-pipe-adaptation-design.md`. Upstream gaps filed on fusion-mlx (English, issue→PR→dep bump flow): #687 (`VideoGenEngine.generate` omits `last_frame_image` from `VideoGenParams` — i2v-with-last-frame degrades to video-only on the engine path), #688 (no `reference_images` param / no ref2va branch in `generate_video` — h3-r2v e2e BLOCKED until PR), #689 (`ImageGenEngine.VARIANT_MAP` has no `qwen_image`/`qwen_image_edit` — AICF image-edit blocked, needs vendored MLX port). Real-33B e2e (model id `FL2VA`, `minimax_h3_fl2va_pruned_nvfp4`, `quantize="dit8_te4"` to avoid jetsam OOM): t2v 59.04s PASS, i2v first_frame 74.20s PASS, r2v staging PASS (drop documented, flip to success on #688 merge). 16 TDD node tests + 3 real-model e2e tests (`tests/test_h3_e2e.py`, skip-guarded `RUN_E2E=1 + Metal GPU + FL2VA present` so CI is safe).

## Project Structure

```
fusion_comfyui/          # unified core (P1: single source of truth)
├── core/                # the ONE core — imported by plugin via pip install -e
│   ├── config.py        # Phase 3 config + RadixCache (radix tree)
│   ├── engine_wrapper.py # fusion-mlx in-process wrapper (API-fresh, public_api imports)
│   ├── lifecycle.py     # Memory guardian + pipeline stage context
│   ├── bridge.py        # torch↔mlx array bridge (numpy-mediated, P2 target)
│   ├── wrappers.py      # FusionModelWrapper + model-path resolution
│   ├── async_utils.py   # cross-thread materialization helpers
│   ├── timer.py         # NodeTimer (drama pipeline instrumentation)
│   └── output_store.py  # File-based output store for /view endpoint
└── nodes/
    ├── base.py          # BaseNode abstract class (drama nodes subclass this)
    └── drama/           # drama pipeline nodes (tts, lipsync, pulid, assemble, vlm)

ComfyUI/custom_nodes/ComfyUI-Fusion-MLX → fusion_comfyui_plugin/  # symlinked custom node
├── __init__.py          # NODE_CLASS_MAPPINGS + native overrides (no sys.path hack)
├── nodes/               # 15 node modules — all import fusion_comfyui.core.X
└── tests/               # 415 unit tests (GPU-free, mocked)

FusionComfyUI/           # Phase 4 macOS native app (Swift)
├── FusionComfyUIApp.swift  # App entry + ContentView
├── ServerManager.swift     # Launch/monitor Python server
├── WebView.swift           # WebKit ComfyUI frontend
├── ModelManager.swift      # Model discovery + download
└── Package.swift
```

> **P1 unification (v0.2.13):** the dormant standalone FastAPI server (`fusion_comfyui/server/`, `dag/`, `main.py`, `nodes/registry.py`, `frontend/`) was removed — ComfyUI's own orchestration is the sole runtime. The plugin's private `core/` was folded into the single `fusion_comfyui/core/`; the plugin now imports it as an installed package (`pip install -e .`), no `sys.path` hack. The `fusion-comfyui` console script and the `fastapi`/`uvicorn`/`websockets` deps were dropped (server-only).


## Dependencies & Netlayer

fusion-comfyui depends on [fusion-mlx](https://github.com/dahai80/fusion-mlx) as
its MLX inference engine. Unlike other fusion-* services that reach fusion-mlx
over HTTP (`localhost:11434`), **fusion-comfyui imports fusion-mlx in-process**:
`fusion_comfyui_plugin/__init__.py` installs the `fusion_mlx._torch_stub` shim,
and the engine/pipeline wrapper code imports fusion-mlx engine classes
(`ImageGenEngine`, `VideoGenEngine`, `TTSEngine`, `LipsyncPipelineMLX`,
`PuLIDPipeline`, etc.) from the stable `fusion_mlx.public_api` layer
(landed by fusion-mlx PR #620, closes #613) within the same Python process.
The four symbols previously deferred to internal module paths
(`EnginePool`, `list_available_models`, `MuseTalkPipeline`,
`VLMBatchedEngine`) were added to `public_api.__all__` by fusion-mlx PR #625
(closes upstream #624, released v0.8.36) and are now imported from
`fusion_mlx.public_api` too — all `TODO(upstream #624)` markers removed,
dependency floor raised to `fusion-mlx>=0.8.36`.

This is a deliberate, declared exception to the monorepo HTTP-routing convention
(documented in `architecture/netlayer-compliance-plan.md` §5.5):

- **Phase 1 (current)** — in-process import is declared here; no network hop,
  no `X-Fusion-Source` header. Behavior unchanged.
- **Phase 2 (future)** — evaluate adding an `X-Fusion-Source: comfyui` marker
  and permitting in-process calls explicitly on the fusion-mlx side.
- **Phase 3 (future)** — evaluate migrating to HTTP/gRPC via `fusion-gateway`,
  or keeping in-process but read-only.

Requires `fusion-mlx` to be importable in the same `.venv` (installed via the
shared monorepo virtualenv at `/Users/dahai/fusion/.venv`).

## Phase 3 Configuration

Phase 3 features are controlled via environment variables. Speculative denoising
machinery has landed in fusion-mlx (env-gated, default-off); radix cache is a
local implementation; NVFP4 is blocked upstream (see below).

```bash
# Speculative denoising (fusion-mlx machinery landed, default-off)
FUSION_SPECULATIVE_DENOISE=1
FUSION_SPEC_K=4
FUSION_SPEC_EPSILON=0.1
FUSION_SPEC_DRAFT_BLOCKS=
FUSION_SPEC_EVAL_STEPS=1

# Radix KV cache (local RadixCache implemented)
FUSION_RADIX_CACHE_ENABLED=1
FUSION_RADIX_CACHE_MAX_MB=512

# NVFP4 weight ingestion (blocked on MLX framework issue mlx#2962)
FUSION_NVFP4_ENABLED=1
FUSION_NVFP4_THRESHOLD_GB=8
```

Denoise stats are queryable at runtime: the `FusionDenoiseStats` node (in the
plugin `nodes/stats.py`) returns the last denoise run's acceptance/speedup
counters as JSON, and fusion-mlx exposes `GET /v1/videos/denoise-stats?model=<name>`.

## Reliability / Production Hardening

The runtime was acceptance-verified against production release standard.
Correctness bugs found by static + live probe and fixed (all in this repo, no
upstream changes needed):
- **Video model routing** (`core/engine_wrapper.py`) — `hunyuan`/`cosmos`/`svd`
  were absent from `_MODEL_TYPES`, so they fell back to `image` and loaded the
  wrong engine with a 4-channel latent. They now route to `video` with correct
  latent channels (hunyuan/cosmos 16, svd 4).
- **bridge.py label** — the `torch↔mlx` bridge was described as "zero-copy"; it is
  numpy-mediated (torch→numpy→mlx), so the README label was corrected.

> **Historical note:** the earlier v0.2.x standalone FastAPI server also carried
> fixes for DAG cycle/unresolved-link detection, a queue state machine, the
> WebSocket event protocol, a `/ws` 403-on-connect annotation bug, and a
> `/history` schema projection. That dormant server was removed in the P1
> unification (v0.2.13) — ComfyUI's own `server.py`/`execution.py` now own those
> concerns — so those fixes are no longer in this repo. The video-routing and
> bridge-label fixes above live in the unified `core/` and remain.

Covered by tests in `tests/test_engine_wrapper_routing.py`,
`tests/test_engine_wrapper_h3_routing.py`, and the plugin suite
`fusion_comfyui_plugin/tests/` (485 unit passing; e2e tests skip unless
`RUN_E2E=1` with a live ComfyUI server). A separate upstream frontend concern
(stable DOM `data-testid` selectors for UI testing) is tracked in issue
[Comfy-Org/ComfyUI#15392](https://github.com/Comfy-Org/ComfyUI/issues/15392).

## Upstream Dependencies

Phase 1 and Phase 2 upstream issues (all resolved):
- [#170](https://github.com/dahai80/fusion-mlx/issues/170) — ✅ Pipeline stage API (`load_text_encoder`, `encode_text`, `load_dit`, `denoise`, `load_vae`, `decode`)
- [#171](https://github.com/dahai80/fusion-mlx/issues/171) — ✅ Streaming progress callback (`StepCallback`)
- [#172](https://github.com/dahai80/fusion-mlx/issues/172) — ✅ Model registry API (`list_available_models`)

Phase 3 status (fusion-mlx machinery landed):
- Speculative denoising ✅ landed in fusion-mlx (`speculative_denoise.py`: draft-predict + batched-verify), env-gated and default-off. The layer-pruned draft was evaluated and FALSIFIED on SkyReels-V3 R2V 14B (0% acceptance at ε=0.1, no speedup) - machinery stays as infrastructure for a future distilled draft. Stats surface is live: `GET /v1/videos/denoise-stats?model=<name>` + `FusionDenoiseStats` node.
- Radix KV cache: ❌ FALSIFIED 2026-08-08. `RadixCache` (prefix-tree byte cache, 8 unit tests pass) was intended for prefix-shared T5 embeddings across short-drama shots, but T5 is bidirectional-attention so token i's embedding depends on ALL tokens — a shared text prefix does NOT yield a shared embedding prefix (prefix-tree collapses to exact-match). Measured on real Wan2.1-1.3B: 1st `encode_text`=3.00s, 2nd same-prompt=0.13s, 3rd DIFFERENT prompt=0.13s — the 22x speedup is MLX graph-compile cache, not prompt-key reuse; the "second-shot → 0ms" goal is already met by the compile cache. The class is kept as a correct data structure for a future genuinely-prefix-structured use (e.g. node-output dedup); the T5-embedding application is abandoned.
- NVFP4 weight reader: blocked on MLX framework issue [mlx#2962](https://github.com/ml-explore/mlx/issues/2962) (not fusion-mlx).
- Metal async dispatch pipeline: still needs fusion-mlx (split command buffer for CPU/GPU overlap).

HunyuanVideo MLX rewrite (all weight-matched, upstream [#15](https://github.com/dahai80/fusion-mlx/issues/15)):
VAE 248/248, DiT 856/856, TextEncoder CLIP-L 196/196 + Llama3-8B 290/290; real tokenizers added; e2e t2v verified.

## Phase 4: macOS Native App

`FusionComfyUI/` is a SwiftPM package: a SwiftUI shell with an embedded WebKit view that wraps the ComfyUI frontend, auto-starts the backend on `127.0.0.1:11445`, and offers model downloads via `fusion-mlx pull` (mirror-aware).

The app does **not** bundle a 3.5GB Python runtime. Instead, on first launch `SetupManager` bootstraps a dedicated venv at `~/.fusion-comfyui/venv`: it finds Homebrew Python ≥3.11, runs `python -m venv`, then `pip install -r ComfyUI/requirements.txt` + `pip install fusion-mlx` (all via the `hf-mirror.com` PyPI mirror). Setup happens once; subsequent launches skip straight to the server. The venv path is passed to `start.sh` via the `FUSION_VENV` env var. Override the venv root with `FUSION_COMFYUI_VENV_ROOT`. Set `FUSION_COMFYUI_START_SH` to point at a different `start.sh` if the app is relocated.

```bash
# Build the .app bundle + DMG (unsigned, runs locally)
cd FusionComfyUI && Scripts/build.sh all
open ".build/Fusion ComfyUI.app"          # or install the DMG

# Regenerate the app icon (.icns) from source
cd FusionComfyUI && Scripts/build.sh icon

# Or run directly from source
cd FusionComfyUI && swift run

# Server lifecycle (the app calls these; you can run them manually)
./start.sh start    # launches ComfyUI on :11445, waits for /system_stats
./start.sh status
./start.sh stop
./start.sh log -f
```

Components:
- `start.sh` — repo-root lifecycle manager (`start|stop|status|log|restart`); uses `FUSION_VENV` (default `/Users/dahai/fusion/.venv`), runs `python ComfyUI/main.py --port 11445 --listen 127.0.0.1`, pidfile + `wait_healthy`, sets `HF_MIRROR=https://hf-mirror.com`.
- `SetupManager.swift` — first-run dependency bootstrap: finds Homebrew Python ≥3.11, creates `~/.fusion-comfyui/venv`, pip-installs ComfyUI requirements + `fusion-mlx` via hf-mirror, streams progress; gates the main UI until ready.
- `ServerManager.swift` — launches `start.sh start` with `FUSION_VENV` in the child env, probes `GET /system_stats` until healthy, exposes `.stopped/.starting/.running/.failed` state.
- `ModelManager.swift` — lists models from `/object_info` + local `~/.fusion-mlx/models` cache; `Pull` button runs `fusion-mlx pull <repo>` with `HF_MIRROR=https://hf-mirror.com` and streams output.
- `WebView.swift` / `FusionComfyUIApp.swift` — WKWebView loads the ComfyUI frontend once the server is healthy; status dot + Models sheet.
- `Scripts/build.sh` — `all|app|package|dmg|icon|clean`; `dmg` builds a UDBZ DMG with a `/Applications` drag-to-install symlink; `icon` regenerates `AppIcon.icns` from `make_icon.py`.
- `Scripts/make_icon.py` — deterministically renders the Fusion app icon (PIL → iconset → `iconutil`).

Requires macOS 14+ (Sonoma), Apple Silicon (arm64). First-run setup needs Homebrew Python ≥3.11 (`brew install python@3.12`).

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1+)
- macOS 14+ (Sonoma)
- [fusion-mlx](https://github.com/dahai80/fusion-mlx) >= 0.4.8
