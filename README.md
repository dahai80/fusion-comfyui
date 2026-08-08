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
| Phase 3 | 🔧 In Progress | Spec denoise machinery ✅ (landed, default-off, accel falsified), Radix cache ✅, stats node ✅, NVFP4 blocked (mlx#2962), async dispatch needs fusion-mlx |
| Phase 4 | 🔧 In Progress | Swift app split ✅, ServerManager/ModelManager ✅, needs packaging |

See [CONSTRUCTION_PLAN.md](CONSTRUCTION_PLAN.md) for full details.

## Project Structure

```
fusion_comfyui/          # Phase 2+ standalone server
├── core/
│   ├── config.py        # Phase 3 config + RadixCache (radix tree)
│   ├── engine_wrapper.py # fusion-mlx in-process wrapper
│   ├── lifecycle.py     # Memory guardian + pipeline stage context
│   └── output_store.py  # File-based output store for /view endpoint
├── dag/
│   ├── executor.py      # Topological sort + sequential execution
│   └── types.py         # NodeDef, LinkDef, Workflow, KNOWN_TYPES
├── nodes/
│   ├── base.py          # BaseNode abstract class
│   └── registry.py      # All pure MLX node implementations (Phase 2 + 3)
├── server/
│   ├── app.py           # FastAPI app + route mounting
│   ├── protocol.py      # ComfyUI protocol endpoints
│   ├── ws.py            # WebSocket progress streaming
│   └── static_files.py  # Output + frontend static file serving
└── main.py              # CLI entry point

ComfyUI/custom_nodes/ComfyUI-Fusion-MLX/  # Phase 1 custom nodes
├── core/
│   ├── bridge.py        # torch↔mlx array bridge (Phase 1 only; numpy-mediated, not zero-copy)
│   ├── lifecycle.py     # FusionMemoryGuardian + PipelineStageContext
│   └── engine_wrapper.py # fusion-mlx wrapper (torch-compatible)
├── nodes/
│   ├── loaders.py       # FusionModelLoaderNode
│   ├── conditioning.py  # FusionTextEncoderNode
│   ├── samplers.py      # FusionKSamplerNode
│   └── vae.py           # FusionVAEDecoderNode
└── __init__.py

FusionComfyUI/           # Phase 4 macOS native app (Swift)
├── FusionComfyUIApp.swift  # App entry + ContentView
├── ServerManager.swift     # Launch/monitor Python server
├── WebView.swift           # WebKit ComfyUI frontend
├── ModelManager.swift      # Model discovery + download
└── Package.swift
```

## Dependencies & Netlayer

fusion-comfyui depends on [fusion-mlx](https://github.com/dahai80/fusion-mlx) as
its MLX inference engine. Unlike other fusion-* services that reach fusion-mlx
over HTTP (`localhost:11434`), **fusion-comfyui imports fusion-mlx in-process**:
`fusion_comfyui_plugin/__init__.py` installs the `fusion_mlx._torch_stub` shim,
and `core/engine_wrapper.py` calls `fusion_mlx.engines` directly within the same
Python process.

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

Denoise stats are queryable at runtime: the `FusionDenoiseStats` node (both
Phase 1 custom nodes and Phase 2 registry) returns the last denoise run's
acceptance/speedup counters as JSON, and fusion-mlx exposes
`GET /v1/videos/denoise-stats?model=<name>`.

## Reliability / Production Hardening

The Phase 2 standalone server was acceptance-verified against production
release standard. Ten correctness bugs were found by static + live probe and
fixed (all in this repo, no upstream changes needed):

- **Frontend default resolution** (`static_files.py:get_frontend_dir`) — an empty
  `FUSION_FRONTEND_DIR` resolved to `Path(".")` (cwd) instead of the bundled
  frontend, so `GET /` returned 404. Now guarded; the bundled frontend is served
  by default. Override with `FUSION_FRONTEND_DIR=/abs/path`.
- **`/view` path traversal** (`static_files.py:view_file`) — `filename`/`subfolder`
  are now resolved and confined to the output dir; `../` escapes return 404.
- **DAG cycle detection** (`dag/types.py:topo_order`) — a cycle previously logged
  and silently skipped the nodes, returning an incomplete order. It now raises
  `ValueError`, surfaced as a workflow `error` status (fail visibly).
- **Unresolved link** (`dag/executor.py:_resolve_inputs`) — a link whose source
  had not executed was passed raw to the node, corrupting it silently. It now
  raises and aborts the workflow with an `error` status.
- **Queue state machine** (`server/protocol.py` + `server/app.py`) — prompts were
  marked `running` immediately, so `queue_pending` was always empty. They now
  start `queued` and flip to `running` when execution begins, so `GET /queue`
  reflects the real queue.
- **WebSocket event protocol** (`server/app.py` + `dag/executor.py`) — the server
  emitted a non-standard `execution_success` event and never sent the upstream
  `executing`/`executed` events that the ComfyUI frontend relies on for node
  highlight and output preview. `node_event_cb` now emits `executing` before each
  node and `executed` after, and prompt completion sends `executing:{node:None}`
  (the upstream `main.py` convention) instead of `execution_success`.
- **Video model routing** (`core/engine_wrapper.py`) — `hunyuan`/`cosmos`/`svd`
  were absent from `_MODEL_TYPES`, so they fell back to `image` and loaded the
  wrong engine with a 4-channel latent. They now route to `video` (matching the
  Phase 1 plugin wrapper) with correct latent channels (hunyuan/cosmos 16, svd 4).
- **bridge.py label** — the `torch↔mlx` bridge was described as "zero-copy"; it is
  numpy-mediated (torch→numpy→mlx), so the README label was corrected.
- **WebSocket 403 on connect** (`server/app.py`) — the `/ws` endpoint declared its
  first parameter as bare `ws` with no `WebSocket` annotation, so FastAPI 0.139.2
  parsed `ws` as a required *query* parameter. With no `?ws=...` in the request
  the handshake failed validation and uvicorn returned HTTP 403 before `accept()`,
  breaking every WS client (frontend and programmatic). Annotating `ws: WebSocket`
  restores the injected connection; verified live (connect → `status` → ping/pong).
- **`/history` schema** (`server/protocol.py`) — records returned `status` as a
  bare string (`"ok"`/`"error"`), but the upstream ComfyUI `/history` contract (and
  the frontend) expect `status` to be an object `{status_str, completed, messages}`.
  Clients calling `.get("status_str")` on the string raised `AttributeError`. The
  records are now projected to the upstream shape (`status_str` `success`/`error`,
  `completed` flag, `messages` from `errors`), so `/history` is client-compatible.

Covered by new tests in `tests/test_dag_types.py`, `tests/test_dag_executor.py`,
`tests/test_server_static_files.py`, `tests/test_server_protocol.py`,
`tests/test_server_ws.py`, and `tests/test_engine_wrapper_routing.py`
(499 unit passing; 8 e2e skip unless a Phase-1 plugin node server is running).
A separate upstream frontend concern (stable DOM `data-testid` selectors for UI
testing) is tracked in issue
[Comfy-Org/ComfyUI#15392](https://github.com/Comfy-Org/ComfyUI/issues/15392).

## Upstream Dependencies

Phase 1 and Phase 2 upstream issues (all resolved):
- [#170](https://github.com/dahai80/fusion-mlx/issues/170) — ✅ Pipeline stage API (`load_text_encoder`, `encode_text`, `load_dit`, `denoise`, `load_vae`, `decode`)
- [#171](https://github.com/dahai80/fusion-mlx/issues/171) — ✅ Streaming progress callback (`StepCallback`)
- [#172](https://github.com/dahai80/fusion-mlx/issues/172) — ✅ Model registry API (`list_available_models`)

Phase 3 status (fusion-mlx machinery landed):
- Speculative denoising ✅ landed in fusion-mlx (`speculative_denoise.py`: draft-predict + batched-verify), env-gated and default-off. The layer-pruned draft was evaluated and FALSIFIED on SkyReels-V3 R2V 14B (0% acceptance at ε=0.1, no speedup) - machinery stays as infrastructure for a future distilled draft. Stats surface is live: `GET /v1/videos/denoise-stats?model=<name>` + `FusionDenoiseStats` node.
- Radix KV cache: local `RadixCache` implemented (prefix-tree); upstream diffusion KV reuse still future.
- NVFP4 weight reader: blocked on MLX framework issue [mlx#2962](https://github.com/ml-explore/mlx/issues/2962) (not fusion-mlx).
- Metal async dispatch pipeline: still needs fusion-mlx (split command buffer for CPU/GPU overlap).

HunyuanVideo MLX rewrite (all weight-matched, upstream [#15](https://github.com/dahai80/fusion-mlx/issues/15)):
VAE 248/248, DiT 856/856, TextEncoder CLIP-L 196/196 + Llama3-8B 290/290; real tokenizers added; e2e t2v verified.

## Phase 4: macOS Native App

`FusionComfyUI/` is a SwiftPM package: a SwiftUI shell with an embedded WebKit view that wraps the ComfyUI frontend, auto-starts the backend on `127.0.0.1:11445`, and offers model downloads via `fusion-mlx pull` (mirror-aware).

The app does **not** bundle a Python runtime — it launches the dev `.venv` through a tracked `start.sh` at the repo root. Set `FUSION_COMFYUI_START_SH` to point at a different `start.sh` if the app is relocated.

```bash
# Build the .app bundle (unsigned, runs locally)
cd FusionComfyUI && Scripts/build.sh
open ".build/Fusion ComfyUI.app"

# Or run directly from source
cd FusionComfyUI && swift run

# Server lifecycle (the app calls these; you can run them manually)
./start.sh start    # launches ComfyUI on :11445, waits for /system_stats
./start.sh status
./start.sh stop
./start.sh log -f
```

Components:
- `start.sh` — repo-root lifecycle manager (`start|stop|status|log|restart`); activates `/Users/dahai/fusion/.venv`, runs `python ComfyUI/main.py --port 11445 --listen 127.0.0.1`, pidfile + `wait_healthy`, sets `HF_MIRROR=https://hf-mirror.com`.
- `ServerManager.swift` — launches `start.sh start`, probes `GET /system_stats` until healthy, exposes `.stopped/.starting/.running/.failed` state.
- `ModelManager.swift` — lists models from `/object_info` + local `~/.fusion-mlx/models` cache; `Pull` button runs `fusion-mlx pull <repo>` with `HF_MIRROR=https://hf-mirror.com` and streams output.
- `WebView.swift` / `FusionComfyUIApp.swift` — WKWebView loads the ComfyUI frontend once the server is healthy; status dot + Models sheet.

Requires macOS 14+ (Sonoma), Apple Silicon (arm64).

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1+)
- macOS 14+ (Sonoma)
- [fusion-mlx](https://github.com/dahai80/fusion-mlx) >= 0.4.8
