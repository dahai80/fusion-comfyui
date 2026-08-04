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
| Phase 3 | 🔧 In Progress | Radix cache ✅, enhanced nodes ✅, spec denoise/NVFP4/async dispatch need fusion-mlx |
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
│   ├── bridge.py        # torch↔mlx zero-copy bridge (Phase 1 only)
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

## Phase 3 Configuration

Phase 3 features are controlled via environment variables (require fusion-mlx support):

```bash
# Speculative denoising
FUSION_SPECULATIVE_DENOISING=1
FUSION_SPEC_DRAFT_STEPS=2
FUSION_SPEC_DRAFT_MODEL=flux-schnell

# Radix KV cache
FUSION_RADIX_CACHE_ENABLED=1
FUSION_RADIX_CACHE_MAX_MB=512

# NVFP4 weight ingestion
FUSION_NVFP4_ENABLED=1
FUSION_NVFP4_THRESHOLD_GB=8
```

## Upstream Dependencies

Phase 1 and Phase 2 upstream issues (all resolved):
- [#170](https://github.com/dahai80/fusion-mlx/issues/170) — ✅ Pipeline stage API (`load_text_encoder`, `encode_text`, `load_dit`, `denoise`, `load_vae`, `decode`)
- [#171](https://github.com/dahai80/fusion-mlx/issues/171) — ✅ Streaming progress callback (`StepCallback`)
- [#172](https://github.com/dahai80/fusion-mlx/issues/172) — ✅ Model registry API (`list_available_models`)

Phase 3 features still depend on future fusion-mlx support:
- Speculative denoising infrastructure (draft model co-loading + parallel verify)
- Radix KV cache for diffusion models (prefix-tree KV cache manager — local RadixCache ✅ implemented)
- NVFP4 weight reader (E2M1 + block scale dequant)
- Metal async dispatch pipeline (split command buffer for CPU/GPU overlap)

HunyuanVideo MLX rewrite (all weight-matched, upstream [#15](https://github.com/dahai80/fusion-mlx/issues/15)):
VAE 248/248, DiT 856/856, TextEncoder CLIP-L 196/196 + Llama3-8B 290/290; real tokenizers added; e2e t2v verified.

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1+)
- macOS 14+ (Sonoma)
- [fusion-mlx](https://github.com/dahai80/fusion-mlx) >= 0.4.8
