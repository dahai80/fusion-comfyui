# fusion-comfyui

[English](README.md) | **中文**

纯 MLX ComfyUI 服务器，运行在 Apple Silicon 上。运行时零 PyTorch 依赖。

基于 [fusion-mlx](https://github.com/dahai80/fusion-mlx) 引擎构建 — 通过 Metal/MLX 实现 FLUX.2、Wan2.2、SkyReels-V3、LTX-2 图像/视频生成。

## 快速开始

```bash
# 安装
pip install -e .

# 启动独立服务器（纯 MLX，无 PyTorch）
fusion-comfyui serve --port 11443

# 或作为 ComfyUI 自定义节点运行（Phase 1，需要 PyTorch 宿主）
cd ComfyUI && python main.py
```

打开 `http://localhost:11443` — ComfyUI 前端直接连接。

## 架构

| 阶段 | 状态 | 说明 |
|---|---|---|
| Phase 1 | ✅ 已完成 | ComfyUI 自定义节点（PyTorch 宿主 + MLX 计算） |
| Phase 2 | ✅ 已完成 | 独立 FastAPI 服务器，ComfyUI 协议，零 PyTorch |
| Phase 3 | 🔧 进行中 | Radix 缓存 ✅，增强节点 ✅，spec denoise/NVFP4/async dispatch 需要 fusion-mlx 支持 |
| Phase 4 | 🔧 进行中 | Swift 应用拆分 ✅，ServerManager/ModelManager ✅，需要打包 |

详见 [CONSTRUCTION_PLAN.md](CONSTRUCTION_PLAN.md)。

## 项目结构

```
fusion_comfyui/          # Phase 2+ 独立服务器
├── core/
│   ├── config.py        # Phase 3 配置 + RadixCache（基数树）
│   ├── engine_wrapper.py # fusion-mlx 进程内封装
│   ├── lifecycle.py     # 内存守护 + 管线阶段上下文
│   └── output_store.py  # 基于文件的输出存储（/view 端点）
├── dag/
│   ├── executor.py      # 拓扑排序 + 顺序执行
│   └── types.py         # NodeDef, LinkDef, Workflow, KNOWN_TYPES
├── nodes/
│   ├── base.py          # BaseNode 抽象类
│   └── registry.py      # 所有纯 MLX 节点实现（Phase 2 + 3）
├── server/
│   ├── app.py           # FastAPI 应用 + 路由挂载
│   ├── protocol.py      # ComfyUI 协议端点
│   ├── ws.py            # WebSocket 进度流
│   └── static_files.py  # 输出 + 前端静态文件服务
└── main.py              # CLI 入口

ComfyUI/custom_nodes/ComfyUI-Fusion-MLX/  # Phase 1 自定义节点
├── core/
│   ├── bridge.py        # torch↔mlx 零拷贝桥接（仅 Phase 1）
│   ├── lifecycle.py     # FusionMemoryGuardian + PipelineStageContext
│   └── engine_wrapper.py # fusion-mlx 封装（torch 兼容）
├── nodes/
│   ├── loaders.py       # FusionModelLoaderNode
│   ├── conditioning.py  # FusionTextEncoderNode
│   ├── samplers.py      # FusionKSamplerNode
│   └── vae.py           # FusionVAEDecoderNode
└── __init__.py

FusionComfyUI/           # Phase 4 macOS 原生应用（Swift）
├── FusionComfyUIApp.swift  # 应用入口 + ContentView
├── ServerManager.swift     # 启动/监控 Python 服务器
├── WebView.swift           # WebKit ComfyUI 前端
├── ModelManager.swift      # 模型发现 + 下载
└── Package.swift
```

## Phase 3 配置

Phase 3 功能通过环境变量控制（需要 fusion-mlx 支持）：

```bash
# 推测去噪
FUSION_SPECULATIVE_DENOISING=1
FUSION_SPEC_DRAFT_STEPS=2
FUSION_SPEC_DRAFT_MODEL=flux-schnell

# Radix KV 缓存
FUSION_RADIX_CACHE_ENABLED=1
FUSION_RADIX_CACHE_MAX_MB=512

# NVFP4 权重加载
FUSION_NVFP4_ENABLED=1
FUSION_NVFP4_THRESHOLD_GB=8
```

## 上游依赖

Phase 1 和 Phase 2 上游问题（已全部解决）：
- [#170](https://github.com/dahai80/fusion-mlx/issues/170) — ✅ 管线阶段 API（`load_text_encoder`、`encode_text`、`load_dit`、`denoise`、`load_vae`、`decode`）
- [#171](https://github.com/dahai80/fusion-mlx/issues/171) — ✅ 流式进度回调（`StepCallback`）
- [#172](https://github.com/dahai80/fusion-mlx/issues/172) — ✅ 模型注册 API（`list_available_models`）

Phase 3 功能仍依赖未来 fusion-mlx 支持：
- 推测去噪基础设施（草稿模型共加载 + 并行验证）
- 扩散模型 Radix KV 缓存（前缀树 KV 缓存管理器 — 本地 RadixCache ✅ 已实现）
- NVFP4 权重读取器（E2M1 + block scale 反量化）
- Metal 异步调度管线（拆分命令缓冲区实现 CPU/GPU 重叠）

## 系统要求

- Python 3.10+
- Apple Silicon Mac（M1+）
- macOS 14+（Sonoma）
- [fusion-mlx](https://github.com/dahai80/fusion-mlx) >= 0.4.8
