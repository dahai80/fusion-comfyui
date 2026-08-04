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
| Phase 3 | 🔧 进行中 | 推测去噪机制 ✅（已落地，默认关闭，加速已证伪），Radix 缓存 ✅，统计节点 ✅，NVFP4 受阻（mlx#2962），异步调度需 fusion-mlx |
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

Phase 3 功能通过环境变量控制。推测去噪机制已在 fusion-mlx 落地（环境变量门控，默认关闭）；Radix 缓存为本地实现；NVFP4 受上游阻塞（见下）。

```bash
# 推测去噪（fusion-mlx 机制已落地，默认关闭）
FUSION_SPECULATIVE_DENOISE=1
FUSION_SPEC_K=4
FUSION_SPEC_EPSILON=0.1
FUSION_SPEC_DRAFT_BLOCKS=
FUSION_SPEC_EVAL_STEPS=1

# Radix KV 缓存（本地 RadixCache 已实现）
FUSION_RADIX_CACHE_ENABLED=1
FUSION_RADIX_CACHE_MAX_MB=512

# NVFP4 权重加载（受 MLX 框架 issue mlx#2962 阻塞）
FUSION_NVFP4_ENABLED=1
FUSION_NVFP4_THRESHOLD_GB=8
```

去噪统计可在运行时查询：`FusionDenoiseStats` 节点（Phase 1 自定义节点与 Phase 2 注册表均支持）
返回最近一次去噪的接受率/加速比计数（JSON），fusion-mlx 暴露
`GET /v1/videos/denoise-stats?model=<name>`。

## 上游依赖

Phase 1 和 Phase 2 上游问题（已全部解决）：
- [#170](https://github.com/dahai80/fusion-mlx/issues/170) — ✅ 管线阶段 API（`load_text_encoder`、`encode_text`、`load_dit`、`denoise`、`load_vae`、`decode`）
- [#171](https://github.com/dahai80/fusion-mlx/issues/171) — ✅ 流式进度回调（`StepCallback`）
- [#172](https://github.com/dahai80/fusion-mlx/issues/172) — ✅ 模型注册 API（`list_available_models`）

Phase 3 状态（fusion-mlx 机制已落地）：
- 推测去噪 ✅ 已在 fusion-mlx 落地（`speculative_denoise.py`：草稿预测 + 批量验证），环境变量门控、默认关闭。层剪枝草稿在 SkyReels-V3 R2V 14B 上经评估证伪（ε=0.1 时接受率 0%，无加速）——机制保留为未来蒸馏草稿的基础设施。统计面已上线：`GET /v1/videos/denoise-stats?model=<name>` + `FusionDenoiseStats` 节点。
- Radix KV 缓存：本地 `RadixCache` ✅ 已实现（前缀树）；上游扩散 KV 复用仍属未来。
- NVFP4 权重读取器：受 MLX 框架 issue [mlx#2962](https://github.com/ml-explore/mlx/issues/2962) 阻塞（非 fusion-mlx）。
- Metal 异步调度管线：仍需 fusion-mlx（拆分命令缓冲区实现 CPU/GPU 重叠）。

## 系统要求

- Python 3.10+
- Apple Silicon Mac（M1+）
- macOS 14+（Sonoma）
- [fusion-mlx](https://github.com/dahai80/fusion-mlx) >= 0.4.8
