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

## H3 视频超分

`SaveVideo` 节点可在保存时对 H3 视频输出的每一帧通过 fusion-mlx RealESRGAN
超分端点进行放大。该功能由环境变量在保存时触发，默认关闭。

| 环境变量 | 默认值 | 取值 | 说明 |
|---|---|---|---|
| `FUSION_H3_SR` | `0` | `0` \| `1` \| `mock` | `0` = 关闭（不做超分）。`1` = 真实 fusion-mlx RealESRGAN 逐帧超分。`mock` = lanczos 4x 占位，用于在无 SR 模型时验证链路。 |
| `FUSION_H3_SR_SCALE` | `4` | `4` \| `2` | 放大倍数。永不自动降级——由操作者显式选择。 |
| `FUSION_H3_SR_ENDPOINT` | `http://127.0.0.1:11434/v1/images/super-resolution` | URL | fusion-mlx 超分端点。 |
| `FUSION_MLX_API_KEY` | 未设置 | 字符串 | 发往 fusion-mlx 超分端点的 API 密钥（Bearer + x-api-key）。设为 fusion-mlx 当前强制使用的同一密钥（fusion-mlx 启动时的 `FUSION_MLX_API_KEY` 环境变量会覆盖 `~/.fusion-mlx/settings.json`）；配置了密钥的服务端会以 401 拒绝无鉴权头的请求。 |

### 模型准备（用于 `FUSION_H3_SR=1`）

- 需要暴露 RealESRGAN 超分端点（`/v1/images/super-resolution`）的 fusion-mlx
  版本（已在 fusion-mlx v0.8.64 落地——`pyproject.toml` 已锁定
  `fusion-mlx>=0.8.64`）。
- 通过镜像 `https://hf-mirror.com` 下载 `RealESRGAN_x4plus.pth`，转换为
  safetensors，放置于
  `~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.safetensors`。

超分为失败即报错模式：若端点调用失败，`SaveVideo` 将抛出异常——不会静默回退到
模糊的原始帧。

### 时间预算与上线策略

在 Apple Silicon（MLX RealESRGAN_x4plus，`tile_size=512`，完整 H3 输出
1344x768）上的实测：

- `scale=4`：约 39 秒/帧 → 52 帧约 34 分钟纯超分。
- `scale=2`：约 41 秒/帧 → 52 帧约 36 分钟纯超分。

超分每帧成本由**输入分辨率**决定，与输出放大倍数无关：23 层 RRDB 主体在
1344x768 输入上对 `scale=2` 与 `scale=4` 都同样运行（最近邻上采样 + 卷积尾
很廉价），所以 `scale=2` **并非**更快的逃生口。一次完整的 H3 i2v 768p dur2
（约 21 分钟生成）+ 真实超分（约 34 分钟）合计约 55 分钟，超出 30 分钟的编排
截止时间。

画质已验证：RealESRGAN 能恢复被 H3 VAE 16 倍低通滤掉的高频细节——在 4x 输出
分辨率上，真实超分携带的高频能量（拉普拉斯方差）约为 lanczos 放大同一模糊帧
的 10000 倍（lanczos 不恢复任何细节，只是放大）。`mock` 路径仅是外观改善
（更大但仍模糊）。

上线策略：`mock` 为编排链路的截止安全默认值；当画质比墙钟时间更重要时设置
`FUSION_H3_SR=1` 用于离线渲染。

```bash
# 示例：真实 4x 超分（离线，画质优先于速度）
FUSION_H3_SR=1 FUSION_H3_SR_SCALE=4 FUSION_MLX_API_KEY=<your-key>

# 示例：2x 超分（文件更小；并非更快——见上方时间预算）
FUSION_H3_SR=1 FUSION_H3_SR_SCALE=2 FUSION_MLX_API_KEY=<your-key>

# 示例：mock 模式（链路验证 / 截止安全默认值，无 SR 模型）
FUSION_H3_SR=mock
```

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
