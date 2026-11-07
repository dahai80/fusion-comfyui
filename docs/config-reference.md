---
doc_id: fusion-comfyui/config-reference
product: fusion-comfyui
class: app
entry: service
doc_type: config-reference
title: { zh: "fusion-comfyui 配置参考", en: "fusion-comfyui config reference" }
audience: engineer
status: draft
version: 0.2.17
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - start.sh
  - README.md
  - fusion_comfyui/core/config.py
---

# fusion-comfyui 配置参考（Config Reference)

> 配置面 = `start.sh` 环境变量（服务生命周期）+ Phase 3/H3 功能环境变量（默认关）。无独立配置文件；全部值可在 `start.sh` 与 `fusion_comfyui/core/config.py` 中溯源。

## 服务生命周期（start.sh）

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `FUSION_VENV` | `/Users/dahai/fusion/.venv` | 启动用的 venv python 路径 |
| `FUSION_COMFYUI_HOST` | `127.0.0.1` | 监听地址 |
| `FUSION_COMFYUI_PORT` | `11445` | 监听端口；必须在 `architecture/port-registry.yaml` 注册（issue #76），未注册拒绝启动 |
| `FUSION_HEALTH_TIMEOUT` | `60` | 健康探针等待秒数（探 `GET /system_stats`） |
| `FUSION_OUTPUT_DIR` | `<repo>/output` | 产物输出目录 |
| `FUSION_REGISTRY` | `~/fusion/architecture/port-registry.yaml` | 端口注册表路径 |
| `FUSION_SKIP_REGISTRY_CHECK` | 未设 | 设 `1` 跳过端口注册自检（仅临时调试） |
| `FUSION_IMAGE_TIMEOUT` | `3600` | 单图生成超时（秒） |

## 网络与缓存（start.sh 固定导出）

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `HF_MIRROR` | `https://hf-mirror.com` | 模型下载镜像 |
| `HUGGINGFACE_HUB_CACHE` | `~/.fusion-mlx/models` | 与 fusion-mlx 共享模型缓存，避免重复下载 |
| `HF_HUB_OFFLINE` | `1` | 强制 HF 离线：模型必须在缓存内，网络探测直接跳过 |
| `TRANSFORMERS_OFFLINE` | `1` | 同上，transformers 层离线 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | 兼容宿主个别 op 的 MPS 回退（运行时仍零 torch 推理依赖） |

## Phase 3 加速开关（全部默认关）

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `FUSION_SPECULATIVE_DENOISE` | `0` | 投机 denoise 总开关；机制在 fusion-mlx 已落地，SkyReels-V3 上 0% 接受率（加速被证伪），仅作基础设施保留 |
| `FUSION_SPEC_K` | `4` | 每次验证的 draft 步数 |
| `FUSION_SPEC_EPSILON` | `0.1` | 接受阈值 |
| `FUSION_SPEC_DRAFT_BLOCKS` | 空 | draft block 选择 |
| `FUSION_SPEC_EVAL_STEPS` | `1` | 验证步数 |
| `FUSION_RADIX_CACHE_ENABLED` | `0` | Radix KV cache 开关（对 T5 前缀复用已证伪；类保留） |
| `FUSION_RADIX_CACHE_MAX_MB` | `512` | 缓存容量上限 |
| `FUSION_NVFP4_ENABLED` | `0` | NVFP4 权重读取（阻塞于上游 mlx#2962，开启无效） |
| `FUSION_NVFP4_THRESHOLD_GB` | `8` | NVFP4 触发阈值 |

## H3 视频超分（SaveVideo 节点，默认关）

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `FUSION_H3_SR` | `0` | `0`=关；`1`=真实 fusion-mlx RealESRGAN 逐帧 SR；`mock`=lanczos 4x 占位（链路验证/deadline 安全默认） |
| `FUSION_H3_SR_SCALE` | `4` | `4` 或 `2`；不会自动降档。注意 scale=2 并不更快（成本绑定输入分辨率，实测约 39-41s/帧 @1344x768） |
| `FUSION_H3_SR_ENDPOINT` | `http://127.0.0.1:11434/v1/images/super-resolution` | fusion-mlx SR 端点（需 fusion-mlx≥0.8.64） |
| `FUSION_CLIENT_KEY` | 未设 | 打到 SR 端点的 API key（Bearer + x-api-key）。规范名来自 fusion-core 契约（issue #76）；废弃名 `FUSION_MLX_API_KEY` 仍可经别名链解析。注意 `start.sh` 会清除继承的 client-key 环境变量——认证以 `~/.fusion-mlx/settings.json` 为准 |

## 模型文件（H3 SR = 1 时）

- 需要 `~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.safetensors`（经 `https://hf-mirror.com` 下载 `RealESRGAN_x4plus.pth` 后转 safetensors）。

## 桌面壳（FusionComfyUI/）

| 环境变量 | 说明 |
|----------|------|
| `FUSION_COMFYUI_VENV_ROOT` | 覆盖壳 bootstrap venv 的根目录（默认 `~/.fusion-comfyui`） |
| `FUSION_COMFYUI_START_SH` | 指向 relocated 后的 `start.sh` |
| `FUSION_VENV` | 壳把 bootstrap 出的 venv 通过它传给 `start.sh` |
