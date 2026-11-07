---
doc_id: fusion-comfyui/troubleshooting
product: fusion-comfyui
class: app
entry: service
doc_type: troubleshooting
title: { zh: "fusion-comfyui 故障排查", en: "fusion-comfyui troubleshooting" }
audience: engineer
status: draft
version: 0.2.17
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - start.sh
  - README.md
  - fusion_comfyui/launch_comfyui.py
---

# fusion-comfyui 故障排查（Troubleshooting）

> 症状 → 定位 → 处置。日志统一在 `.run/comfyui.log`（`./start.sh log`）。

## 1. 启动失败

### 1.1 `start.sh` 拒绝启动：端口未注册

- 定位：Phase 3.3（issue #76）启动前自检 `architecture/port-registry.yaml`，`fusion-comfyui` 条目的 `port:` 必须与 `FUSION_COMFYUI_PORT` 一致。
- 处置：在注册表补条目；临时调试可 `FUSION_SKIP_REGISTRY_CHECK=1`（会打 warn，不推荐长期使用）。

### 1.2 `ComfyUI main.py not found`

- 定位：`launch_comfyui.py` 要求仓库内存在 `ComfyUI/main.py`（宿主树是本地工作副本，未入库）。
- 处置：恢复 `ComfyUI/` 目录；不要把它入库，也不要移动位置。

### 1.3 健康探针超时（60s）

- 定位：`GET /system_stats` 未在 `FUSION_HEALTH_TIMEOUT` 内就绪；多为模型导入慢或依赖缺失。
- 处置：`./start.sh log` 看最后 40 行（脚本超时会自动打印尾部）；确认 venv 正确（`FUSION_VENV`）且 `pip install -e .` 已执行。

### 1.4 进程标题是裸 `Python`

- 定位：`launch_comfyui.py` 经 `setproctitle` 设置 `fusion-comfyui serve :PORT`（issue #74）；未生效说明绕过了 launcher 直接跑 `ComfyUI/main.py`。
- 处置：统一经 `./start.sh` 或 `launch_comfyui.py` 启动。

## 2. 模型与下载

### 2.1 模型加载报找不到 / HF 网络错误

- 定位：`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` 是强制离线（省掉无谓的镜像探测超时）；模型必须在 `HUGGINGFACE_HUB_CACHE`（默认 `~/.fusion-mlx/models`）内。
- 处置：用镜像站 `https://hf-mirror.com` 预先下载进该缓存；缓存与 fusion-mlx 共用，同模型不重复下载。

### 2.2 H3 SR 报模型缺失

- 定位：`FUSION_H3_SR=1` 需要 `~/.fusion-mlx/models/realesrgan/RealESRGAN_x4plus.safetensors`。
- 处置：按 README H3 节下载 `RealESRGAN_x4plus.pth`（走 hf-mirror）并转 safetensors；链路验证可先用 `FUSION_H3_SR=mock`（lanczos 占位，不加载模型）。

## 3. 鉴权

### 3.1 SR 调用 401

- 定位：`start.sh` 启动时会清除继承的 client-key 环境变量（glob 匹配，issue #76），认证统一来自 `~/.fusion-mlx/settings.json`；key 配置过的 fusion-mlx 会拒绝无头请求。
- 处置：把 key 写进 `~/.fusion-mlx/settings.json` 的 `auth.api_key`，或在直连 launch（不经 start.sh）时显式传 `FUSION_CLIENT_KEY`。废弃名 `FUSION_MLX_API_KEY` / `MLX_API_KEY` 仍可经 fusion-core 别名链解析但会有告警（issue #75）。

### 3.2 key 解析告警 "settings.json missing/malformed"

- 定位：`fusion_core.config.resolve_api_key` 找不到或解析不了 settings 文件时降级为匿名（空 key）+ 告警，不中断启动。
- 处置：修复 `~/.fusion-mlx/settings.json`；本地无鉴权部署可忽略该告警。

## 4. 运行时行为

### 4.1 视频模型加载了错误引擎（4 通道 latent 报错）

- 定位：旧版 `hunyuan`/`cosmos`/`svd` 不在 `_MODEL_TYPES`，会回退到 image 引擎；该 bug 已在统一 `core/engine_wrapper.py` 修复（hunyuan/cosmos 16 通道、svd 4 通道，README Reliability 节）。
- 处置：升级到当前版本；回归测试在 `tests/test_engine_wrapper_routing.py` 与 `tests/test_engine_wrapper_h3_routing.py`。

### 4.2 找不到 `/ws` 403 修复或自研 `/history` 投影

- 定位：这些修复属于 v0.2.x 独立 FastAPI 服务器，该服务器已在 v0.2.13（P1 统一）移除；现由 ComfyUI 上游 `server.py`/`execution.py` 负责。
- 处置：按 ComfyUI 上游Issue 渠道反馈；本仓库不再携带这些补丁。

### 4.3 SR 很慢 / scale=2 没有变快

- 定位：SR 成本绑定输入分辨率（23 层 RRDB 在 1344x768 输入上运行），scale=2 与 scale=4 实测同为约 39-41s/帧；H3 整片 SR 约 34-36 分钟。
- 处置：编排链上保持 `FUSION_H3_SR=mock`（deadline 安全默认）；只在离线渲染且清晰度优先时用 `FUSION_H3_SR=1`（README Time budget 节）。

## 5. 测试

### 5.1 e2e 用例全部 skip

- 定位：e2e 仅在 `RUN_E2E=1` 且存在 LIVE ComfyUI 服务器时执行（README Reliability 节）。
- 处置：先 `./start.sh start`，再 `RUN_E2E=1` 运行；单元测试（插件 485 个）无需真实服务器。
