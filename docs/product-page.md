---
doc_id: fusion-comfyui/product-page
product: fusion-comfyui
class: app
entry: service
doc_type: product-page
title: { zh: "fusion-comfyui 产品页", en: "fusion-comfyui product page" }
audience: engineer
status: draft
version: 0.2.17
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - pyproject.toml
  - start.sh
  - fusion_comfyui/launch_comfyui.py
---

# fusion-comfyui 产品页（Product Page）

> 定位：Apple Silicon 上的纯 MLX ComfyUI 服务器——运行时零 PyTorch，以 ComfyUI 协议对外提供 FLUX.2 / Wan2.2 / SkyReels-V3 / LTX-2 图像与视频生成（README.md）。
> Positioning: a pure-MLX ComfyUI server on Apple Silicon — zero PyTorch at runtime, exposing FLUX.2 / Wan2.2 / SkyReels-V3 / LTX-2 image and video generation over the ComfyUI protocol (README.md).

## 网格定位（Grid position）

- class = `app`：面向使用者的图像/视频生成服务产品。
- entry = `service`：入口是常驻 HTTP 服务（ComfyUI 协议，默认 `127.0.0.1:11445`），由 `start.sh` 管理；Swift 壳应用（Phase 4）只是该服务的桌面封装。
- 证据：`start.sh` 的生命周期管理（start/stop/status/restart/log，健康探针 `GET /system_stats`，PID/日志落在 `.run/`）；`fusion_comfyui/launch_comfyui.py` 以 `fusion-comfyui serve :PORT` 进程标题拉起内置 ComfyUI 宿主；`pyproject.toml` 声明 `mlx>=0.22.0` + `fusion-mlx>=0.8.64`，无 torch 依赖。

## 使用者能力（What users no longer have to do）

| 能力 | 使用者不再需要做 | 约束 / 限制 |
|------|------------------|-------------|
| Apple Silicon 生成管线 | 不用装 PyTorch 运行时；推理全程走 MLX/Metal | 需要 fusion-mlx≥0.8.64 在同一 venv 内可导入（进程内调用，见下） |
| ComfyUI 工作流 | 不用改现有 ComfyUI 工作流；协议与前端直接兼容，自定义节点即插即用 | 宿主是仓库内置的 ComfyUI 树（未入库，本地工作副本） |
| 模型缓存 | 不用重复下载模型；HF 缓存对齐 `~/.fusion-mlx/models`，并强制 HF 离线模式 | 新模型需先经镜像站下载进该缓存 |
| 鉴权配置 | 不用手写 key 解析器；统一走 `fusion_core.config.resolve_api_key`（`FUSION_CLIENT_KEY` → `~/.fusion-mlx/settings.json`） | `start.sh` 会清掉继承的 client-key 环境变量，认证只认 settings.json |
| 生命周期管理 | 不用手工管 PID/日志；`start.sh` 提供 start/stop/status/restart/log 与健康探针 | 端口必须在 `architecture/port-registry.yaml` 注册，否则拒绝启动（issue #76） |
| 桌面使用 | 不用开终端；Phase 4 SwiftUI 壳首启自动 bootstrap 专用 venv 并拉起后端 | 壳不打包 Python 运行时，首启需 Homebrew Python≥3.11 联网装依赖 |

## 架构摘要（Architecture summary）

单核结构（v0.2.13 P1 统一后）：`fusion_comfyui/core/` 是唯一核心（config / engine_wrapper / lifecycle / bridge / wrappers / async_utils / timer / output_store），`fusion_comfyui_plugin/` 是 ComfyUI 自定义节点包（15 个节点模块，经 `pip install -e` 导入核心，无 sys.path hack）；`FusionComfyUI/` 是 SwiftPM SwiftUI 壳（WebKit 前端 + ServerManager + ModelManager）。已删除的独立 FastAPI 服务器不再存在，ComfyUI 自身编排是唯一运行时。

与 fusion-mlx 的连接是**进程内导入**（`fusion_mlx.public_api`），是 monorepo HTTP 路由约定的已声明例外（`architecture/netlayer-compliance-plan.md` §5.5）：无网络跳数、无 `X-Fusion-Source` 头。

## 边界（Boundaries）

- Phase 3 加速项如实标注：spec denoise 机制已落地但默认关（在 SkyReels-V3 上 0% 接受率，加速被证伪）；Radix KV cache 对 T5 前缀复用被证伪（保留数据结构）；NVFP4 阻塞在上游 mlx#2962。
- H3 视频 SR 默认关（`FUSION_H3_SR=0`）；`mock` 模式是编排链的 deadline 安全默认，真 SR 约 39s/帧（scale=2 并不更快，成本随输入分辨率）。
- SR 是 fail-loud：端点失败直接抛错，不静默回退到原始模糊帧。

## 文档地图（Doc map）

- 快速上手：`docs/get-started.md`
- 接口参考：`docs/api-reference.md`
- 配置参考：`docs/config-reference.md`
- 故障排查：`docs/troubleshooting.md`
- 插件与节点：`fusion_comfyui_plugin/README.md`
