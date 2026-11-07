---
doc_id: fusion-comfyui/api-reference
product: fusion-comfyui
class: app
entry: service
doc_type: api-reference
title: { zh: "fusion-comfyui 接口参考", en: "fusion-comfyui API reference" }
audience: engineer
status: draft
version: 0.2.17
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - start.sh
  - fusion_comfyui/core/output_store.py
---

# fusion-comfyui 接口参考（API Reference）

> 对外协议是 **ComfyUI 原生协议**（由内置 ComfyUI 宿主的 `server.py`/`execution.py` 提供，v0.2.13 P1 统一后的唯一运行时）。本仓库不另设 API 层；本文只列入口与扩展点。

## 服务入口（Endpoints）

| 路径 | 方法 | 说明 | 来源 |
|------|------|------|------|
| `/` | GET | ComfyUI 前端（直连，无需另起前端） | ComfyUI 宿主 |
| `/system_stats` | GET | 系统统计；`start.sh` 的健康探针即打此端点（`HEALTH_URL`） | ComfyUI 宿主 |
| `/prompt` | POST | 提交工作流队列（ComfyUI 协议标准入口） | ComfyUI 宿主 |
| `/history` | GET | 任务历史 | ComfyUI 宿主 |
| `/view` | GET | 读取输出产物（经 `fusion_comfyui/core/output_store.py` 的文件化 output store 支撑） | ComfyUI 宿主 + 本仓库 |
| `/ws` | WebSocket | 队列/进度事件流 | ComfyUI 宿主 |

> 注：v0.2.x 时代自研的 `/ws` 403 修复、`/history` schema 投影等随独立服务器一同移除，现由 ComfyUI 上游实现负责（README "Historical note"）。

## 自定义节点（Plugin nodes，`fusion_comfyui_plugin/`）

15 个节点模块全部 import `fusion_comfyui.core.*`：

- drama 管线节点（`fusion_comfyui/nodes/drama/` + 插件侧）：tts、lipsync、pulid、assemble、vlm 等，继承 `fusion_comfyui/nodes/base.py` 的 `BaseNode`。
- `FusionDenoiseStats` 节点（插件 `nodes/stats.py`）：返回最近一次 denoise 的接受率/加速计数器 JSON。
- 节点输出统一落 `output/`（`FUSION_OUTPUT_DIR` 可覆盖）。

## fusion-mlx 侧的关联端点（in-process，非本服务暴露）

- 引擎能力经 `fusion_mlx.public_api` 进程内导入：`ImageGenEngine`、`VideoGenEngine`、`TTSEngine`、`LipsyncPipelineMLX`、`PuLIDPipeline`、`EnginePool`、`list_available_models`、`MuseTalkPipeline`、`VLMBatchedEngine`（fusion-mlx PR #620/#625 落地）。
- denoise 统计在 fusion-mlx 侧可经 `GET /v1/videos/denoise-stats?model=<name>` 查询（如该服务在跑）。
- H3 SR 调用 fusion-mlx 的 `POST /v1/images/super-resolution`（`FUSION_H3_SR_ENDPOINT`，默认 `http://127.0.0.1:11434/v1/images/super-resolution`），鉴权 Bearer + x-api-key，key 来自 `fusion_core.config.resolve_api_key`。

## 错误语义（Error semantics）

- SR 是 fail-loud：SR 端点失败时 `SaveVideo` 节点直接抛错，不回退输出原始帧（README H3 节）。
- API key 解析失败降级为匿名（空 key）并告警，不中断启动（issue #75 语义）。
- ComfyUI 协议本身的错误结构（node errors、执行中断）由宿主协议定义，客户端按 ComfyUI 前端惯例处理。
