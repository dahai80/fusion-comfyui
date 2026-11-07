---
doc_id: fusion-comfyui/get-started
product: fusion-comfyui
class: app
entry: service
doc_type: get-started
title: { zh: "fusion-comfyui 快速上手", en: "fusion-comfyui quick start" }
audience: engineer
status: draft
version: 0.2.17
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - start.sh
  - README.md
  - fusion_comfyui/launch_comfyui.py
---

# fusion-comfyui 快速上手（Get Started）

> 目标：装好包、起服务、用 ComfyUI 前端跑通一次生成。

## 1. 前置条件

- macOS，Apple Silicon；Python ≥3.10（推荐 monorepo 共享 venv `/Users/dahai/fusion/.venv`，`start.sh` 默认用它，可用 `FUSION_VENV` 覆盖）
- fusion-mlx ≥0.8.64 可在同一 venv 导入（进程内调用，无 HTTP 依赖）
- 仓库内的 ComfyUI 宿主树（`ComfyUI/` 本地工作副本，未入库）
- 模型走镜像站 `https://hf-mirror.com`，缓存目录与 fusion-mlx 共用 `~/.fusion-mlx/models`

## 2. 安装与启动

```bash
cd /Users/dahai/fusion/fusion-comfyui
source /Users/dahai/fusion/.venv/bin/activate
pip install -e .

./start.sh start          # 默认 127.0.0.1:11445
./start.sh status
```

确认点：

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11445/system_stats
# 200
```

- 进程标题应为 `fusion-comfyui serve :11445`（issue #74，非裸 `Python`）
- 端口未在 `architecture/port-registry.yaml` 注册时启动会被拒绝（issue #76；逃生口 `FUSION_SKIP_REGISTRY_CHECK=1` 仅限临时调试）
- 日志：`.run/comfyui.log`；`./start.sh log` 可直接查看

## 3. 跑通第一次生成

1. 打开 `http://127.0.0.1:11445` —— ComfyUI 前端直连，无需另起前端。
2. 用默认工作流提交一次图生成；输出落 `output/`（`FUSION_OUTPUT_DIR` 可覆盖）。
3. 确认点：队列页任务完成，`output/` 出现产物文件。

说明：`fusion-comfyui serve` 控制台脚本与独立 FastAPI 服务器已在 v0.2.13（P1 统一）移除；如老文档提到它们，以本页与 `start.sh` 为准。

## 4. 桌面壳（可选，Phase 4）

```bash
cd FusionComfyUI && Scripts/build.sh all    # 出 .app + DMG（未签名，本地运行）
open ".build/Fusion ComfyUI.app"
```

首启由 `SetupManager` 在 `~/.fusion-comfyui/venv` bootstrap 专用 venv（需 Homebrew Python≥3.11），后续启动直接复用；venv 根可用 `FUSION_COMFYUI_VENV_ROOT` 覆盖，`FUSION_COMFYUI_START_SH` 指定 relocated 的 `start.sh`。

## 5. 测试

```bash
python -m pytest tests/ -q                          # 本仓库核心测试
python -m pytest fusion_comfyui_plugin/tests/ -q    # 插件 485 个单元测试（README Reliability 节）
RUN_E2E=1 python -m pytest ...                      # e2e 仅在 LIVE ComfyUI 服务器存在时执行
```

## 常见问题速查

```yaml
troubleshooting:
  - q: { zh: "start.sh 报端口未注册拒绝启动", en: "start.sh refuses to start: port not registered" }
    a: { zh: "在 architecture/port-registry.yaml 给 fusion-comfyui 补 11445 条目，或临时 FUSION_SKIP_REGISTRY_CHECK=1。", en: "Register port 11445 for fusion-comfyui in port-registry.yaml, or temporarily set FUSION_SKIP_REGISTRY_CHECK=1." }
  - q: { zh: "模型加载超时", en: "Model load times out" }
    a: { zh: "HF 已强制离线（HF_HUB_OFFLINE=1），确认模型已在 ~/.fusion-mlx/models 缓存内。", en: "HF is forced offline (HF_HUB_OFFLINE=1); make sure the model is already in the ~/.fusion-mlx/models cache." }
```

## 提交检查清单

- [x] 所有命令来自 `start.sh` / `launch_comfyui.py` / README Phase 4 节，真实可执行
- [x] 每步有确认点；已注明已移除的旧入口
- [x] zh/en 成对（命令除外）
