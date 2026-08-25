#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${FUSION_VENV:-/Users/dahai/fusion/.venv}"
VENV_PYTHON="$VENV_DIR/bin/python"
RUN_DIR="$PROJECT_DIR/.run"
PID_FILE="$RUN_DIR/comfyui.pid"
LOG_FILE="$RUN_DIR/comfyui.log"
HOST="${FUSION_COMFYUI_HOST:-127.0.0.1}"
PORT="${FUSION_COMFYUI_PORT:-11445}"
HEALTH_URL="http://$HOST:$PORT/system_stats"
HEALTH_TIMEOUT="${FUSION_HEALTH_TIMEOUT:-60}"

export HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export FUSION_OUTPUT_DIR="${FUSION_OUTPUT_DIR:-$PROJECT_DIR/output}"
# Match fusion-mlx's model cache so hf_hub_download finds already-downloaded
# models (SD1.5, etc.) under ~/.fusion-mlx/models without re-downloading.
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/.fusion-mlx/models}"
# Force HF/transformers offline: models live in the cache above, so any network
# probe (e.g. transformers 5.x list_repo_templates on a repo-id tokenizer load,
# fusion-mlx wan2 umt5-xxl) only wastes time and times out when the mirror is
# unreachable. HF_HUB_OFFLINE=1 makes huggingface_hub skip the network entirely.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Image-gen per-call timeout (fusion-mlx #481). SD1.5 1024 img2img (hires-fix
# 2nd pass) can exceed the 600s default on Apple Silicon.
export FUSION_IMAGE_TIMEOUT="${FUSION_IMAGE_TIMEOUT:-3600}"

# 运维修补：清掉 shell 继承的 FUSION_MLX_API_KEY（可能过期 → TTS http backend 401）。
# 统一走 ~/.fusion-mlx/settings.json auth.api_key，与 mlx daemon 同源。
unset FUSION_MLX_API_KEY

mkdir -p "$RUN_DIR"

log() { echo "[start.sh] $*"; }

get_pid() {
    [ -f "$PID_FILE" ] || { echo ""; return 0; }
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [ -n "$pid" ] || { echo ""; return 0; }
    if kill -0 "$pid" 2>/dev/null; then
        echo "$pid"
    else
        rm -f "$PID_FILE"
        echo ""
    fi
}

is_running() {
    local pid
    pid="$(get_pid)"
    [ -n "$pid" ]
}

probe_health() {
    curl -s -o /dev/null --max-time 2 "$HEALTH_URL" 2>/dev/null
}

wait_healthy() {
    local waited=0
    while [ "$waited" -lt "$HEALTH_TIMEOUT" ]; do
        if probe_health; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

do_start() {
    if is_running; then
        local pid
        pid="$(get_pid)"
        log "already running PID=$pid on $HOST:$PORT"
        return 0
    fi
    if [ ! -x "$VENV_PYTHON" ]; then
        log "ERROR: venv python not found at $VENV_PYTHON"
        return 2
    fi
    log "starting ComfyUI on $HOST:$PORT ..."
    nohup "$VENV_PYTHON" "$PROJECT_DIR/ComfyUI/main.py" \
        --listen "$HOST" --port "$PORT" \
        > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    log "launched PID=$pid, waiting for health (up to ${HEALTH_TIMEOUT}s) ..."
    if wait_healthy; then
        log "healthy at $HEALTH_URL (PID=$pid)"
        return 0
    fi
    log "ERROR: health probe timed out. Tail of log:"
    tail -n 40 "$LOG_FILE" || true
    return 1
}

do_stop() {
    local pid
    pid="$(get_pid)"
    if [ -z "$pid" ]; then
        log "not running"
        rm -f "$PID_FILE"
        return 0
    fi
    log "stopping PID=$pid ..."
    kill "$pid" 2>/dev/null || true
    local waited=0
    while [ "$waited" -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        log "force killing PID=$pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    log "stopped"
}

do_status() {
    local pid
    pid="$(get_pid)"
    if [ -n "$pid" ]; then
        log "running PID=$pid on $HOST:$PORT"
        if probe_health; then
            log "health: OK"
        else
            log "health: NOT responding at $HEALTH_URL"
        fi
        return 0
    fi
    log "not running"
    return 1
}

do_log() {
    if [ "${1:-}" = "-f" ]; then
        tail -f "$LOG_FILE"
    else
        tail -n 200 "$LOG_FILE"
    fi
}

# ── launchd install/uninstall ──────────────────────────────────────
# 让 comfyui 开机自启 + 崩溃/被停后自动拉起, 保证 fusion-operation 补货链不因
# comfyui 缺失而断 (库存耗尽 -> comfyui_image/tts 造片 -> enqueue).
# 背景: 2026-08-24 comfyui 进程不在 -> 补货链 "comfyui unavailable" stop cascade
# -> pending 持续 0 -> 4 个发布 cron 空跑 0 发布. 同 fusion-mlx/fusion-agent-studio
# 已落地的 launchd KeepAlive 模式.
_LAUNCHD_PLIST="${HOME}/Library/LaunchAgents/com.fusion-comfyui.server.plist"
_LAUNCHD_LABEL="com.fusion-comfyui.server"

install_launchd() {
    if [[ -f "${_LAUNCHD_PLIST}" ]]; then
        log "LaunchAgent already installed at ${_LAUNCHD_PLIST}"
        log "Use 'start.sh uninstall-launchd' to remove first"
        exit 0
    fi

    mkdir -p "$(dirname "${_LAUNCHD_PLIST}")"
    mkdir -p "$RUN_DIR"

    local py_bin
    if [[ -x "$VENV_PYTHON" ]]; then
        py_bin="$VENV_PYTHON"
    else
        py_bin="$(command -v python3)"
        log "ERROR: no venv python at $VENV_PYTHON, falling back to ${py_bin}"
    fi

    cat > "${_LAUNCHD_PLIST}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${_LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${py_bin}</string>
        <string>${PROJECT_DIR}/ComfyUI/main.py</string>
        <string>--listen</string>
        <string>${HOST}</string>
        <string>--port</string>
        <string>${PORT}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_FILE}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:${VENV_DIR}/bin</string>
        <key>HF_MIRROR</key>
        <string>https://hf-mirror.com</string>
        <key>HUGGINGFACE_HUB_CACHE</key>
        <string>${HOME}/.fusion-mlx/models</string>
        <key>HF_HUB_OFFLINE</key>
        <string>1</string>
        <key>TRANSFORMERS_OFFLINE</key>
        <string>1</string>
        <key>PYTORCH_ENABLE_MPS_FALLBACK</key>
        <string>1</string>
        <key>FUSION_IMAGE_TIMEOUT</key>
        <string>3600</string>
        <key>FUSION_OUTPUT_DIR</key>
        <string>${PROJECT_DIR}/output</string>
    </dict>
</dict>
</plist>
PLIST

    launchctl load "${_LAUNCHD_PLIST}" 2>/dev/null || true
    log "LaunchAgent installed and loaded: ${_LAUNCHD_PLIST}"
    log "ComfyUI will auto-start on login and restart on crash/stop"
    log "NOTE: FUSION_MLX_API_KEY intentionally NOT set in plist -> TTS http backend uses ~/.fusion-mlx/settings.json"
}

uninstall_launchd() {
    if [[ ! -f "${_LAUNCHD_PLIST}" ]]; then
        log "No LaunchAgent found at ${_LAUNCHD_PLIST}"
        exit 0
    fi

    launchctl unload "${_LAUNCHD_PLIST}" 2>/dev/null || true
    rm -f "${_LAUNCHD_PLIST}"
    log "LaunchAgent uninstalled"
}

case "${1:-status}" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    status)
        do_status
        ;;
    log)
        do_log "${2:-}"
        ;;
    restart)
        do_stop || true
        do_start
        ;;
    install-launchd)
        install_launchd
        ;;
    uninstall-launchd)
        uninstall_launchd
        ;;
    *)
        echo "Usage: $0 {start|stop|status|log [-f]|restart|install-launchd|uninstall-launchd}" >&2
        exit 2
        ;;
esac
