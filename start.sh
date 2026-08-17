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
# Image-gen per-call timeout (fusion-mlx #481). SD1.5 1024 img2img (hires-fix
# 2nd pass) can exceed the 600s default on Apple Silicon.
export FUSION_IMAGE_TIMEOUT="${FUSION_IMAGE_TIMEOUT:-3600}"

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
    *)
        echo "Usage: $0 {start|stop|status|log [-f]|restart}" >&2
        exit 2
        ;;
esac
