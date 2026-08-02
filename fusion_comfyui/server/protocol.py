import logging
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from fusion_comfyui.dag.executor import parse_workflow
from fusion_comfyui.nodes.registry import build_node_info, set_global_step_callback

logger = logging.getLogger("fusion_comfyui.server.protocol")


async def object_info():
    info = build_node_info()
    return JSONResponse(content=info)


async def object_info_single(node_name: str):
    info = build_node_info()
    if node_name not in info:
        raise HTTPException(status_code=404, detail=f"node type '{node_name}' not found")
    return JSONResponse(content={node_name: info[node_name]})


async def submit_prompt(prompt_data: dict, executor, pending: dict, run_workflow_fn):
    client_id = prompt_data.get("client_id", "")
    prompt = prompt_data.get("prompt", prompt_data)
    prompt_id = prompt_data.get("prompt_id") or str(uuid.uuid4())

    workflow = parse_workflow(prompt)
    pending[prompt_id] = {"status": "running", "workflow": workflow}

    import asyncio
    asyncio.create_task(run_workflow_fn(prompt_id, workflow, client_id))

    return JSONResponse(content={"prompt_id": prompt_id, "status": "queued", "number": len(pending)})


async def history(pending: dict):
    return JSONResponse(content=pending)


async def history_single(prompt_id: str, pending: dict):
    if prompt_id not in pending:
        raise HTTPException(status_code=404, detail="prompt not found")
    return JSONResponse(content={prompt_id: pending[prompt_id]})


async def upload_image(
    image: UploadFile = File(...),
    overwrite: bool = Form(False),
    subfolder: str = Form(""),
    type: str = Form("output"),
):
    from fusion_comfyui.core.output_store import get_store_dir, save_bytes
    allowed_ext = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    suffix = Path(image.filename or "upload.png").suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"unsupported image format: {suffix}")
    filename = image.filename or f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    if not overwrite:
        base = get_store_dir() / subfolder if subfolder else get_store_dir()
        base.mkdir(parents=True, exist_ok=True)
        target = base / filename
        if target.exists():
            stem = Path(filename).stem
            filename = f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    data = await image.read()
    saved = save_bytes(data, filename, subfolder=subfolder)
    logger.info("uploaded image: %s (%d bytes)", saved, len(data))
    return JSONResponse(content={
        "name": filename,
        "subfolder": subfolder,
        "type": type,
    })


def get_queue_status(pending: dict) -> dict:
    running = []
    pending_queue = []
    done = []
    for pid, info in pending.items():
        status = info.get("status", "unknown")
        if status == "running":
            running.append(pid)
        elif status == "queued":
            pending_queue.append(pid)
        else:
            done.append(pid)
    return {
        "queue_running": [{"prompt_id": pid} for pid in running],
        "queue_pending": [{"prompt_id": pid} for pid in pending_queue],
    }


_interrupt_flag = False


def request_interrupt():
    global _interrupt_flag
    _interrupt_flag = True
    logger.info("interrupt requested")


def check_interrupt() -> bool:
    global _interrupt_flag
    if _interrupt_flag:
        _interrupt_flag = False
        return True
    return False


async def interrupt():
    request_interrupt()
    return JSONResponse(content={"status": "ok"})


async def get_settings():
    return JSONResponse(content={
        "fusion-comfyui": True,
        "mlx_backend": True,
        "torch_backend": False,
    })


async def get_extensions():
    return JSONResponse(content=[])


async def system_stats():
    import mlx.core as mx
    try:
        active_mb = round(mx.metal.get_active_memory() / 1024 / 1024, 1)
        peak_mb = round(mx.metal.get_peak_memory() / 1024 / 1024, 1)
    except Exception:
        active_mb = 0
        peak_mb = 0
    return JSONResponse(content={
        "system": {
            "active_memory_mb": active_mb,
            "peak_memory_mb": peak_mb,
        },
        "devices": [{
            "name": "Apple Silicon (MLX)",
            "type": "mlx",
            "vram_total_mb": 0,
            "vram_free_mb": 0,
        }],
    })
