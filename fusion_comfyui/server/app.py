import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, UploadFile, File as FAFile, Form as FAForm
from fastapi.responses import JSONResponse

from fusion_comfyui.dag.executor import DAGExecutor
from fusion_comfyui.nodes.registry import NODE_CLASS_MAPPINGS, set_global_step_callback
from fusion_comfyui.core.config import load_config
from fusion_comfyui.core.output_store import init_store
from fusion_comfyui.server.protocol import (
    object_info,
    object_info_single,
    submit_prompt,
    history,
    history_single,
    upload_image,
    get_queue_status,
    interrupt,
    get_settings,
    get_extensions,
    system_stats,
    check_interrupt,
)
from fusion_comfyui.server.ws import websocket_handler, send_to_client
from fusion_comfyui.server.static_files import (
    init_output_dir,
    view_file,
    mount_output,
    mount_frontend,
)

logger = logging.getLogger("fusion_comfyui.server.app")

_pending: dict[str, dict] = {}
_executor: Optional[DAGExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor
    init_store()
    init_output_dir()
    _executor = DAGExecutor(NODE_CLASS_MAPPINGS)
    logger.info("fusion-comfyui server started, %d node types registered", len(NODE_CLASS_MAPPINGS))
    yield
    logger.info("fusion-comfyui server shutting down")


app = FastAPI(title="fusion-comfyui", version="0.1.0", lifespan=lifespan)


@app.get("/object_info")
async def _object_info():
    return await object_info()


@app.get("/object_info/{node_name}")
async def _object_info_single(node_name: str):
    return await object_info_single(node_name)


@app.post("/prompt")
async def _submit_prompt(prompt_data: dict):
    return await submit_prompt(prompt_data, _executor, _pending, _run_workflow)


@app.get("/history")
async def _history():
    return await history(_pending)


@app.get("/history/{prompt_id}")
async def _history_single(prompt_id: str):
    return await history_single(prompt_id, _pending)


@app.get("/view")
async def _view_file(
    filename: str = Query(...),
    subfolder: str = Query(""),
    type: str = Query("output"),
):
    return view_file(filename, subfolder, type)


@app.get("/config")
async def get_config():
    cfg = load_config()
    return cfg.to_dict()


@app.post("/upload/image")
async def _upload_image(
    image: UploadFile = FAFile(...),
    overwrite: bool = FAForm(False),
    subfolder: str = FAForm(""),
    type: str = FAForm("output"),
):
    return await upload_image(image, overwrite, subfolder, type)


@app.get("/queue")
async def _queue():
    return JSONResponse(content=get_queue_status(_pending))


@app.post("/interrupt")
async def _interrupt():
    return await interrupt()


@app.get("/settings")
async def _settings():
    return await get_settings()


@app.get("/extensions")
async def _extensions():
    return await get_extensions()


@app.get("/system_stats")
async def _system_stats():
    return await system_stats()


@app.websocket("/ws")
async def _websocket_endpoint(ws, client_id: str = ""):
    await websocket_handler(ws, client_id)


async def _run_workflow(prompt_id: str, workflow, client_id: str):
    async def node_progress_cb(current, total, node_id, node_type):
        msg = {
            "type": "progress",
            "data": {
                "prompt_id": prompt_id,
                "value": current,
                "max": total,
                "node_id": node_id,
                "node_type": node_type,
            },
        }
        await send_to_client(client_id, msg)

    async def step_progress_cb(step: int, total_steps: int):
        msg = {
            "type": "execution_progress",
            "data": {
                "prompt_id": prompt_id,
                "step": step,
                "total_steps": total_steps,
            },
        }
        await send_to_client(client_id, msg)

    set_global_step_callback(step_progress_cb)
    try:
        result = await _executor.execute(workflow, progress_cb=node_progress_cb)
        _pending[prompt_id] = {**result, "status": result.get("status", "ok")}
        await send_to_client(client_id, {
            "type": "execution_success",
            "data": {"prompt_id": prompt_id},
        })
    except Exception as e:
        logger.exception("workflow %s failed", prompt_id)
        _pending[prompt_id] = {"status": "error", "errors": [str(e)]}
        await send_to_client(client_id, {
            "type": "execution_error",
            "data": {"prompt_id": prompt_id, "error": str(e)},
        })
    finally:
        set_global_step_callback(None)


mount_output(app)
mount_frontend(app)
