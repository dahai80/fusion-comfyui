import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, UploadFile, File as FAFile, Form as FAForm, WebSocket
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


@app.get("/queue-api")
async def _queue_api():
    q = get_queue_status(_pending)
    remaining = len(q.get("queue_pending", [])) + len(q.get("queue_running", []))
    return JSONResponse(content={"status": {"exec_info": {"queue_remaining": remaining}}})


@app.post("/interrupt")
async def _interrupt():
    return await interrupt()


@app.get("/settings")
async def _settings():
    return await get_settings()


@app.post("/settings/{key}")
async def _settings_set_key(key: str, body=None):
    logger.info("settings set key=%s", key)
    return JSONResponse(content={"status": "ok"})


@app.get("/extensions")
async def _extensions():
    return await get_extensions()


@app.get("/system_stats")
async def _system_stats():
    return await system_stats()


@app.get("/users")
async def _users():
    return JSONResponse(content={"default": {"name": "default", "id": "default"}})


@app.get("/userdata")
async def _userdata(dir: str = Query(""), recurse: bool = Query(False)):
    logger.info("userdata dir=%s recurse=%s (no persisted userdata)", dir, recurse)
    return JSONResponse(content=[])


@app.get("/i18n")
async def _i18n():
    return JSONResponse(content={})


@app.get("/global_subgraphs")
async def _global_subgraphs():
    return JSONResponse(content={})


@app.get("/jobs")
async def _jobs(status: str = Query(""), limit: int = Query(200), offset: int = Query(0)):
    logger.info("jobs status=%s limit=%d offset=%d (no persisted jobs)", status, limit, offset)
    return JSONResponse(content={"jobs": [], "pagination": {"total": 0, "offset": offset, "limit": limit, "has_more": False}})


@app.get("/jobs/{job_id}")
async def _jobs_single(job_id: str):
    logger.info("jobs detail job_id=%s (no persisted jobs)", job_id)
    raise HTTPException(status_code=404, detail="job not found")


@app.get("/embeddings")
async def _embeddings():
    return JSONResponse(content=[])


@app.get("/experiment/models")
async def _experiment_models():
    return JSONResponse(content=[])


@app.get("/workflow_templates")
async def _workflow_templates():
    return JSONResponse(content=[])


@app.get("/view_metadata/{filename}")
async def _view_metadata(filename: str):
    logger.info("view_metadata filename=%s (no metadata)", filename)
    return JSONResponse(content={})


@app.get("/userdata/{filename:path}")
async def _userdata_file(filename: str):
    logger.info("userdata file get filename=%s (no persisted userdata)", filename)
    if filename.endswith(".json"):
        return JSONResponse(content=[])
    raise HTTPException(status_code=404, detail="userdata file not found")


@app.post("/free")
async def _free():
    logger.info("free memory requested (no-op)")
    return JSONResponse(content={"status": "ok"})


@app.get("/global_subgraphs/{subgraph_id}")
async def _global_subgraphs_single(subgraph_id: str):
    logger.info("global_subgraphs detail id=%s (none)", subgraph_id)
    raise HTTPException(status_code=404, detail="subgraph not found")


@app.websocket("/ws")
async def _websocket_endpoint(ws: WebSocket, client_id: str = ""):
    logger.debug("ws endpoint invoked client_id=%s", client_id)
    await websocket_handler(ws, client_id)


# New Vue frontend calls /api/* prefixed routes; mount every API handler under
# /api too so the frontend initializes (else it 404s and the toolbar never renders).
_api_router = APIRouter()
_api_router.add_api_route("/object_info", _object_info, methods=["GET"])
_api_router.add_api_route("/object_info/{node_name}", _object_info_single, methods=["GET"])
_api_router.add_api_route("/prompt", _submit_prompt, methods=["POST"])
_api_router.add_api_route("/history", _history, methods=["GET"])
_api_router.add_api_route("/history/{prompt_id}", _history_single, methods=["GET"])
_api_router.add_api_route("/view", _view_file, methods=["GET"])
_api_router.add_api_route("/queue", _queue_api, methods=["GET"])
_api_router.add_api_route("/interrupt", _interrupt, methods=["POST"])
_api_router.add_api_route("/settings", _settings, methods=["GET"])
_api_router.add_api_route("/settings/{key}", _settings_set_key, methods=["POST"])
_api_router.add_api_route("/extensions", _extensions, methods=["GET"])
_api_router.add_api_route("/system_stats", _system_stats, methods=["GET"])
_api_router.add_api_route("/users", _users, methods=["GET"])
_api_router.add_api_route("/userdata", _userdata, methods=["GET"])
_api_router.add_api_route("/i18n", _i18n, methods=["GET"])
_api_router.add_api_route("/global_subgraphs", _global_subgraphs, methods=["GET"])
_api_router.add_api_route("/jobs", _jobs, methods=["GET"])
_api_router.add_api_route("/jobs/{job_id}", _jobs_single, methods=["GET"])
_api_router.add_api_route("/config", get_config, methods=["GET"])
_api_router.add_api_route("/upload/image", _upload_image, methods=["POST"])
_api_router.add_api_route("/embeddings", _embeddings, methods=["GET"])
_api_router.add_api_route("/experiment/models", _experiment_models, methods=["GET"])
_api_router.add_api_route("/workflow_templates", _workflow_templates, methods=["GET"])
_api_router.add_api_route("/view_metadata/{filename}", _view_metadata, methods=["GET"])
_api_router.add_api_route("/userdata/{filename:path}", _userdata_file, methods=["GET"])
_api_router.add_api_route("/free", _free, methods=["POST"])
_api_router.add_api_route("/global_subgraphs/{subgraph_id}", _global_subgraphs_single, methods=["GET"])
app.include_router(_api_router, prefix="/api")


@app.get("/templates/{name:path}")
async def _core_templates(name: str):
    logger.info("core template requested name=%s (no bundled templates)", name)
    if name.endswith(".json"):
        return JSONResponse(content=[])
    raise HTTPException(status_code=404, detail="template not found")


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

    async def node_event_cb(phase, node_id, node_type, current, total):
        if phase == "start":
            await send_to_client(client_id, {
                "type": "executing",
                "data": {"node": node_id, "prompt_id": prompt_id},
            })
        else:
            await send_to_client(client_id, {
                "type": "executed",
                "data": {"node": node_id, "prompt_id": prompt_id},
            })

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
    _pending[prompt_id] = {**_pending.get(prompt_id, {}), "status": "running"}
    try:
        result = await _executor.execute(
            workflow, progress_cb=node_progress_cb, node_event_cb=node_event_cb,
        )
        _pending[prompt_id] = {**result, "status": result.get("status", "ok")}
        await send_to_client(client_id, {
            "type": "executing",
            "data": {"node": None, "prompt_id": prompt_id},
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
