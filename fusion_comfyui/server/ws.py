import json
import logging
import uuid

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("fusion_comfyui.server.ws")

_clients: dict[str, WebSocket] = {}


def register_client(ws: WebSocket, client_id: str | None = None) -> str:
    if not client_id:
        client_id = str(uuid.uuid4())
    _clients[client_id] = ws
    return client_id


def unregister_client(client_id: str):
    _clients.pop(client_id, None)


async def send_to_client(client_id: str, msg: dict):
    ws = _clients.get(client_id)
    if ws:
        try:
            await ws.send_json(msg)
        except Exception:
            logger.warning("failed to send ws msg to %s", client_id)


async def websocket_handler(ws: WebSocket, client_id: str = ""):
    await ws.accept()
    cid = register_client(ws, client_id or None)
    await ws.send_json({
        "type": "status",
        "data": {
            "status": {"exec_info": {"queue_remaining": 0}},
            "sid": cid,
        },
    })
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        logger.info("ws client disconnected: %s", cid)
    finally:
        unregister_client(cid)
