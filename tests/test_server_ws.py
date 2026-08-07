import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from fusion_comfyui.server.app import app
from fusion_comfyui.server.ws import register_client, unregister_client, send_to_client


class TestWSClientManagement:
    def test_register_client(self):
        ws = MagicMock()
        cid = register_client(ws, "test-id")
        assert cid == "test-id"
        unregister_client("test-id")

    def test_register_client_auto_id(self):
        ws = MagicMock()
        cid = register_client(ws)
        assert cid
        unregister_client(cid)

    def test_unregister_nonexistent(self):
        unregister_client("nonexistent")

    @pytest.mark.asyncio
    async def test_send_to_client(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        register_client(ws, "send-test")
        await send_to_client("send-test", {"type": "test"})
        ws.send_json.assert_called_once_with({"type": "test"})
        unregister_client("send-test")

    @pytest.mark.asyncio
    async def test_send_to_missing_client(self):
        await send_to_client("no-such-client", {"type": "test"})


class TestWSEndpointRouting:
    async def _drive_ws(self, query: str):
        scope = {
            "type": "websocket", "asgi": {"version": "3.0"},
            "http_version": "1.1", "client": ("127.0.0.1", 55555),
            "server": ("127.0.0.1", 11443), "scheme": "ws",
            "path": "/ws", "raw_path": b"/ws", "query_string": query.encode(),
            "headers": [], "subprotocols": [], "state": {}, "extensions": {},
        }
        events = []
        connected = False

        async def receive():
            nonlocal connected
            if not connected:
                connected = True
                return {"type": "websocket.connect"}
            await asyncio.sleep(0)
            return {"type": "websocket.disconnect", "code": 1000}

        async def send(msg):
            events.append(msg.get("type"))

        await app(scope, receive, send)
        return events

    @pytest.mark.asyncio
    async def test_ws_endpoint_accepts_connection(self):
        events = await self._drive_ws("client_id=probe-regression")
        assert "websocket.accept" in events, f"ws not accepted, events={events}"

    @pytest.mark.asyncio
    async def test_ws_endpoint_sends_initial_status(self):
        events = await self._drive_ws("client_id=probe-status")
        assert "websocket.accept" in events
        assert "websocket.send" in events, f"no status sent, events={events}"
