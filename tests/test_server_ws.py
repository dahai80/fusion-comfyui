import pytest
from unittest.mock import AsyncMock, MagicMock

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
