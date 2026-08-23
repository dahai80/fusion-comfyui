import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from fusion_comfyui.server.protocol import (
    object_info, submit_prompt, history, history_single,
    get_queue_status, interrupt, check_interrupt, request_interrupt,
    get_settings, get_extensions,
)


class TestObjectInfo:
    @pytest.mark.asyncio
    async def test_returns_node_info(self):
        resp = await object_info()
        assert resp.body is not None

    @pytest.mark.asyncio
    async def test_contains_known_node(self):
        resp = await object_info()
        data = json.loads(resp.body)
        assert "FusionModelLoader" in data or "FusionKSampler" in data


class TestSubmitPrompt:
    @pytest.mark.asyncio
    async def test_queues_prompt(self):
        executor = MagicMock()
        pending = {}
        run_fn = AsyncMock()
        data = {
            "prompt": {
                "1": {"class_type": "FusionModelLoader", "inputs": {}},
            },
        }
        resp = await submit_prompt(data, executor, pending, run_fn)
        result = json.loads(resp.body)
        assert result["status"] == "queued"
        assert "prompt_id" in result

    @pytest.mark.asyncio
    async def test_custom_prompt_id(self):
        executor = MagicMock()
        pending = {}
        run_fn = AsyncMock()
        data = {
            "prompt_id": "my-custom-id",
            "prompt": {"1": {"class_type": "FusionModelLoader", "inputs": {}}},
        }
        resp = await submit_prompt(data, executor, pending, run_fn)
        result = json.loads(resp.body)
        assert result["prompt_id"] == "my-custom-id"


class TestHistory:
    @pytest.mark.asyncio
    async def test_empty_history(self):
        resp = await history({})
        data = json.loads(resp.body)
        assert data == {}

    @pytest.mark.asyncio
    async def test_with_entries(self):
        pending = {"abc": {"status": "done"}}
        resp = await history(pending)
        data = json.loads(resp.body)
        assert "abc" in data
        rec = data["abc"]
        assert "outputs" in rec
        assert isinstance(rec["status"], dict)
        assert rec["status"]["status_str"] in ("success", "error", "done", "unknown", "queued", "running")

    @pytest.mark.asyncio
    async def test_success_record_has_completed_status(self):
        pending = {"abc": {"status": "ok", "outputs": {"1": {"image": "x.png"}}}}
        resp = await history_single("abc", pending)
        data = json.loads(resp.body)
        st = data["abc"]["status"]
        assert st["status_str"] == "success"
        assert st["completed"] is True
        assert data["abc"]["outputs"]["1"]["image"] == "x.png"

    @pytest.mark.asyncio
    async def test_error_record_has_error_messages(self):
        pending = {"abc": {"status": "error", "errors": ["boom"]}}
        resp = await history_single("abc", pending)
        data = json.loads(resp.body)
        st = data["abc"]["status"]
        assert st["status_str"] == "error"
        assert st["completed"] is True
        assert ["execution_error", "boom"] in st["messages"]

    @pytest.mark.asyncio
    async def test_history_single_found(self):
        pending = {"abc": {"status": "done"}}
        resp = await history_single("abc", pending)
        data = json.loads(resp.body)
        assert "abc" in data

    @pytest.mark.asyncio
    async def test_history_single_not_found(self):
        with pytest.raises(Exception):
            await history_single("missing", {})


class TestQueueStatus:
    def test_empty_queue(self):
        result = get_queue_status({})
        assert result["queue_running"] == []
        assert result["queue_pending"] == []

    def test_running_and_done(self):
        pending = {
            "r1": {"status": "running"},
            "d1": {"status": "ok"},
            "r2": {"status": "running"},
        }
        result = get_queue_status(pending)
        assert len(result["queue_running"]) == 2
        assert len(result["queue_pending"]) == 0

    def test_queued_status_classified_to_pending(self):
        pending = {
            "q1": {"status": "queued"},
            "r1": {"status": "running"},
        }
        result = get_queue_status(pending)
        assert len(result["queue_running"]) == 1
        assert len(result["queue_pending"]) == 1
        assert result["queue_pending"][0]["prompt_id"] == "q1"


class TestInterrupt:
    @pytest.mark.asyncio
    async def test_interrupt_sets_flag(self):
        resp = await interrupt()
        data = json.loads(resp.body)
        assert data["status"] == "ok"
        # interrupt() 置 _interrupt_flag=True 但本测试不消费；
        # 残留 flag 会污染后续 DAG executor（check_interrupt 读到残留 → interrupted）。
        # 消费掉，保持全局状态干净。
        assert check_interrupt() is True

    def test_check_interrupt_consumes_flag(self):
        request_interrupt()
        assert check_interrupt() is True
        assert check_interrupt() is False


class TestSettings:
    @pytest.mark.asyncio
    async def test_settings_content(self):
        resp = await get_settings()
        data = json.loads(resp.body)
        assert data["fusion-comfyui"] is True
        assert data["mlx_backend"] is True


class TestExtensions:
    @pytest.mark.asyncio
    async def test_extensions_empty(self):
        resp = await get_extensions()
        data = json.loads(resp.body)
        assert isinstance(data, list)
