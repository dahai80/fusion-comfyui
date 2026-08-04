import json
import logging

import pytest

logger = logging.getLogger("fusion.test.registry_stats")


class _FakeWrapper:
    def __init__(self, stats):
        self._stats = stats

    def last_denoise_stats(self):
        return self._stats


class TestFusionDenoiseStats:
    def test_input_types(self):
        from fusion_comfyui.nodes.registry import FusionDenoiseStats
        inputs = FusionDenoiseStats.INPUT_TYPES()
        assert "required" in inputs
        assert "model" in inputs["required"]

    def test_return_types_and_names(self):
        from fusion_comfyui.nodes.registry import FusionDenoiseStats
        assert FusionDenoiseStats.RETURN_TYPES == ("STRING",)
        assert FusionDenoiseStats.RETURN_NAMES == ("stats_json",)
        assert FusionDenoiseStats.CATEGORY == "fusion-mlx/debug"

    @pytest.mark.asyncio
    async def test_execute_returns_json(self):
        from fusion_comfyui.nodes.registry import FusionDenoiseStats
        model = _FakeWrapper({
            "available": True,
            "enabled": True,
            "avg_accept": 0.55,
            "speedup": 1.2,
            "accepted": [3, 2, 1],
        })
        node = FusionDenoiseStats()
        (out,) = await node.execute(model)
        parsed = json.loads(out)
        assert parsed["available"] is True
        assert parsed["enabled"] is True
        assert parsed["avg_accept"] == 0.55
        logger.info("test_execute_returns_json: parsed=%s", parsed)

    @pytest.mark.asyncio
    async def test_execute_disabled_when_not_started(self):
        from fusion_comfyui.nodes.registry import FusionDenoiseStats
        model = _FakeWrapper({"available": False, "enabled": False})
        node = FusionDenoiseStats()
        (out,) = await node.execute(model)
        parsed = json.loads(out)
        assert parsed["available"] is False
        assert parsed["enabled"] is False

    @pytest.mark.asyncio
    async def test_execute_serializes_nested_and_lists(self):
        from fusion_comfyui.nodes.registry import FusionDenoiseStats
        model = _FakeWrapper({
            "available": True,
            "accepted": [1, 2, 3],
            "nested": {"k": "v"},
        })
        node = FusionDenoiseStats()
        (out,) = await node.execute(model)
        parsed = json.loads(out)
        assert parsed["accepted"] == [1, 2, 3]
        assert parsed["nested"]["k"] == "v"
