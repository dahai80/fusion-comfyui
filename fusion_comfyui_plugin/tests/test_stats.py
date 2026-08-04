import json
import logging

logger = logging.getLogger("fusion.test.stats")


class _FakePipeline:
    def __init__(self, stats=None, engine=None, raise_exc=None):
        self._stats = stats
        self._engine = engine
        self._raise_exc = raise_exc

    def last_denoise_stats(self):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._stats


class _FakeEngine:
    def __init__(self, stats):
        self._stats = stats

    def last_denoise_stats(self):
        return self._stats


class _BarePipeline:
    pass


class TestFusionDenoiseStatsNode:
    def test_input_types(self):
        from nodes.stats import FusionDenoiseStatsNode
        inputs = FusionDenoiseStatsNode.INPUT_TYPES()
        assert "required" in inputs
        assert "pipeline" in inputs["required"]

    def test_return_types(self):
        from nodes.stats import FusionDenoiseStatsNode
        assert FusionDenoiseStatsNode.RETURN_TYPES == ("STRING",)
        assert FusionDenoiseStatsNode.RETURN_NAMES == ("stats_json",)

    def test_get_stats_available(self):
        from nodes.stats import FusionDenoiseStatsNode
        pipeline = _FakePipeline(stats={
            "available": True,
            "enabled": True,
            "avg_accept": 0.42,
            "speedup": 1.1,
        })
        node = FusionDenoiseStatsNode()
        (out,) = node.get_stats(pipeline)
        parsed = json.loads(out)
        assert parsed["available"] is True
        assert parsed["enabled"] is True
        assert parsed["avg_accept"] == 0.42
        logger.info("test_get_stats_available: parsed=%s", parsed)

    def test_get_stats_disabled(self):
        from nodes.stats import FusionDenoiseStatsNode
        pipeline = _FakePipeline(stats={"available": False, "enabled": False})
        node = FusionDenoiseStatsNode()
        (out,) = node.get_stats(pipeline)
        parsed = json.loads(out)
        assert parsed["available"] is False
        assert parsed["enabled"] is False

    def test_get_stats_via_engine_fallback(self):
        from nodes.stats import FusionDenoiseStatsNode
        engine = _FakeEngine(stats={"available": True, "enabled": False})
        pipeline = _BarePipeline()
        pipeline._engine = engine
        node = FusionDenoiseStatsNode()
        (out,) = node.get_stats(pipeline)
        parsed = json.loads(out)
        assert parsed["available"] is True
        logger.info("test_get_stats_via_engine_fallback: used _engine attr")

    def test_get_stats_missing_method(self):
        from nodes.stats import FusionDenoiseStatsNode
        pipeline = _BarePipeline()
        node = FusionDenoiseStatsNode()
        (out,) = node.get_stats(pipeline)
        parsed = json.loads(out)
        assert parsed["available"] is False
        assert parsed["enabled"] is False
        assert parsed["reason"] == "no spec-denoise backend"

    def test_get_stats_exception_caught(self):
        from nodes.stats import FusionDenoiseStatsNode
        pipeline = _FakePipeline(raise_exc=RuntimeError("boom"))
        node = FusionDenoiseStatsNode()
        (out,) = node.get_stats(pipeline)
        parsed = json.loads(out)
        assert parsed["available"] is False
        assert parsed["enabled"] is False
        assert "boom" in parsed["error"]

    def test_get_stats_json_serializes_nonstandard(self):
        from nodes.stats import FusionDenoiseStatsNode
        pipeline = _FakePipeline(stats={
            "available": True,
            "accepted": [1, 2, 3],
            "nested": {"k": "v"},
        })
        node = FusionDenoiseStatsNode()
        (out,) = node.get_stats(pipeline)
        parsed = json.loads(out)
        assert parsed["accepted"] == [1, 2, 3]
        assert parsed["nested"]["k"] == "v"
