
from fusion_comfyui.core.config import Phase3Config, RadixCache, load_config


class TestPhase3Config:
    def test_defaults(self):
        cfg = Phase3Config()
        assert cfg.spec_denoise_enabled is False
        assert cfg.spec_k == 4
        assert cfg.spec_epsilon == 0.1
        assert cfg.spec_draft_blocks == ""
        assert cfg.spec_eval_steps == 1
        assert cfg.radix_cache_enabled is False
        assert cfg.radix_cache_max_mb == 512
        assert cfg.nvfp4_enabled is False

    def test_to_dict(self):
        cfg = Phase3Config()
        d = cfg.to_dict()
        assert "spec_denoise_enabled" in d
        assert "radix_cache_max_mb" in d
        assert d["spec_k"] == 4
        assert d["spec_epsilon"] == 0.1
        assert d["spec_eval_steps"] == 1

    def test_load_config_defaults(self, monkeypatch):
        for key in [
            "FUSION_SPECULATIVE_DENOISE", "FUSION_SPEC_K",
            "FUSION_SPEC_EPSILON", "FUSION_SPEC_DRAFT_BLOCKS",
            "FUSION_SPEC_EVAL_STEPS", "FUSION_RADIX_CACHE_ENABLED",
            "FUSION_RADIX_CACHE_MAX_MB", "FUSION_NVFP4_ENABLED",
            "FUSION_NVFP4_THRESHOLD_GB",
        ]:
            monkeypatch.delenv(key, raising=False)
        cfg = load_config()
        assert cfg.spec_denoise_enabled is False
        assert cfg.spec_k == 4
        assert cfg.spec_epsilon == 0.1
        assert cfg.radix_cache_max_mb == 512

    def test_load_config_env_override(self, monkeypatch):
        monkeypatch.setenv("FUSION_RADIX_CACHE_ENABLED", "1")
        monkeypatch.setenv("FUSION_RADIX_CACHE_MAX_MB", "1024")
        monkeypatch.setenv("FUSION_NVFP4_ENABLED", "true")
        monkeypatch.setenv("FUSION_SPECULATIVE_DENOISE", "1")
        monkeypatch.setenv("FUSION_SPEC_K", "8")
        monkeypatch.setenv("FUSION_SPEC_EPSILON", "0.25")
        cfg = load_config()
        assert cfg.radix_cache_enabled is True
        assert cfg.radix_cache_max_mb == 1024
        assert cfg.nvfp4_enabled is True
        assert cfg.spec_denoise_enabled is True
        assert cfg.spec_k == 8
        assert cfg.spec_epsilon == 0.25


class TestRadixCache:
    def test_put_get(self):
        cache = RadixCache(max_mb=1)
        cache.put("hello", b"world")
        assert cache.get("hello") == b"world"

    def test_miss(self):
        cache = RadixCache(max_mb=1)
        assert cache.get("nonexistent") is None

    def test_overwrite(self):
        cache = RadixCache(max_mb=1)
        cache.put("key", b"v1")
        cache.put("key", b"v2")
        assert cache.get("key") == b"v2"

    def test_prefix_sharing(self):
        cache = RadixCache(max_mb=1)
        cache.put("comfy/nodes/1", b"a")
        cache.put("comfy/nodes/2", b"b")
        cache.put("comfy/edges/1", b"c")
        assert cache.get("comfy/nodes/1") == b"a"
        assert cache.get("comfy/nodes/2") == b"b"
        assert cache.get("comfy/edges/1") == b"c"

    def test_stats(self):
        cache = RadixCache(max_mb=1)
        cache.put("k1", b"v1")
        cache.get("k1")
        cache.get("miss")
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["entries"] == 1

    def test_eviction(self):
        cache = RadixCache(max_mb=0)
        cache.put("a", b"x" * 100)
        cache.put("b", b"y" * 100)
        s = cache.stats()
        assert s["total_bytes"] <= 100

    def test_clear(self):
        cache = RadixCache(max_mb=1)
        cache.put("k", b"v")
        cache.clear()
        assert cache.get("k") is None
        assert cache.stats()["entries"] == 0

    def test_common_prefix_split(self):
        cache = RadixCache(max_mb=1)
        cache.put("abc", b"1")
        cache.put("abd", b"2")
        assert cache.get("abc") == b"1"
        assert cache.get("abd") == b"2"
