import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("fusion_comfyui.core.config")

_DEFAULTS = {
    "FUSION_SPECULATIVE_DENOISING": "0",
    "FUSION_SPEC_DRAFT_STEPS": "2",
    "FUSION_SPEC_DRAFT_MODEL": "",
    "FUSION_RADIX_CACHE_ENABLED": "0",
    "FUSION_RADIX_CACHE_MAX_MB": "512",
    "FUSION_NVFP4_ENABLED": "0",
    "FUSION_NVFP4_THRESHOLD_GB": "8",
}


@dataclass
class Phase3Config:
    speculative_denoising: bool = False
    spec_draft_steps: int = 2
    spec_draft_model: str = ""
    radix_cache_enabled: bool = False
    radix_cache_max_mb: int = 512
    nvfp4_enabled: bool = False
    nvfp4_threshold_gb: int = 8

    def to_dict(self) -> dict:
        return {
            "speculative_denoising": self.speculative_denoising,
            "spec_draft_steps": self.spec_draft_steps,
            "spec_draft_model": self.spec_draft_model,
            "radix_cache_enabled": self.radix_cache_enabled,
            "radix_cache_max_mb": self.radix_cache_max_mb,
            "nvfp4_enabled": self.nvfp4_enabled,
            "nvfp4_threshold_gb": self.nvfp4_threshold_gb,
        }


def load_config() -> Phase3Config:
    cfg = Phase3Config()
    for env_key, default in _DEFAULTS.items():
        val = os.environ.get(env_key, default)
        if env_key == "FUSION_SPECULATIVE_DENOISING":
            cfg.speculative_denoising = val in ("1", "true", "yes")
        elif env_key == "FUSION_SPEC_DRAFT_STEPS":
            cfg.spec_draft_steps = int(val)
        elif env_key == "FUSION_SPEC_DRAFT_MODEL":
            cfg.spec_draft_model = val
        elif env_key == "FUSION_RADIX_CACHE_ENABLED":
            cfg.radix_cache_enabled = val in ("1", "true", "yes")
        elif env_key == "FUSION_RADIX_CACHE_MAX_MB":
            cfg.radix_cache_max_mb = int(val)
        elif env_key == "FUSION_NVFP4_ENABLED":
            cfg.nvfp4_enabled = val in ("1", "true", "yes")
        elif env_key == "FUSION_NVFP4_THRESHOLD_GB":
            cfg.nvfp4_threshold_gb = int(val)
    logger.info("Phase3Config: %s", cfg.to_dict())
    return cfg


class _RadixNode:
    __slots__ = ("children", "value", "size_bytes", "last_access")

    def __init__(self):
        self.children: dict[str, "_RadixNode"] = {}
        self.value: bytes | None = None
        self.size_bytes: int = 0
        self.last_access: int = 0


class RadixCache:
    def __init__(self, max_mb: int = 512):
        self.max_bytes = max_mb * 1024 * 1024
        self._root = _RadixNode()
        self._hits = 0
        self._misses = 0
        self._total_bytes = 0
        self._clock = 0
        self._leaf_count = 0

    def get(self, key: str) -> bytes | None:
        self._clock += 1
        node = self._root
        remainder = key
        while remainder:
            matched = False
            for prefix, child in node.children.items():
                common = self._common_prefix(remainder, prefix)
                if not common:
                    continue
                if common == prefix:
                    node = child
                    remainder = remainder[len(prefix):]
                    matched = True
                    break
                if common == remainder:
                    partial_key = prefix[len(common):]
                    if partial_key in node.children.get(remainder, _RadixNode()).children:
                        remainder = ""
                        matched = True
                        break
                return None
            if not matched:
                self._misses += 1
                return None
        if node.value is not None:
            node.last_access = self._clock
            self._hits += 1
            logger.debug("radix cache hit: %s (%d bytes)", key[:16], node.size_bytes)
            return node.value
        self._misses += 1
        return None

    def put(self, key: str, value: bytes):
        self._clock += 1
        node = self._root
        remainder = key
        while remainder:
            matched = False
            for prefix, child in list(node.children.items()):
                common = self._common_prefix(remainder, prefix)
                if not common:
                    continue
                if common == prefix:
                    node = child
                    remainder = remainder[len(prefix):]
                    matched = True
                    break
                split_node = _RadixNode()
                old_suffix = prefix[len(common):]
                new_suffix = remainder[len(common):]
                split_node.children[old_suffix] = child
                del node.children[prefix]
                node.children[common] = split_node
                if new_suffix:
                    leaf = _RadixNode()
                    split_node.children[new_suffix] = leaf
                    node = leaf
                else:
                    node = split_node
                remainder = ""
                matched = True
                break
            if not matched:
                leaf = _RadixNode()
                node.children[remainder] = leaf
                node = leaf
                remainder = ""
        if node.value is None:
            self._leaf_count += 1
        else:
            self._total_bytes -= node.size_bytes
        node.value = value
        node.size_bytes = len(value)
        node.last_access = self._clock
        self._total_bytes += node.size_bytes
        self._evict_if_needed()

    def _evict_if_needed(self):
        while self._total_bytes > self.max_bytes and self._leaf_count > 1:
            victim = self._find_lru_leaf(self._root, None, "")
            if victim is None:
                break
            parent, edge_key, lru_node = victim
            self._total_bytes -= lru_node.size_bytes
            self._leaf_count -= 1
            del parent.children[edge_key]
            self._cleanup_chains(self._root, None, "")
            logger.debug("radix cache evicted (%d bytes, %d leaves remain)", lru_node.size_bytes, self._leaf_count)

    def _find_lru_leaf(self, node, parent, edge_key):
        if not node.children and node.value is not None:
            return (parent, edge_key, node)
        best = None
        best_access = float("inf")
        for prefix, child in node.children.items():
            result = self._find_lru_leaf(child, node, prefix)
            if result is not None and result[2].last_access < best_access:
                best = result
                best_access = result[2].last_access
        return best

    def _cleanup_chains(self, node, parent, edge_key):
        for prefix, child in list(node.children.items()):
            self._cleanup_chains(child, node, prefix)
        if parent is not None and len(node.children) == 1 and node.value is None:
            only_prefix = next(iter(node.children))
            only_child = node.children[only_prefix]
            merged = edge_key + only_prefix
            parent.children[merged] = only_child
            del parent.children[edge_key]

    @staticmethod
    def _common_prefix(a: str, b: str) -> str:
        i = 0
        limit = min(len(a), len(b))
        while i < limit and a[i] == b[i]:
            i += 1
        return a[:i]

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": self._leaf_count,
            "total_bytes": self._total_bytes,
            "max_bytes": self.max_bytes,
        }

    def clear(self):
        self._root = _RadixNode()
        self._hits = 0
        self._misses = 0
        self._total_bytes = 0
        self._leaf_count = 0
