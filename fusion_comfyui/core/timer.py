import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("fusion_comfyui.core.timer")


@dataclass
class TimingRecord:
    node_name: str
    phase: str
    elapsed_s: float
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        mem_delta = self.memory_after_mb - self.memory_before_mb
        return (
            f"[TIMER] {self.node_name}/{self.phase}: "
            f"{self.elapsed_s:.3f}s "
            f"mem={self.memory_after_mb:.0f}MB "
            f"delta={mem_delta:+.0f}MB"
        )


class NodeTimer:
    _records: list[TimingRecord] = []
    _enabled: bool = True

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def clear(cls):
        cls._records.clear()

    @classmethod
    def records(cls) -> list[TimingRecord]:
        return list(cls._records)

    @classmethod
    def summary(cls) -> str:
        if not cls._records:
            return "[TIMER] No records"
        lines = ["[TIMER] Summary:", "-" * 72]
        by_node: dict[str, list[TimingRecord]] = {}
        for r in cls._records:
            by_node.setdefault(r.node_name, []).append(r)
        total = 0.0
        for node, recs in by_node.items():
            node_total = sum(r.elapsed_s for r in recs)
            total += node_total
            lines.append(f"  {node}: {node_total:.3f}s total")
            for r in recs:
                lines.append(f"    {r.phase}: {r.elapsed_s:.3f}s")
        lines.append("-" * 72)
        lines.append(f"  TOTAL: {total:.3f}s")
        return "\n".join(lines)

    @classmethod
    def export_csv(cls) -> str:
        import io
        buf = io.StringIO()
        buf.write("node_name,phase,elapsed_s,memory_before_mb,memory_after_mb\n")
        for r in cls._records:
            buf.write(
                f"{r.node_name},{r.phase},{r.elapsed_s:.3f},"
                f"{r.memory_before_mb:.0f},{r.memory_after_mb:.0f}\n"
            )
        return buf.getvalue()

    @classmethod
    @asynccontextmanager
    async def timed(cls, node_name: str, phase: str, **metadata):
        if not cls._enabled:
            yield
            return
        mem_before = cls._get_memory_mb()
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            mem_after = cls._get_memory_mb()
            record = TimingRecord(
                node_name=node_name,
                phase=phase,
                elapsed_s=elapsed,
                memory_before_mb=mem_before,
                memory_after_mb=mem_after,
                metadata=metadata,
            )
            cls._records.append(record)
            logger.info(str(record))

    @staticmethod
    def _get_memory_mb() -> float:
        try:
            import mlx.core as mx
            return mx.metal.get_active_memory() / 1024 / 1024
        except Exception:
            return 0.0
