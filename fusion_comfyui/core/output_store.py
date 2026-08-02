import logging
import os
from pathlib import Path

logger = logging.getLogger("fusion_comfyui.core.output_store")

_store_dir: Path = Path(os.environ.get("FUSION_OUTPUT_DIR", "output"))


def init_store():
    _store_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output store initialized at %s", _store_dir)


def get_store_dir() -> Path:
    return _store_dir


def resolve_path(filename: str, subfolder: str = "") -> Path | None:
    base = _store_dir / subfolder if subfolder else _store_dir
    fpath = base / filename
    if fpath.exists():
        return fpath
    return None


def list_outputs(subfolder: str = "") -> list[str]:
    base = _store_dir / subfolder if subfolder else _store_dir
    if not base.exists():
        return []
    return [f.name for f in base.iterdir() if f.is_file()]


def save_bytes(data: bytes, filename: str, subfolder: str = "") -> Path:
    base = _store_dir / subfolder if subfolder else _store_dir
    base.mkdir(parents=True, exist_ok=True)
    fpath = base / filename
    fpath.write_bytes(data)
    logger.info("saved %s (%d bytes)", fpath, len(data))
    return fpath
