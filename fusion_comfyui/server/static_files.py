import logging
import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("fusion_comfyui.server.static_files")

_output_dir: Path = Path(os.environ.get("FUSION_OUTPUT_DIR", "output"))
_frontend_dir: Path = Path(os.environ.get("FUSION_FRONTEND_DIR", ""))


def init_output_dir():
    _output_dir.mkdir(parents=True, exist_ok=True)


def get_output_dir() -> Path:
    return _output_dir


def get_frontend_dir() -> Path:
    if _frontend_dir.exists():
        return _frontend_dir
    bundled = Path(__file__).parent.parent / "frontend"
    if bundled.exists():
        return bundled
    return Path("")


def view_file(filename: str, subfolder: str = "", ttype: str = "output"):
    base = _output_dir / subfolder if subfolder else _output_dir
    fpath = base / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(fpath))


def mount_output(app):
    if _output_dir.exists():
        app.mount("/output", StaticFiles(directory=str(_output_dir)), name="output")


def mount_frontend(app):
    fdir = get_frontend_dir()
    if fdir.exists():
        app.mount("/", StaticFiles(directory=str(fdir), html=True), name="frontend")
        logger.info("frontend mounted from %s", fdir)
