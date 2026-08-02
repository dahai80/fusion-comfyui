import argparse
import logging
import os
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="fusion-comfyui: pure MLX ComfyUI server")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve"], help="Command to run")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8188, help="Bind port")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("fusion_comfyui")
    logger.info("fusion-comfyui starting on %s:%d", args.host, args.port)

    os.environ["FUSION_OUTPUT_DIR"] = args.output_dir

    uvicorn.run(
        "fusion_comfyui.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
