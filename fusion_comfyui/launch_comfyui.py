import logging
import os
import runpy
import sys

logger = logging.getLogger("fusion_comfyui.launch")

_DEFAULT_PORT = "11445"


def _port_from_argv(argv):
    port = _DEFAULT_PORT
    for i, arg in enumerate(argv):
        if arg == "--port" and i + 1 < len(argv):
            port = argv[i + 1]
        elif arg.startswith("--port="):
            port = arg.split("=", 1)[1]
    return port


def _set_process_title(argv):
    port = _port_from_argv(argv)
    title = f"fusion-comfyui serve :{port}"
    try:
        import setproctitle
        setproctitle.setproctitle(title)
        logger.info("process title set: %s", title)
    except Exception as e:
        logger.warning("setproctitle unavailable, process title stays default: %s", e)


def main():
    _set_process_title(sys.argv[1:])
    here = os.path.dirname(os.path.abspath(__file__))
    comfy_dir = os.path.join(os.path.dirname(here), "ComfyUI")
    comfy_main = os.path.join(comfy_dir, "main.py")
    if not os.path.isfile(comfy_main):
        raise SystemExit(f"ComfyUI main.py not found at {comfy_main}")
    # Running main.py as a script puts ComfyUI/ at sys.path[0]; runpy does not,
    # so add it ourselves or `import comfy` fails.
    sys.path.insert(0, comfy_dir)
    sys.argv = [comfy_main] + sys.argv[1:]
    runpy.run_path(comfy_main, run_name="__main__")


if __name__ == "__main__":
    main()
