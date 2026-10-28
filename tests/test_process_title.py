import os
import sys
import types


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fusion_comfyui import launch_comfyui  # noqa: E402


class TestPortFromArgv:
    def test_default_port_when_absent(self):
        assert launch_comfyui._port_from_argv([]) == "11445"

    def test_port_space_form(self):
        assert launch_comfyui._port_from_argv(["--listen", "127.0.0.1", "--port", "12345"]) == "12345"

    def test_port_equals_form(self):
        assert launch_comfyui._port_from_argv(["--port=9999"]) == "9999"


class TestSetProcessTitle:
    def test_sets_title_with_port(self, monkeypatch):
        recorded = {}
        fake = types.ModuleType("setproctitle")
        fake.setproctitle = lambda t: recorded.setdefault("title", t)
        monkeypatch.setitem(sys.modules, "setproctitle", fake)
        launch_comfyui._set_process_title(["--port", "12345"])
        assert recorded["title"] == "fusion-comfyui serve :12345"

    def test_fallback_safe_when_setproctitle_missing(self, monkeypatch, caplog):
        import builtins

        real_import = builtins.__import__

        def _no_setproctitle(name, *args, **kwargs):
            if name == "setproctitle":
                raise ImportError("no setproctitle")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_setproctitle)
        with caplog.at_level("WARNING"):
            launch_comfyui._set_process_title(["--port", "12345"])
        assert any("setproctitle unavailable" in r.message for r in caplog.records)


class TestLauncherWiring:
    def test_start_sh_uses_launcher(self):
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(repo_root, "start.sh")) as f:
            content = f.read()
        assert "fusion_comfyui/launch_comfyui.py" in content
        assert '"$PROJECT_DIR/ComfyUI/main.py"' not in content

    def test_launchd_plist_uses_launcher(self):
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(repo_root, "start.sh")) as f:
            content = f.read()
        assert "${PROJECT_DIR}/fusion_comfyui/launch_comfyui.py" in content

    def test_comfyui_main_resolves(self):
        here = os.path.dirname(os.path.abspath(launch_comfyui.__file__))
        comfy_main = os.path.join(os.path.dirname(here), "ComfyUI", "main.py")
        assert os.path.isfile(comfy_main)


class TestMainDispatch:
    def test_main_runs_comfyui_with_argv0_swapped(self, monkeypatch):
        recorded = {}

        def fake_run_path(path, run_name):
            recorded["path"] = path
            recorded["run_name"] = run_name
            recorded["argv"] = list(sys.argv)

        monkeypatch.setattr(launch_comfyui.runpy, "run_path", fake_run_path)
        monkeypatch.setattr(sys, "argv", ["launch_comfyui.py", "--listen", "127.0.0.1", "--port", "12345"])
        launch_comfyui.main()
        assert recorded["run_name"] == "__main__"
        assert recorded["path"].endswith("ComfyUI/main.py")
        assert recorded["argv"][0].endswith("ComfyUI/main.py")
        assert recorded["argv"][1:] == ["--listen", "127.0.0.1", "--port", "12345"]
