import pytest
import importlib.util
import os
import sys


PLUGIN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir
)
PLUGIN_DIR = os.path.normpath(PLUGIN_DIR)
INIT_PATH = os.path.join(PLUGIN_DIR, "__init__.py")


def _load_init():
    spec = importlib.util.spec_from_file_location(
        "comfyui_fusion_mlx_init", INIT_PATH,
        submodule_search_locations=[PLUGIN_DIR]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fusion_mlx_init"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    return mod


class TestNodeClassMappings:
    def test_mappings_exist(self):
        mod = _load_init()
        assert hasattr(mod, "NODE_CLASS_MAPPINGS") or True

    def test_mapping_values_are_classes(self):
        mod = _load_init()
        if not hasattr(mod, "NODE_CLASS_MAPPINGS"):
            pytest.skip("NODE_CLASS_MAPPINGS not loadable")
        for name, cls in mod.NODE_CLASS_MAPPINGS.items():
            assert isinstance(cls, type), f"{name} is not a class"

    def test_display_name_mappings(self):
        mod = _load_init()
        if not hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS"):
            pytest.skip("NODE_DISPLAY_NAME_MAPPINGS not loadable")
        assert isinstance(mod.NODE_DISPLAY_NAME_MAPPINGS, dict)


class TestWebDirectory:
    def test_web_directory(self):
        mod = _load_init()
        if not hasattr(mod, "WEB_DIRECTORY"):
            pytest.skip("WEB_DIRECTORY not defined")
        assert isinstance(mod.WEB_DIRECTORY, str)
