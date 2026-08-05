import importlib
import importlib.util
import os
import sys

import pytest


PLUGIN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir
)
PLUGIN_DIR = os.path.normpath(PLUGIN_DIR)
INIT_PATH = os.path.join(PLUGIN_DIR, "__init__.py")

REQUIRED_NODES = [
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
    "KSamplerSelect",
    "BasicScheduler",
    "SaveImage",
    "LoadImage",
    "UNETLoader",
    "VAEDecode",
    "FusionKSampler",
    "FusionImageGen",
    "FusionVideoGen",
    "FusionDenoiseStats",
]


class _PyTorchBlocker:
    """Meta-path finder that raises ImportError for comfy/torch/torchvision.

    Proves the plugin registers its nodes WITHOUT importing any PyTorch-core
    module. comfy is still mocked (MagicMock) because the plugin's runtime
    helpers may reference it lazily inside methods, but a hard ImportError
    proves no module-load-time dependency on the real PyTorch stack.
    """

    _BLOCKED = ("comfy", "torch", "torchvision")

    def find_spec(self, name, path, target=None):
        top = name.split(".", 1)[0]
        if top in self._BLOCKED:
            raise ImportError(f"zero-PyTorch gate blocked import of {name}")
        return None


def _load_plugin_zero_pytorch():
    blocker = _PyTorchBlocker()
    sys.meta_path.insert(0, blocker)
    os.environ["FUSION_MLX_NO_STUB"] = "1"
    saved = {
        k: sys.modules.get(k) for k in list(sys.modules)
    }
    try:
        spec = importlib.util.spec_from_file_location(
            "fusion_comfyui_zero_pt", INIT_PATH,
            submodule_search_locations=[PLUGIN_DIR],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["fusion_comfyui_zero_pt"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop("fusion_comfyui_zero_pt", None)
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)


def test_plugin_imports_without_pytorch():
    mod = _load_plugin_zero_pytorch()
    assert hasattr(mod, "NODE_CLASS_MAPPINGS"), "NODE_CLASS_MAPPINGS missing"


def test_required_nodes_registered():
    mod = _load_plugin_zero_pytorch()
    mappings = mod.NODE_CLASS_MAPPINGS
    missing = [n for n in REQUIRED_NODES if n not in mappings]
    assert not missing, f"required nodes not registered: {missing}"


def test_registered_values_are_classes():
    mod = _load_plugin_zero_pytorch()
    for name, cls in mod.NODE_CLASS_MAPPINGS.items():
        assert isinstance(cls, type), f"{name} is not a class: {type(cls)}"


def test_display_name_mappings_present():
    mod = _load_plugin_zero_pytorch()
    assert isinstance(mod.NODE_DISPLAY_NAME_MAPPINGS, dict)
    for name in REQUIRED_NODES:
        if name not in mod.NODE_DISPLAY_NAME_MAPPINGS:
            pytest.fail(f"display name missing for {name}")
