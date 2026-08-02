import sys
import os
import types
from unittest.mock import MagicMock

import numpy as np

PLUGIN_ROOT = os.path.dirname(os.path.abspath(__file__))


def pytest_configure(config):
    """Install all mocks BEFORE pytest collects any packages.

    pytest's Package collector imports __init__.py as a standalone module
    (no parent package context), causing `from .core.lifecycle import ...`
    to fail. We monkey-patch Package.setup to skip the __init__.py import
    and pre-install all mocks so test imports resolve correctly.
    """
    # Monkey-patch _pytest.python.Package.setup to skip __init__.py import.
    import _pytest.python as _pp

    def _skip_init_setup(self):
        pass

    _pp.Package.setup = _skip_init_setup

    # --- mlx.core mock ---
    # _MxArray must be a real class so isinstance(data, mx.array) works
    class _MxArray:
        def __init__(self, *a, **kw):
            self.shape = getattr(a[0], "shape", (1,)) if a else (1,)
            self.dtype = kw.get("dtype", "float32")

        def __array__(self, dtype=None):
            return np.zeros(self.shape, dtype=dtype or np.float32)

        def __ge__(self, other):
            return True

        def __le__(self, other):
            return True

        def __gt__(self, other):
            return False

        def __lt__(self, other):
            return False

        def __mul__(self, other):
            return self

        def __rmul__(self, other):
            return self

    mock_mx = MagicMock()
    mock_mx.zeros = MagicMock(side_effect=lambda shape, dtype=None: np.zeros(shape, dtype=np.float32))
    mock_mx.array = _MxArray
    mock_mx.eval = MagicMock()
    mock_mx.float16 = "float16"
    mock_mx.bfloat16 = "bfloat16"
    mock_mx.float32 = "float32"
    mock_mx.int32 = "int32"
    mock_mx.uint8 = "uint8"
    mock_mx.metal = MagicMock()
    mock_mx.metal.clear_cache = MagicMock()
    mock_mx.metal.get_active_memory = MagicMock(return_value=1024 * 1024 * 100)
    mock_mx.metal.get_peak_memory = MagicMock(return_value=1024 * 1024 * 200)
    mock_mx.transpose = MagicMock(side_effect=lambda x, *a: x)
    mock_mx.concatenate = MagicMock(side_effect=lambda *a, **kw: np.concatenate(a, axis=kw.get("axis", 0)))
    mock_mx.broadcast_to = MagicMock(side_effect=lambda x, shape: np.broadcast_to(x, shape))
    mock_mx.split = MagicMock(side_effect=lambda x, n, **kw: np.array_split(x, n))
    mock_mx.fast = MagicMock()
    mock_mx.fast.scaled_dot_product_attention = MagicMock(
        return_value=np.zeros((1, 24, 10, 128), dtype=np.float32)
    )
    mock_mx.fast.rms_norm = MagicMock(side_effect=lambda x, w, eps: x)

    # --- mlx.nn mock ---
    mock_nn = MagicMock()
    mock_nn.Module = type("Module", (), {"__init__": lambda self: None})
    mock_nn.Linear = MagicMock(return_value=MagicMock())
    mock_nn.LayerNorm = MagicMock(return_value=MagicMock())
    mock_nn.Conv2d = MagicMock(return_value=MagicMock())
    mock_nn.RMSNorm = MagicMock(return_value=MagicMock())
    mock_nn.gelu = MagicMock(side_effect=lambda x: x)

    mock_mlx = MagicMock(core=mock_mx, nn=mock_nn)
    sys.modules["mlx"] = mock_mlx
    sys.modules["mlx.core"] = mock_mx
    sys.modules["mlx.nn"] = mock_nn

    # --- fusion_mlx mocks ---
    mock_fusion_mlx = MagicMock()
    mock_fusion_mlx._torch_stub = MagicMock()
    mock_fusion_mlx._torch_stub.install = MagicMock(return_value=True)
    sys.modules["fusion_mlx"] = mock_fusion_mlx
    sys.modules["fusion_mlx._torch_stub"] = mock_fusion_mlx._torch_stub
    sys.modules["fusion_mlx.engines"] = MagicMock()
    _engines_pkg = types.ModuleType("fusion_mlx.engines")
    _engines_pkg.__path__ = []
    _engines_pkg.__package__ = "fusion_mlx.engines"
    sys.modules["fusion_mlx.engines"] = _engines_pkg
    sys.modules["fusion_mlx.engines.image_gen"] = MagicMock()
    _video_engine = types.ModuleType("fusion_mlx.engines.video")
    _video_engine.VideoGenEngine = MagicMock()
    sys.modules["fusion_mlx.engines.video"] = _video_engine
    sys.modules["fusion_mlx.model_registry"] = MagicMock()
    sys.modules["fusion_mlx.video"] = MagicMock()
    sys.modules["fusion_mlx.video.pulid_mlx"] = MagicMock()
    sys.modules["fusion_mlx.video.pulid_mlx.pipeline"] = MagicMock()
    sys.modules["fusion_mlx.video.latentsync_mlx"] = MagicMock()
    sys.modules["fusion_mlx.video.latentsync_mlx.pipeline"] = MagicMock()

    # --- folder_paths mock ---
    mock_fp = MagicMock()
    mock_fp.models_dir = "/tmp/models"
    mock_fp.get_output_directory = MagicMock(return_value="/tmp/comfyui_output")
    mock_fp.get_temp_directory = MagicMock(return_value="/tmp/comfyui_temp")
    mock_fp.get_input_directory = MagicMock(return_value="/tmp/comfyui_input")
    mock_fp.get_save_image_path = MagicMock(return_value=(
        "/tmp/comfyui_output", "ComfyUI", 1, "", "ComfyUI"
    ))
    mock_fp.get_annotated_filepath = MagicMock(side_effect=lambda x: x)
    mock_fp.exists_annotated_filepath = MagicMock(return_value=True)
    mock_fp.get_filename_list = MagicMock(return_value=[])
    mock_fp.filter_files_content_types = MagicMock(side_effect=lambda files, types: files)
    sys.modules["folder_paths"] = mock_fp

    # --- comfy mock ---
    mock_comfy = MagicMock()
    mock_comfy.cli_args = MagicMock()
    mock_comfy.cli_args.disable_metadata = False
    sys.modules["comfy"] = mock_comfy
    sys.modules["comfy.cli_args"] = mock_comfy.cli_args
    sys.modules["comfy.samplers"] = MagicMock()

    # --- core package + sub-modules ---
    _core_pkg = types.ModuleType("core")
    _core_pkg.__path__ = [os.path.join(PLUGIN_ROOT, "core")]
    _core_pkg.__package__ = "core"
    sys.modules["core"] = _core_pkg

    _core_lifecycle = types.ModuleType("core.lifecycle")
    _core_lifecycle.FusionMemoryGuardian = MagicMock()
    _core_lifecycle.PipelineStageContext = MagicMock()
    _core_lifecycle._PURGE_THRESHOLD_MB = 1024
    sys.modules["core.lifecycle"] = _core_lifecycle

    # Replace stub with real module so PipelineStageContext tests work.
    try:
        import importlib
        real_lifecycle = importlib.reload(_core_lifecycle)
        sys.modules["core.lifecycle"] = real_lifecycle
    except Exception:
        pass

    _core_wrappers = types.ModuleType("core.wrappers")
    _core_wrappers.FusionModelWrapper = type("FusionModelWrapper", (), {})
    _core_wrappers.FusionCLIPWrapper = type("FusionCLIPWrapper", (), {})
    _core_wrappers.FusionVAEWrapper = type("FusionVAEWrapper", (), {})
    sys.modules["core.wrappers"] = _core_wrappers

    # Replace stub with real module so all classes/functions resolve.
    try:
        import importlib
        real_wrappers = importlib.reload(_core_wrappers)
        sys.modules["core.wrappers"] = real_wrappers
    except Exception:
        pass

    # --- nodes package (plugin's nodes, not ComfyUI's) ---
    _nodes_pkg = types.ModuleType("nodes")
    _nodes_pkg.__path__ = [os.path.join(PLUGIN_ROOT, "nodes")]
    _nodes_pkg.__package__ = "nodes"
    _nodes_pkg.NODE_CLASS_MAPPINGS = {}
    sys.modules["nodes"] = _nodes_pkg
