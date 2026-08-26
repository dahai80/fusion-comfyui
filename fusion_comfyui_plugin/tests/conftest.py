import sys
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# The plugin directory name "ComfyUI-Fusion-MLX" contains a hyphen, making
# `from custom_nodes.ComfyUI-Fusion-MLX.xxx` a SyntaxError.  We solve this
# by adding the plugin dir itself to sys.path so we can import core/nodes
# directly:  `from fusion_comfyui.core.wrappers import FusionModelWrapper`
#
# All heavy mock setup lives in the root conftest.py's pytest_configure hook,
# which runs before package discovery so __init__.py relative imports resolve.
# ---------------------------------------------------------------------------

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(PLUGIN_DIR)
COMFYUI_ROOT = os.path.dirname(os.path.dirname(PLUGIN_DIR))

sys.path.insert(0, COMFYUI_ROOT)
sys.path.insert(0, PLUGIN_DIR)
sys.path.insert(0, PLUGIN_ROOT)


@pytest.fixture
def mock_mlx_zeros():
    def _zeros(shape, dtype=None):
        return np.zeros(shape, dtype=np.float32)
    return _zeros


@pytest.fixture
def sample_model_wrapper():
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    return FusionModelWrapper(
        model_path="/tmp/test_model",
        model_name="test-model",
        model_type="video",
    )


@pytest.fixture
def sample_clip_wrapper():
    from fusion_comfyui.core.wrappers import FusionCLIPWrapper
    return FusionCLIPWrapper(
        model_path="/tmp/test_model",
        model_name="test-model",
        clip_type="wan",
    )


@pytest.fixture
def sample_vae_wrapper():
    from fusion_comfyui.core.wrappers import FusionVAEWrapper
    return FusionVAEWrapper(
        model_path="/tmp/test_model",
        model_name="test-model",
    )
