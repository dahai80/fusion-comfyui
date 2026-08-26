import numpy as np
import pytest
from unittest.mock import MagicMock


def _model(model_type):
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    m = MagicMock(spec=FusionModelWrapper)
    m.model_type = model_type
    return m


def _latent(**extra):
    base = {"samples": np.zeros((1, 16, 5, 64, 64), dtype=np.float32)}
    base.update(extra)
    return base


@pytest.mark.parametrize("model_type,latent_extra,denoise,positive_extra,expected", [
    ("video", {}, 1.0, {}, True),                                   # T2V pure text -> staged
    ("video", {"_i2v_image_path": "/tmp/x.png"}, 1.0, {}, False),   # I2V -> monolith
    ("video", {"_vace_control_video": "/tmp/v.mp4"}, 1.0, {}, False),  # VACE ctrl -> monolith
    ("video", {"_vace_control_mask": "/tmp/m.png"}, 1.0, {}, False),   # VACE mask -> monolith
    ("video", {"_vace_reference_images": "/tmp/r.png"}, 1.0, {}, False),  # VACE ref -> monolith
    ("image", {"_image_init_path": "/tmp/i.png"}, 0.5, {}, False),  # img2img (denoise<1) -> monolith
    ("image", {"_image_init_path": "/tmp/i.png"}, 1.0, {}, True),   # txt2img w/ stale init key, denoise=1 -> staged
    ("image", {}, 1.0, {}, True),                                   # txt2img FLUX.2 -> staged
    ("image", {}, 1.0, {"stable_cascade_prior": np.zeros((64, 64, 3))}, False),  # cascade stage_b -> pass-through
])
def test_should_use_staged_matrix(model_type, latent_extra, denoise, positive_extra, expected):
    from nodes.samplers import _should_use_staged
    model = _model(model_type)
    positive = {"prompt": "p", **positive_extra}
    negative = {"prompt": "n"}
    latent = _latent(**latent_extra)
    assert _should_use_staged(model, positive, negative, latent, denoise) is expected


def test_should_use_staged_negative_none():
    from nodes.samplers import _should_use_staged
    model = _model("video")
    assert _should_use_staged(model, {"prompt": "p"}, None, _latent(), 1.0) is True


def test_should_use_staged_unknown_type():
    from nodes.samplers import _should_use_staged
    model = _model("audio")  # neither video nor image
    assert _should_use_staged(model, {"prompt": "p"}, None, _latent(), 1.0) is False
