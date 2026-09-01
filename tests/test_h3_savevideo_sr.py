import base64
import io

import numpy as np
import pytest
from unittest.mock import patch

from fusion_comfyui_plugin.nodes.sr import super_resolve_frames


def test_sr_mode_0_passthrough():
    imgs = np.random.rand(3, 16, 20, 3).astype(np.float32)
    out = super_resolve_frames(imgs, mode="0", scale=4)
    assert out.shape == (3, 16, 20, 3)
    assert np.array_equal(out, imgs)


def test_sr_mock_resize_4x_dims():
    imgs = np.random.rand(2, 16, 20, 3).astype(np.float32)
    out = super_resolve_frames(imgs, mode="mock", scale=4)
    assert out.shape == (2, 64, 80, 3), out.shape
    assert out.dtype == np.float32


def test_sr_mock_resize_2x_dims():
    imgs = np.zeros((1, 32, 40, 3), dtype=np.float32)
    out = super_resolve_frames(imgs, mode="mock", scale=2)
    assert out.shape == (1, 64, 80, 3), out.shape


def test_sr_mock_preserves_range():
    imgs = np.zeros((1, 12, 12, 3), dtype=np.float32)
    out = super_resolve_frames(imgs, mode="mock", scale=4)
    assert np.isfinite(out).all()


def _fake_sr_response(out_h, out_w):
    from PIL import Image
    arr = (np.random.rand(out_h, out_w, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode(),
            "width": out_w, "height": out_h,
            "in_width": out_w // 4, "in_height": out_h // 4, "elapsed": 0.1}


def test_sr_real_per_frame_post_called_n_times():
    from fusion_comfyui_plugin.nodes.sr import super_resolve_frames

    calls = []

    class FakeResp:
        def __init__(self, body):
            self.status_code = 200
            self._body = body
            self.text = ""

        def json(self):
            return self._body

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **k):
            calls.append((url, k.get("data", {}).get("scale")))
            return FakeResp(_fake_sr_response(8, 8))

    with patch("httpx.Client", FakeClient):
        imgs = np.random.rand(3, 8, 8, 3).astype(np.float32)
        out = super_resolve_frames(imgs, mode="1", scale=4)
    assert len(calls) == 3, calls
    assert out.shape == (3, 8, 8, 3), out.shape


def test_sr_real_5xx_raises_runtime_error_no_fallback():
    from fusion_comfyui_plugin.nodes.sr import super_resolve_frames

    class FakeResp:
        status_code = 500
        text = "boom"

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **k):
            return FakeResp()

    with patch("httpx.Client", FakeClient):
        imgs = np.zeros((2, 8, 8, 3), dtype=np.float32)
        with pytest.raises(RuntimeError):
            super_resolve_frames(imgs, mode="1", scale=4)


def test_sr_real_timeout_raises():
    import httpx
    from fusion_comfyui_plugin.nodes.sr import super_resolve_frames

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **k):
            raise httpx.TimeoutException("timed out")

    with patch("httpx.Client", FakeClient):
        imgs = np.zeros((1, 8, 8, 3), dtype=np.float32)
        with pytest.raises(RuntimeError):
            super_resolve_frames(imgs, mode="1", scale=4)
