import base64
import io
import sys

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


class TestSaveVideoSR:
    def _setup_savevideo(self, monkeypatch, tmp_path, encoded):
        from fusion_comfyui_plugin.nodes import video_io as vio
        monkeypatch.setattr(
            vio.FusionSaveVideoNode, "_encode_video_av",
            lambda self, images, output_path, fps, codec, crf:
                (encoded.append((output_path, images.shape)), None)[1],
        )
        fake_fp = type(sys)("folder_paths")
        fake_fp.get_output_directory = lambda: str(tmp_path)
        fake_fp.get_save_image_path = lambda prefix, outdir, w, h: (
            str(tmp_path), prefix, 1, "", prefix)
        monkeypatch.setitem(sys.modules, "folder_paths", fake_fp)
        return encoded

    def test_savevideo_sr_off_unchanged_dims(self, monkeypatch, tmp_path):
        encoded = []
        self._setup_savevideo(monkeypatch, tmp_path, encoded)
        from fusion_comfyui_plugin.nodes.h3 import SaveVideo
        monkeypatch.delenv("FUSION_H3_SR", raising=False)
        video = {"images": np.zeros((2, 16, 20, 3), dtype=np.float32), "fps": 24.0}
        SaveVideo().save(video=video, filename_prefix="h3_sr")
        assert encoded
        _, shape = encoded[0]
        assert shape[1] == 16 and shape[2] == 20  # unchanged

    def test_savevideo_sr_mock_4x_dims(self, monkeypatch, tmp_path):
        encoded = []
        self._setup_savevideo(monkeypatch, tmp_path, encoded)
        monkeypatch.setenv("FUSION_H3_SR", "mock")
        from fusion_comfyui_plugin.nodes.h3 import SaveVideo
        video = {"images": np.zeros((2, 16, 20, 3), dtype=np.float32), "fps": 24.0}
        SaveVideo().save(video=video, filename_prefix="h3_sr")
        _, shape = encoded[0]
        assert shape[1] == 64 and shape[2] == 80  # 4x

    def test_savevideo_sr_mock_scale2_env(self, monkeypatch, tmp_path):
        encoded = []
        self._setup_savevideo(monkeypatch, tmp_path, encoded)
        monkeypatch.setenv("FUSION_H3_SR", "mock")
        monkeypatch.setenv("FUSION_H3_SR_SCALE", "2")
        from fusion_comfyui_plugin.nodes.h3 import SaveVideo
        video = {"images": np.zeros((2, 16, 20, 3), dtype=np.float32), "fps": 24.0}
        SaveVideo().save(video=video, filename_prefix="h3_sr")
        _, shape = encoded[0]
        assert shape[1] == 32 and shape[2] == 40  # 2x

    def test_savevideo_sr_unknown_mode_raises_no_silent_encode(self, monkeypatch, tmp_path):
        encoded = []
        self._setup_savevideo(monkeypatch, tmp_path, encoded)
        monkeypatch.setenv("FUSION_H3_SR", "true")
        from fusion_comfyui_plugin.nodes.h3 import SaveVideo
        video = {"images": np.zeros((2, 16, 20, 3), dtype=np.float32), "fps": 24.0}
        with pytest.raises(RuntimeError, match="unknown FUSION_H3_SR"):
            SaveVideo().save(video=video, filename_prefix="h3_sr")
        assert encoded == []  # no silent blurry encode

    def test_savevideo_sr_bad_scale_raises(self, monkeypatch, tmp_path):
        encoded = []
        self._setup_savevideo(monkeypatch, tmp_path, encoded)
        monkeypatch.setenv("FUSION_H3_SR", "mock")
        monkeypatch.setenv("FUSION_H3_SR_SCALE", "3")
        from fusion_comfyui_plugin.nodes.h3 import SaveVideo
        video = {"images": np.zeros((2, 16, 20, 3), dtype=np.float32), "fps": 24.0}
        with pytest.raises(RuntimeError, match="not in \\(2, 4\\)"):
            SaveVideo().save(video=video, filename_prefix="h3_sr")

    def test_savevideo_sr_non_numeric_scale_raises(self, monkeypatch, tmp_path):
        encoded = []
        self._setup_savevideo(monkeypatch, tmp_path, encoded)
        monkeypatch.setenv("FUSION_H3_SR", "mock")
        monkeypatch.setenv("FUSION_H3_SR_SCALE", "fast")
        from fusion_comfyui_plugin.nodes.h3 import SaveVideo
        video = {"images": np.zeros((2, 16, 20, 3), dtype=np.float32), "fps": 24.0}
        with pytest.raises(RuntimeError, match="not an integer"):
            SaveVideo().save(video=video, filename_prefix="h3_sr")


def test_sr_real_non_transport_exception_propagates_no_retry():
    from fusion_comfyui_plugin.nodes.sr import super_resolve_frames

    calls = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **k):
            calls.append(1)
            raise ValueError("programming bug, not transient")

    with patch("httpx.Client", FakeClient):
        imgs = np.zeros((1, 8, 8, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="programming bug"):
            super_resolve_frames(imgs, mode="1", scale=4)
    assert len(calls) == 1, "non-transport exception must not retry"


def test_sr_real_empty_input_returns_empty_array():
    from fusion_comfyui_plugin.nodes.sr import super_resolve_frames
    imgs = np.zeros((0, 8, 8, 3), dtype=np.float32)
    out = super_resolve_frames(imgs, mode="1", scale=4)
    assert out.shape == (0, 0, 0, 3), out.shape
    assert out.dtype == np.float32


def test_sr_url_normalization_trailing_slash():
    from fusion_comfyui_plugin.nodes.sr import super_resolve_frames

    captured = {}

    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return _fake_sr_response(8, 8)

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **k):
            captured["url"] = url
            return FakeResp()

    with patch("httpx.Client", FakeClient):
        imgs = np.random.rand(1, 8, 8, 3).astype(np.float32)
        super_resolve_frames(
            imgs, mode="1", scale=4,
            endpoint="http://host/v1/images/super-resolution/")
    assert captured["url"] == "http://host/v1/images/super-resolution", captured

