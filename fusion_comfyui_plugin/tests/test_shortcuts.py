import numpy as np
import pytest
import sys
from unittest.mock import MagicMock, patch, AsyncMock



def _run_async_hack(coro):
    import asyncio as _asyncio
    try:
        loop = _asyncio.new_event_loop()
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_model(model_type="image"):
    from fusion_comfyui.core.wrappers import FusionModelWrapper
    mock = MagicMock(spec=FusionModelWrapper)
    mock.model_type = model_type
    mock.model_name = f"test-{model_type}"
    mock.get_engine.return_value._engine.generate = AsyncMock(
        return_value=[b"\x89PNG" + b"\x00" * 100]
    )
    mock.get_engine.return_value.ensure_started = AsyncMock()
    return mock


def _install_mock_av(container=None, import_error=False):
    mock_av = MagicMock()
    if import_error:
        mock_av.open.side_effect = ImportError("no av")
    elif container:
        mock_av.open.return_value = container
    else:
        mock_av.open.side_effect = ImportError("no av")
    sys.modules["av"] = mock_av
    return mock_av


def _remove_mock_av():
    sys.modules.pop("av", None)


class TestBytesToImageArray:
    def test_rgb(self):
        from nodes.shortcuts import _bytes_to_image_array
        from PIL import Image
        import io
        img = Image.new("RGB", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _bytes_to_image_array(buf.getvalue())
        assert isinstance(result, np.ndarray)
        assert result.ndim == 4
        assert result.shape == (1, 64, 64, 3)

    def test_rgba(self):
        from nodes.shortcuts import _bytes_to_image_array
        from PIL import Image
        import io
        img = Image.new("RGBA", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _bytes_to_image_array(buf.getvalue())
        assert isinstance(result, np.ndarray)
        assert result.ndim == 4
        assert result.shape == (1, 64, 64, 3)

    def test_grayscale(self):
        from nodes.shortcuts import _bytes_to_image_array
        from PIL import Image
        import io
        img = Image.new("L", (64, 64), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _bytes_to_image_array(buf.getvalue())
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 2


class TestVideoBytesToFrameArray:
    def test_decode_video(self):
        from nodes.shortcuts import _video_bytes_to_frame_array
        mock_frame1 = MagicMock()
        mock_frame1.to_ndarray.return_value = np.zeros((512, 768, 3), dtype=np.uint8)
        mock_frame2 = MagicMock()
        mock_frame2.to_ndarray.return_value = np.zeros((512, 768, 3), dtype=np.uint8)
        mock_container = MagicMock()
        mock_container.decode.return_value = [mock_frame1, mock_frame2]
        mock_container.close = MagicMock()
        _install_mock_av(container=mock_container)
        try:
            result = _video_bytes_to_frame_array(b"fake_mp4_data")
            assert isinstance(result, np.ndarray)
            assert result.ndim == 4
            assert result.shape[0] == 2
        finally:
            _remove_mock_av()

    def test_empty_frames_raises(self):
        from nodes.shortcuts import _video_bytes_to_frame_array
        mock_container = MagicMock()
        mock_container.decode.return_value = []
        mock_container.close = MagicMock()
        _install_mock_av(container=mock_container)
        try:
            with pytest.raises(RuntimeError, match="No frames"):
                _video_bytes_to_frame_array(b"fake_mp4_data")
        finally:
            _remove_mock_av()

    def test_no_av(self):
        from nodes.shortcuts import _video_bytes_to_frame_array
        _install_mock_av(import_error=True)
        try:
            with pytest.raises(ImportError):
                _video_bytes_to_frame_array(b"fake_mp4_data")
        finally:
            _remove_mock_av()


class TestFusionImageGenNode:
    def test_input_types(self):
        from nodes.shortcuts import FusionImageGenNode
        inputs = FusionImageGenNode.INPUT_TYPES()
        assert "required" in inputs

    def test_generate_with_pipeline(self):
        from nodes.shortcuts import FusionImageGenNode
        from PIL import Image
        import io as _io
        node = FusionImageGenNode()
        pipeline = MagicMock()
        img = Image.new("RGB", (64, 64), color="red")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        with patch("fusion_comfyui.core.lifecycle.FusionMemoryGuardian.maybe_purge"), \
             patch("fusion_comfyui.core.async_utils.run_async", return_value=[buf.getvalue()]):
            result = node.generate(pipeline, "a cat", "bad", 64, 64, 20, 6.0, 42)
        assert result is not None
        assert isinstance(result[0], np.ndarray)


class TestFusionVideoGenNode:
    def test_input_types(self):
        from nodes.shortcuts import FusionVideoGenNode
        inputs = FusionVideoGenNode.INPUT_TYPES()
        assert "required" in inputs


class TestFusionImageToVideoNode:
    def test_input_types(self):
        from nodes.shortcuts import FusionImageToVideoNode
        inputs = FusionImageToVideoNode.INPUT_TYPES()
        assert "required" in inputs

    def test_generate_i2v_raw_path_cleans_temp_png(self):
        from nodes.shortcuts import FusionImageToVideoNode
        from PIL import Image
        import os
        node = FusionImageToVideoNode()
        control = Image.new("RGB", (32, 32), color="blue")
        pipeline = MagicMock()
        pipeline.ensure_started = AsyncMock()
        captured = {}

        async def fake_generate(**kwargs):
            p = kwargs.get("image")
            captured["image_path"] = p
            assert p is not None and p.startswith("/tmp/fusion_i2v_"), p
            assert os.path.exists(p), f"temp png missing during generate: {p}"
            return [np.zeros((2, 32, 32, 3), dtype=np.uint8)]

        pipeline._engine.generate = fake_generate
        result = _run_async_hack(node._generate_i2v(
            pipeline, control, "prompt", "", 64, 64, 4, 24, 2, 5.0, 42, "none"))
        assert isinstance(result[0], np.ndarray)
        assert not os.path.exists(captured["image_path"]), \
            f"temp png not cleaned: {captured['image_path']}"

    def test_generate_i2v_mp4_path_cleans_temp_png(self):
        from nodes.shortcuts import FusionImageToVideoNode
        from PIL import Image
        import os
        node = FusionImageToVideoNode()
        control = Image.new("RGB", (32, 32), color="blue")
        pipeline = MagicMock()
        pipeline.ensure_started = AsyncMock()
        captured = {}

        call_count = {"n": 0}

        async def fake_generate(**kwargs):
            call_count["n"] += 1
            p = kwargs.get("image")
            captured["image_path"] = p
            assert p is not None and p.startswith("/tmp/fusion_i2v_"), p
            assert os.path.exists(p), f"temp png missing during generate: {p}"
            if call_count["n"] == 1:
                raise TypeError("raw not supported")
            return [b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50]

        pipeline._engine.generate = fake_generate
        result = _run_async_hack(node._generate_i2v(
            pipeline, control, "prompt", "", 64, 64, 4, 24, 2, 5.0, 42, "none"))
        assert isinstance(result[0], (bytes, bytearray))
        assert call_count["n"] == 2
        assert not os.path.exists(captured["image_path"]), \
            f"temp png not cleaned: {captured['image_path']}"
