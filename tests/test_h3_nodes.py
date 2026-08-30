import os
import sys

import numpy as np
import pytest

from fusion_comfyui_plugin.nodes.h3 import (
    MiniMaxH3SigmaShift,
    EmptyMiniMaxH3LatentAV,
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    VAEDecodeAudio,
    CreateVideo,
    SaveVideo,
    ImageScaleToTotalPixels,
    PrimitiveFloat,
    ComfyMathExpression,
)
from fusion_comfyui_plugin.nodes.samplers import _generate_monolithic


def _make_real_mp4(path):
    import av
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=4)
    stream.width = 8
    stream.height = 8
    stream.pix_fmt = "yuv420p"
    for _ in range(4):
        frame = av.VideoFrame(8, 8, "yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    with open(path, "rb") as _f:
        return _f.read()


class TestMiniMaxH3SigmaShift:
    def test_passthrough_returns_model_unchanged(self):
        model = {"_kind": "h3_model"}
        out = MiniMaxH3SigmaShift().shift(model, 12.0, 3.0)
        assert isinstance(out, tuple)
        assert out[0] is model

    def test_input_types_has_shift_params(self):
        req = MiniMaxH3SigmaShift.INPUT_TYPES()["required"]
        assert "model" in req
        assert "shift_video" in req
        assert "shift_audio" in req


class TestEmptyMiniMaxH3LatentAV:
    def test_latent_shape_z24_div16_div4(self):
        out = EmptyMiniMaxH3LatentAV().generate(width=960, height=544, length=73)
        latent = out[0]
        # z_channels=24, t=(73-1)//4+1=19, h//16=34, w//16=60
        assert latent["samples"].shape == (1, 24, 19, 34, 60)
        assert latent["num_frames"] == 73
        assert latent["width"] == 960
        assert latent["height"] == 544
        assert latent["_h3_audio"] is True


class TestMiniMaxH3ImageToVideo:
    def test_conditioning_dict_audio_false_no_image(self):
        out = MiniMaxH3ImageToVideo().generate(
            clip={}, vae={}, prompt="a cat", width=64, height=64, length=8
        )
        assert len(out) == 2
        assert out[0] == {"prompt": "a cat"}
        latent = out[1]
        assert latent["_h3_quantize"] == "dit8_te4"
        # fl2va (image/last_frame) is video-only: audio+image mutually exclusive
        assert latent["_h3_audio"] is False
        assert "_h3_first_frame_path" not in latent
        assert "_h3_last_frame_path" not in latent

    def test_first_frame_saved_under_tmp_png(self):
        img = np.zeros((1, 64, 64, 3), dtype=np.float32)
        out = MiniMaxH3ImageToVideo().generate(
            clip={}, vae={}, prompt="p", width=64, height=64, length=8,
            first_frame=img,
        )
        path = out[1]["_h3_first_frame_path"]
        try:
            # minimax_h3 _ALLOWED_READ_DIRS only permits /tmp (not macOS $TMPDIR)
            assert path.startswith("/tmp/")
            assert path.endswith(".png")
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_last_frame_saved_and_quantize_override(self):
        img = np.zeros((1, 32, 32, 3), dtype=np.float32)
        out = MiniMaxH3ImageToVideo().generate(
            clip={}, vae={}, prompt="p", width=64, height=64, length=8,
            last_frame=img, quantize="dit8",
        )
        path = out[1]["_h3_last_frame_path"]
        try:
            assert path.startswith("/tmp/") and path.endswith(".png")
            assert os.path.exists(path)
            assert out[1]["_h3_quantize"] == "dit8"
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestH3Res768pOverride:
    # AICF hardcodes 16:9 -> 960x544 (540p) in provider.ts (off-limits).
    # FUSION_H3_VIDEO_768P=1 raises the video resolution to 768p (1344x768
    # for 16:9) inside the fusion-comfyui node, no AICF code change. Only
    # raises (never lowers); rounds to mult of 32 (H3 patchify /2 needs even
    # latent dims); preserves aspect.

    def test_i2v_768p_raises_16x9_540p_to_768p_short_side(self, monkeypatch):
        monkeypatch.setenv("FUSION_H3_VIDEO_768P", "1")
        out = MiniMaxH3ImageToVideo().generate(
            clip={}, vae={}, prompt="p", width=960, height=544, length=73,
        )
        latent = out[1]
        # short side 544 -> 768, aspect preserved, mult of 32: 960x544 -> 1344x768
        assert latent["width"] == 1344
        assert latent["height"] == 768
        # latent spatial must match new res (/16), both dims even for patchify /2
        assert latent["samples"].shape[-2:] == (768 // 16, 1344 // 16)

    def test_i2v_768p_off_keeps_540p(self, monkeypatch):
        monkeypatch.delenv("FUSION_H3_VIDEO_768P", raising=False)
        out = MiniMaxH3ImageToVideo().generate(
            clip={}, vae={}, prompt="p", width=960, height=544, length=73,
        )
        latent = out[1]
        assert latent["width"] == 960
        assert latent["height"] == 544

    def test_i2v_768p_raises_9x16_portrait_to_768p_short_side(self, monkeypatch):
        monkeypatch.setenv("FUSION_H3_VIDEO_768P", "1")
        out = MiniMaxH3ImageToVideo().generate(
            clip={}, vae={}, prompt="p", width=544, height=960, length=73,
        )
        latent = out[1]
        # portrait: short side 544 -> 768 -> 768x1344
        assert latent["width"] == 768
        assert latent["height"] == 1344

    def test_i2v_768p_never_lowers_higher_res(self, monkeypatch):
        # 1:1 768x768 already >= 768p short side -> unchanged
        monkeypatch.setenv("FUSION_H3_VIDEO_768P", "1")
        out = MiniMaxH3ImageToVideo().generate(
            clip={}, vae={}, prompt="p", width=768, height=768, length=73,
        )
        latent = out[1]
        assert latent["width"] == 768
        assert latent["height"] == 768

    def test_r2v_768p_also_raises_16x9(self, monkeypatch):
        monkeypatch.setenv("FUSION_H3_VIDEO_768P", "1")
        out = MiniMaxH3ReferenceToVideo().generate(
            clip={}, vae={}, prompt="p", width=960, height=544, length=73,
        )
        latent = out[1]
        assert latent["width"] == 1344
        assert latent["height"] == 768


class TestMiniMaxH3ReferenceToVideo:
    def test_ref_images_dict_to_tmp_paths(self):
        ref = np.zeros((1, 32, 32, 3), dtype=np.float32)
        out = MiniMaxH3ReferenceToVideo().generate(
            clip={}, vae={}, prompt="p", width=64, height=64, length=8,
            ref_images={"ref_image_1": ref},
        )
        refs = out[1]["_h3_ref_images"]
        try:
            assert isinstance(refs, list)
            assert len(refs) == 1
            assert refs[0].startswith("/tmp/") and refs[0].endswith(".png")
            assert os.path.exists(refs[0])
            assert out[1]["_h3_audio"] is False
        finally:
            for p in refs:
                if os.path.exists(p):
                    os.unlink(p)

    def test_ref_images_plain_image_batch(self):
        batch = np.zeros((2, 16, 16, 3), dtype=np.float32)
        out = MiniMaxH3ReferenceToVideo().generate(
            clip={}, vae={}, prompt="p", width=64, height=64, length=8,
            ref_images=batch,
        )
        refs = out[1]["_h3_ref_images"]
        try:
            assert len(refs) == 2
            for p in refs:
                assert os.path.exists(p)
        finally:
            for p in refs:
                if os.path.exists(p):
                    os.unlink(p)


class TestVAEDecodeAudio:
    def test_returns_silent_dummy(self):
        out = VAEDecodeAudio().decode(samples={"samples": np.zeros(1)}, vae={})
        assert isinstance(out, tuple)
        audio = out[0]
        assert audio["waveform"].shape == (1, 2)
        assert np.allclose(audio["waveform"], 0.0)
        assert audio["sample_rate"] == 24000


class TestCreateVideo:
    def test_passthrough_forwards_frames_fps_audio(self):
        images = np.zeros((4, 8, 8, 3), dtype=np.float32)
        audio = {"waveform": np.zeros((1, 2), dtype=np.float32)}
        out = CreateVideo().create(images=images, audio=audio, fps=24.0)
        assert isinstance(out, tuple)
        vid = out[0]
        assert vid["images"] is images
        assert vid["fps"] == 24.0
        assert vid["audio"] is audio


class TestSaveVideo:
    def test_writes_mp4_ui_output(self, monkeypatch, tmp_path):
        # Stub the heavy PyAV encode so the test only checks path/ui wiring.
        from fusion_comfyui_plugin.nodes import video_io as vio
        encoded = []
        monkeypatch.setattr(
            vio.FusionSaveVideoNode, "_encode_video_av",
            lambda self, images, output_path, fps, codec, crf: encoded.append(output_path),
        )
        # Inject a fake folder_paths so SaveVideo never touches the real ComfyUI one.
        fake_fp = type(sys)("folder_paths")
        fake_fp.get_output_directory = lambda: str(tmp_path)
        fake_fp.get_save_image_path = lambda prefix, outdir, w, h: (
            str(tmp_path), prefix, 1, "", prefix,
        )
        monkeypatch.setitem(sys.modules, "folder_paths", fake_fp)

        video = {"images": np.zeros((2, 8, 8, 3), dtype=np.float32), "fps": 24.0}
        result = SaveVideo().save(video=video, filename_prefix="h3_test")
        assert "ui" in result and "videos" in result["ui"]
        entry = result["ui"]["videos"][0]
        assert entry["filename"].endswith(".mp4")
        assert entry["type"] == "output"
        assert encoded and encoded[0].endswith(entry["filename"])


class TestAuxNodes:
    def test_image_scale_to_total_pixels_1mp(self):
        img = np.zeros((1, 64, 64, 3), dtype=np.float32)
        out = ImageScaleToTotalPixels().upscale(
            image=img, upscale_method="lanczos", megapixels=1.0, resolution_steps=32
        )
        # 64x64 -> 1024x1024 at 1MP, step 32
        assert out[0].shape == (1, 1024, 1024, 3)

    def test_primitive_float_emit(self):
        assert PrimitiveFloat().emit(float=7.5) == (7.5,)

    def test_comfy_math_expression_max_round(self):
        val = ComfyMathExpression().eval_expr(expression="max(5, round(a*24))", a=5)
        assert val == (120.0,)

    def test_comfy_math_expression_sandbox_no_builtins(self):
        with pytest.raises(NameError):
            ComfyMathExpression().eval_expr(expression="__import__('os')", a=0)


class TestGenerateMonolithicH3Forwarding:
    async def test_h3_keys_forwarded_to_engine(self):
        class FakeInner:
            def __init__(self):
                self.kwargs = None

            async def generate(self, **kwargs):
                self.kwargs = kwargs
                return [np.zeros((2, 4, 4, 3), dtype=np.uint8)]

        class FakeEngineWrapper:
            def __init__(self):
                self._engine = FakeInner()

            async def ensure_started(self):
                pass

        class FakeModelWrapper:
            model_type = "video"

            def __init__(self):
                self._wrapper = FakeEngineWrapper()

            def get_engine(self):
                return self._wrapper

        mw = FakeModelWrapper()
        latent_image = {
            "samples": np.zeros((1, 24, 3, 4, 4), dtype=np.float32),
            "num_frames": 8,
            "width": 64,
            "height": 64,
            "_h3_quantize": "dit8_te4",
            "_h3_audio": True,
            "_h3_first_frame_path": "/tmp/fake_first.png",
            "_h3_ref_images": ["/tmp/fake_ref.png"],
        }
        await _generate_monolithic(
            mw, {"prompt": "p"}, {"prompt": ""},
            latent_image, steps=10, cfg=6.0, seed=1,
            width=64, height=64, num_frames=8,
        )
        kwargs = mw.get_engine()._engine.kwargs
        assert kwargs["quantize"] == "dit8_te4"
        assert kwargs["audio"] is True
        assert kwargs["image"] == "/tmp/fake_first.png"
        assert kwargs["reference_images"] == ["/tmp/fake_ref.png"]


class TestGenerateMonolithicMp4NoDoubleGenerate:
    async def test_h3_mp4_bytes_generate_called_once(self, tmp_path):
        # H3 backend ignores output_format="raw" and always returns mp4 bytes.
        # The first generate already produced the mp4 — re-generating doubles
        # wall-clock (17min x2 = 34min) and blows the AICF 30min poll deadline.
        # Regression guard: when the engine returns bytes, generate MUST be
        # called exactly once (reuse the bytes, do not re-generate).
        fake_mp4 = _make_real_mp4(tmp_path / "fixture.mp4")

        class FakeInner:
            def __init__(self):
                self.call_count = 0

            async def generate(self, **kwargs):
                self.call_count += 1
                return [fake_mp4]

        class FakeEngineWrapper:
            def __init__(self):
                self._engine = FakeInner()

            async def ensure_started(self):
                pass

        class FakeModelWrapper:
            model_type = "video"

            def __init__(self):
                self._wrapper = FakeEngineWrapper()

            def get_engine(self):
                return self._wrapper

        mw = FakeModelWrapper()
        latent_image = {
            "samples": np.zeros((1, 24, 3, 4, 4), dtype=np.float32),
            "num_frames": 8,
            "width": 64,
            "height": 64,
            "_h3_quantize": "dit8_te4",
            "_h3_first_frame_path": "/tmp/fake_first.png",
            "_h3_last_frame_path": "/tmp/fake_last.png",
        }
        result = await _generate_monolithic(
            mw, {"prompt": "p"}, {"prompt": ""},
            latent_image, steps=10, cfg=6.0, seed=1,
            width=64, height=64, num_frames=8,
        )
        assert mw.get_engine()._engine.call_count == 1, (
            "mp4-bytes path must not re-generate (would double wall-clock)"
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim >= 3
