class TestGenerateVideo:
    async def test_generate_video_writes_mp4_and_returns_path(
        self, monkeypatch, tmp_path
    ):
        from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper

        wrapper = FusionEngineWrapper.__new__(FusionEngineWrapper)
        wrapper.model_name = "minimax-h3"
        wrapper.model_type = "video"
        wrapper._started = False
        wrapper._engine = None
        wrapper._on_step = None

        async def fake_ensure_started():
            wrapper._started = True

        monkeypatch.setattr(wrapper, "ensure_started", fake_ensure_started)

        class FakeEngine:
            async def generate(self, **kwargs):
                return (b"FAKE_MP4_BYTES",)

        wrapper._engine = FakeEngine()

        out = str(tmp_path / "scene_1.mp4")
        result = await wrapper.generate_video(
            prompt="a monkey king leaps",
            num_frames=25,
            width=768,
            height=448,
            seed=1000,
            output_path=out,
        )
        assert result == out
        with open(out, "rb") as f:
            assert f.read() == b"FAKE_MP4_BYTES"


class TestPipelineVideoPath:
    async def test_video_model_env_routes_to_generate_video(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("DRAMA_VIDEO_MODEL", "minimax-h3")
        monkeypatch.setenv("DRAMA_MODEL", "FLUX.2-klein-base-4B")
        monkeypatch.setenv("FUSION_OUTPUT_DIR", str(tmp_path))

        calls = {"video": [], "image": []}

        class FakeWrapper:
            model_type = "video"

            def __init__(self, *a, **k):
                pass

            async def ensure_started(self):
                pass

            async def generate_image(self, **k):
                calls["image"].append(k)
                return b"\x89PNG"

            async def generate_video(self, **k):
                out = str(tmp_path / f"v{len(calls['video'])}.mp4")
                with open(out, "wb") as f:
                    f.write(b"MP4")
                calls["video"].append(k)
                return out

            async def stop(self):
                pass

        monkeypatch.setattr(
            "fusion_comfyui.nodes.drama.pipeline.FusionEngineWrapper", FakeWrapper
        )
        monkeypatch.setattr(
            "fusion_comfyui.nodes.drama.pipeline.unload_all_fusion_engines",
            lambda *a: __import__("asyncio").sleep(0),
        )

        from fusion_comfyui.nodes.drama import pipeline as pl

        monkeypatch.setattr(
            pl.DramaChapterParser,
            "split_only",
            lambda self, t: [
                {
                    "scene_id": 1,
                    "description_en": "monkey king born",
                    "description_cn": "猴王出世",
                    "duration_seconds": 3,
                    "dialogue": ["I am born!"],
                },
                {
                    "scene_id": 2,
                    "description_en": "monkey learns magic",
                    "description_cn": "拜师学艺",
                    "duration_seconds": 3,
                    "dialogue": ["Teach me!"],
                },
            ],
        )
        from fusion_comfyui.nodes.drama.pipeline import run_chapter

        await run_chapter("第一回\n场景一：x。", "testchapter")

        assert len(calls["video"]) == 2, f"expected 2 video gens, got {calls}"
        assert len(calls["image"]) == 0, "video model set -> no image gen"

    async def test_no_video_env_falls_back_to_image(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DRAMA_VIDEO_MODEL", raising=False)
        monkeypatch.setenv("DRAMA_MODEL", "FLUX.2-klein-base-4B")
        monkeypatch.setenv("FUSION_OUTPUT_DIR", str(tmp_path))

        calls = {"video": [], "image": []}

        class FakeWrapper:
            model_type = "image"

            def __init__(self, *a, **k):
                pass

            async def ensure_started(self):
                pass

            async def generate_image(self, **k):
                calls["image"].append(k)
                return b"\x89PNG"

            async def generate_video(self, **k):
                calls["video"].append(k)
                return "x.mp4"

            async def stop(self):
                pass

        monkeypatch.setattr(
            "fusion_comfyui.nodes.drama.pipeline.FusionEngineWrapper", FakeWrapper
        )
        monkeypatch.setattr(
            "fusion_comfyui.nodes.drama.pipeline.unload_all_fusion_engines",
            lambda *a: __import__("asyncio").sleep(0),
        )
        from fusion_comfyui.nodes.drama import pipeline as pl

        monkeypatch.setattr(
            pl.DramaChapterParser,
            "split_only",
            lambda self, t: [
                {
                    "scene_id": 1,
                    "description_en": "x",
                    "description_cn": "x",
                    "duration_seconds": 3,
                    "dialogue": [],
                }
            ],
        )
        from fusion_comfyui.nodes.drama.pipeline import run_chapter

        await run_chapter("第一回\n场景一：x。", "testimg")

        assert len(calls["image"]) == 1, f"expected image fallback, got {calls}"
        assert len(calls["video"]) == 0


class TestNativeAudioPipeline:
    async def test_native_audio_env_forwards_audio_true(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DRAMA_VIDEO_MODEL", "minimax-h3")
        monkeypatch.setenv("DRAMA_NATIVE_AUDIO", "1")
        monkeypatch.setenv("FUSION_OUTPUT_DIR", str(tmp_path))

        calls = {"video": []}

        class FakeWrapper:
            model_type = "video"
            def __init__(self, *a, **k):
                pass
            async def ensure_started(self):
                pass
            async def generate_image(self, **k):
                return b"\x89PNG"
            async def generate_video(self, **k):
                out = str(tmp_path / f"v{len(calls['video'])}.mp4")
                with open(out, "wb") as f:
                    f.write(b"MP4")
                calls["video"].append(k)
                return out
            async def stop(self):
                pass

        monkeypatch.setattr("fusion_comfyui.nodes.drama.pipeline.FusionEngineWrapper", FakeWrapper)
        monkeypatch.setattr("fusion_comfyui.nodes.drama.pipeline.unload_all_fusion_engines",
                            lambda *a: __import__("asyncio").sleep(0))
        from fusion_comfyui.nodes.drama import pipeline as pl
        monkeypatch.setattr(pl.DramaChapterParser, "split_only",
                            lambda self, t: [
                                {"scene_id": 1, "description_en": "x", "description_cn": "x", "duration_seconds": 3, "dialogue": []},
                            ])
        from fusion_comfyui.nodes.drama.pipeline import run_chapter
        await run_chapter("第一回\n场景一：x。", "natv")

        assert calls["video"], "no video gen captured"
        assert calls["video"][0]["audio"] is True, f"audio=True not forwarded: {calls['video'][0]}"

    async def test_no_native_audio_env_forwards_audio_false(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DRAMA_VIDEO_MODEL", "minimax-h3")
        monkeypatch.delenv("DRAMA_NATIVE_AUDIO", raising=False)
        monkeypatch.setenv("FUSION_OUTPUT_DIR", str(tmp_path))

        calls = {"video": []}

        class FakeWrapper:
            model_type = "video"
            def __init__(self, *a, **k):
                pass
            async def ensure_started(self):
                pass
            async def generate_image(self, **k):
                return b"\x89PNG"
            async def generate_video(self, **k):
                out = str(tmp_path / f"v{len(calls['video'])}.mp4")
                with open(out, "wb") as f:
                    f.write(b"MP4")
                calls["video"].append(k)
                return out
            async def stop(self):
                pass

        monkeypatch.setattr("fusion_comfyui.nodes.drama.pipeline.FusionEngineWrapper", FakeWrapper)
        monkeypatch.setattr("fusion_comfyui.nodes.drama.pipeline.unload_all_fusion_engines",
                            lambda *a: __import__("asyncio").sleep(0))
        from fusion_comfyui.nodes.drama import pipeline as pl
        monkeypatch.setattr(pl.DramaChapterParser, "split_only",
                            lambda self, t: [
                                {"scene_id": 1, "description_en": "x", "description_cn": "x", "duration_seconds": 3, "dialogue": []},
                            ])
        from fusion_comfyui.nodes.drama.pipeline import run_chapter
        await run_chapter("第一回\n场景一：x。", "nonatv")

        assert calls["video"], "no video gen captured"
        assert calls["video"][0]["audio"] is False, f"audio should default False: {calls['video'][0]}"


class TestNativeAudioAssembler:
    async def _capture_cmd(self, monkeypatch, tmp_path, video_path, audio_path, native_audio):
        monkeypatch.setenv("FUSION_OUTPUT_DIR", str(tmp_path))
        captured = {}

        class FakeResult:
            returncode = 0
            stderr = ""

        def fake_run(cmd, **k):
            captured["cmd"] = cmd
            return FakeResult()

        import fusion_comfyui.nodes.drama.assemble as asm
        monkeypatch.setattr(asm.subprocess, "run", fake_run)
        node = asm.SceneVideoAssembler()
        await node.execute(
            video_path=str(video_path), audio_path=str(audio_path),
            native_audio=native_audio,
        )
        return captured["cmd"]

    async def test_amix_when_native_audio_and_tts(self, monkeypatch, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"MP4")
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"WAV")
        cmd = await self._capture_cmd(monkeypatch, tmp_path, video, audio, True)
        has_filter = any("filter_complex" in str(c) for c in cmd)
        has_amix = any("amix=inputs=2" in str(c) for c in cmd)
        assert has_filter and has_amix, f"expected amix filter_complex, got {cmd}"
        assert "-shortest" not in cmd, "amix path must not use -shortest"

    async def test_no_amix_when_native_audio_without_tts(self, monkeypatch, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"MP4")
        cmd = await self._capture_cmd(monkeypatch, tmp_path, video, "", True)
        has_filter = any("filter_complex" in str(c) for c in cmd)
        assert not has_filter, f"no amix without TTS, got {cmd}"
        assert cmd.count("-i") == 1, f"single input (native track only), got {cmd}"

    async def test_no_amix_when_tts_without_native_audio(self, monkeypatch, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"MP4")
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"WAV")
        cmd = await self._capture_cmd(monkeypatch, tmp_path, video, audio, False)
        has_filter = any("filter_complex" in str(c) for c in cmd)
        assert not has_filter, f"legacy path no amix, got {cmd}"
        assert "-shortest" in cmd, "legacy path keeps -shortest"
class TestExtractLastFrame:
    def test_extracts_final_frame_png(self, tmp_path):
        import subprocess
        from fusion_comfyui.nodes.drama.pipeline import _extract_last_frame

        video = str(tmp_path / "clip.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=0.3:size=32x32:rate=10",
                "-pix_fmt",
                "yuv420p",
                video,
            ],
            check=True,
        )
        out = str(tmp_path / "last.png")
        result = _extract_last_frame(video, out)
        assert result == out
        assert result is not None and __import__("os").path.exists(out)
        assert __import__("os").path.getsize(out) > 0

    def test_missing_video_returns_none(self, tmp_path):
        from fusion_comfyui.nodes.drama.pipeline import _extract_last_frame

        result = _extract_last_frame(
            str(tmp_path / "nope.mp4"), str(tmp_path / "o.png")
        )
        assert result is None


class TestSceneContinuity:
    async def test_second_scene_receives_first_frame_image(self, monkeypatch, tmp_path):
        # Real temp MP4s so _extract_last_frame produces a valid PNG that
        # scene 2 forwards as image= (H3 i2va continuity keyframe).
        import subprocess

        monkeypatch.setenv("DRAMA_VIDEO_MODEL", "minimax-h3")
        monkeypatch.setenv("FUSION_OUTPUT_DIR", str(tmp_path))
        calls = {"video": []}

        def make_clip(path):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=duration=0.2:size=32x32:rate=10",
                    "-pix_fmt",
                    "yuv420p",
                    path,
                ],
                check=True,
            )

        class FakeWrapper:
            model_type = "video"

            def __init__(self, *a, **k):
                pass

            async def ensure_started(self):
                pass

            async def generate_video(self, **k):
                import os

                calls["video"].append(
                    dict(
                        k,
                        _image_exists=(
                            os.path.exists(k["image"]) if "image" in k else None
                        ),
                    )
                )
                out = str(tmp_path / f"v{len(calls['video'])}.mp4")
                make_clip(out)
                return out

            async def stop(self):
                pass

        monkeypatch.setattr(
            "fusion_comfyui.nodes.drama.pipeline.FusionEngineWrapper", FakeWrapper
        )
        monkeypatch.setattr(
            "fusion_comfyui.nodes.drama.pipeline.unload_all_fusion_engines",
            lambda *a: __import__("asyncio").sleep(0),
        )
        from fusion_comfyui.nodes.drama import pipeline as pl

        monkeypatch.setattr(
            pl.DramaChapterParser,
            "split_only",
            lambda self, t: [
                {
                    "scene_id": 1,
                    "description_en": "born",
                    "description_cn": "出世",
                    "duration_seconds": 3,
                    "dialogue": [],
                },
                {
                    "scene_id": 2,
                    "description_en": "learns",
                    "description_cn": "学艺",
                    "duration_seconds": 3,
                    "dialogue": [],
                },
            ],
        )
        await pl.run_chapter("第一回\n场景一：x。", "cont")

        assert len(calls["video"]) == 2
        assert "image" not in calls["video"][0], "scene 1 has no prior frame"
        assert "image" in calls["video"][1], "scene 2 must get continuity first-frame"
        assert calls["video"][1][
            "_image_exists"
        ], "continuity image must exist at call time"
