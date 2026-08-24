import asyncio
import logging
import os
import subprocess
import sys

from fusion_comfyui.core.timer import NodeTimer
from fusion_comfyui.core.engine_wrapper import (
    FusionEngineWrapper,
    unload_all_fusion_engines,
)
from fusion_comfyui.core.lifecycle import FusionMemoryGuardian
from fusion_comfyui.nodes.drama.vlm import DramaChapterParser
from fusion_comfyui.nodes.drama.assemble import SceneVideoAssembler, ChapterVideoConcat

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("drama_pipeline")

CHARACTER_PORTRAITS = {
    "sunwukong": os.environ.get("PORTRAIT_SUNWUKONG", ""),
    "tangseng": os.environ.get("PORTRAIT_TANGSENG", ""),
    "zhubajie": os.environ.get("PORTRAIT_ZHUBAJIE", ""),
    "shaseng": os.environ.get("PORTRAIT_SHASENG", ""),
    "bailongma": os.environ.get("PORTRAIT_BAILONGMA", ""),
}

DEFAULT_MODEL = os.environ.get("DRAMA_MODEL", "FLUX.2-klein-base-4B")
DEFAULT_VIDEO_MODEL = os.environ.get("DRAMA_VIDEO_MODEL", "")
DEFAULT_QUANTIZE = os.environ.get("DRAMA_QUANTIZE", "dit8_te4")
DEFAULT_VLM = os.environ.get("DRAMA_VLM", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
DEFAULT_TTS = os.environ.get("DRAMA_TTS", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
DEFAULT_LIPSYNC_MODEL = os.environ.get("DRAMA_LIPSYNC_DIR", "")
DRAMA_ENABLE_TTS = os.environ.get("DRAMA_TTS_ENABLED", "0") == "1"
DEFAULT_STEPS = int(os.environ.get("DRAMA_STEPS", "4"))
DEFAULT_WIDTH = int(os.environ.get("DRAMA_WIDTH", "512"))
DEFAULT_HEIGHT = int(os.environ.get("DRAMA_HEIGHT", "512"))
DEFAULT_VIDEO_FRAMES = int(os.environ.get("DRAMA_VIDEO_FRAMES", "49"))
DEFAULT_VIDEO_FPS = int(os.environ.get("DRAMA_VIDEO_FPS", "24"))
# F1: H3 原生音频联合生成（t2va A/V）。默认关——H3 33B 联合生成显存峰值更高，
# 且 F2 TTS 台词外挂仍是主要人声来源。开启后 video mp4 自带音频轨（环境音/配乐），
# SceneVideoAssembler 用 amix 把原生音频轨 + F2 TTS wav 混合（人声叠环境音）。
DRAMA_NATIVE_AUDIO = os.environ.get("DRAMA_NATIVE_AUDIO", "0") == "1"


def _extract_last_frame(video_path: str, out_path: str) -> str | None:
    # Extract the final video frame to out_path (PNG) for scene continuity
    # conditioning. Returns out_path on success, None on failure (logged).
    # Uses ffprobe for frame count then ffmpeg select=eq(n,N-1) — robust for
    # short clips where -sseof under-reads.
    if not os.path.exists(video_path):
        logger.warning("last-frame extract: video missing %s", video_path)
        return None
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        n_frames = int(probe.stdout.strip())
        if n_frames <= 0:
            logger.warning("last-frame extract: 0 frames in %s", video_path)
            return None
        last_idx = n_frames - 1
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-vf",
                f"select=eq(n\\,{last_idx})",
                "-frames:v",
                "1",
                "-update",
                "1",
                out_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            logger.warning("last-frame extract failed: %s", result.stderr[:200])
            return None
        logger.info(
            "last-frame extract: %s -> %s (frame %d/%d)",
            video_path,
            out_path,
            last_idx,
            n_frames,
        )
        return out_path
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning("last-frame extract error on %s: %s", video_path, e)
        return None


async def run_chapter(chapter_text: str, chapter_title: str = "chapter"):
    logger.info("=== Drama Pipeline: %s ===", chapter_title)

    output_dir = os.environ.get("FUSION_OUTPUT_DIR", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Phase 0: Unload all engines
    logger.info("[Phase 0] Unloading all fusion-mlx engines...")
    await unload_all_fusion_engines()

    # Phase 1: Parse chapter into scenes (programmatic, no VLM to save memory)
    logger.info("[Phase 1] Parsing chapter into scenes (programmatic)...")
    parser = DramaChapterParser()
    scenes = parser.split_only(chapter_text)
    logger.info("[Phase 1] Got %d scenes", len(scenes))

    # Phase 2: Pre-compute character identity embeddings
    logger.info("[Phase 2] Pre-computing character identity embeddings...")
    pulid_model_dir = os.environ.get("PULID_MODEL_DIR", "")
    skip_pulid = not bool(pulid_model_dir)
    if skip_pulid:
        logger.info(
            "[Phase 2] PULID_MODEL_DIR not set, skipping PuLID identity extraction"
        )

    # Phase 3: Generate scene media (video if DRAMA_VIDEO_MODEL set, else image)
    # Read env at call time (not import) so tests + runtime can toggle per-run.
    video_model = os.environ.get("DRAMA_VIDEO_MODEL", "")
    use_video = bool(video_model)
    logger.info("[Phase 3] Generating scene %s...", "video" if use_video else "images")

    scene_videos = []
    scene_audios = []
    # native_audio_per_scene: 记录每场景 video 是否已含 H3 原生音频轨，
    # 供 Phase 4 assembler 决定 amix（原生轨+F2 TTS）还是单轨替换。
    native_audio_per_scene = []

    model_name = (
        video_model if use_video else os.environ.get("DRAMA_MODEL", DEFAULT_MODEL)
    )
    model = FusionEngineWrapper(
        model_name, offload_strategy="sequential", quant_bit="none"
    )

    # Scene visual continuity: previous scene's final frame conditions the
    # next scene's first frame (H3 i2va keyframe). Cleared after the loop.
    prev_last_frame = ""
    continuity_frames = []

    # Optional TTS engine (loaded once, reused per scene)
    tts_engine = None
    if os.environ.get("DRAMA_TTS_ENABLED", "0") == "1":
        try:
            tts_engine = FusionEngineWrapper(DEFAULT_TTS, offload_strategy="sequential")
            await tts_engine.load_tts(DEFAULT_TTS)
            logger.info("[Phase 3] TTS engine loaded: %s", DEFAULT_TTS)
        except Exception as e:
            logger.warning("[Phase 3] TTS load failed, audio disabled: %s", e)
            tts_engine = None

    for idx, scene in enumerate(scenes):
        scene_id = int(scene.get("scene_id", idx + 1))
        desc_en = scene.get("description_en", "")
        desc_cn = scene.get("description_cn", "")
        dialogue = scene.get("dialogue", [])
        seed = int(scene_id) * 1000

        logger.info("[Phase 3] Scene %d: %s", scene_id, desc_cn[:60])

        try:
            if use_video:
                video_path = os.path.join(
                    output_dir, f"{chapter_title}_scene_{scene_id}.mp4"
                )
                # call-time 读 DRAMA_NATIVE_AUDIO（与 use_video 同理，测试可 per-run toggle）。
                native_audio = os.environ.get("DRAMA_NATIVE_AUDIO", "0") == "1"
                # Scene continuity: previous scene's final frame conditions this
                # scene's first frame (H3 fl2va keyframe).
                continuity_kwargs = {}
                if prev_last_frame and os.path.exists(prev_last_frame):
                    continuity_kwargs["image"] = prev_last_frame
                    logger.info(
                        "[Phase 3] Scene %d continuity: first-frame from %s",
                        scene_id,
                        prev_last_frame,
                    )
                video_path = await model.generate_video(
                    prompt=desc_en,
                    num_frames=DEFAULT_VIDEO_FRAMES,
                    width=DEFAULT_WIDTH,
                    height=DEFAULT_HEIGHT,
                    seed=seed,
                    output_path=video_path,
                    fps=DEFAULT_VIDEO_FPS,
                    quantize=DEFAULT_QUANTIZE,
                    audio=native_audio,
                    **continuity_kwargs,
                )
                scene_videos.append(video_path)
                native_audio_per_scene.append(native_audio)
                logger.info(
                    "[Phase 3] Scene %d video saved: %s (native_audio=%s)",
                    scene_id, video_path, native_audio,
                )
                # Extract this scene's last frame for the next scene's continuity.
                next_frame = os.path.join(
                    output_dir, f"{chapter_title}_scene_{scene_id}_last.png"
                )
                extracted = _extract_last_frame(video_path, next_frame)
                if extracted:
                    prev_last_frame = extracted
                    continuity_frames.append(extracted)
                else:
                    prev_last_frame = ""
            else:
                frame_path = os.path.join(
                    output_dir, f"{chapter_title}_scene_{scene_id}.png"
                )
                png_bytes = await model.generate_image(
                    prompt=desc_en,
                    negative_prompt="low quality, blurry, deformed, watermark, text",
                    width=DEFAULT_WIDTH,
                    height=DEFAULT_HEIGHT,
                    steps=DEFAULT_STEPS,
                    cfg=3.5,
                    seed=seed,
                )
                with open(frame_path, "wb") as f:
                    f.write(png_bytes)
                scene_videos.append(frame_path)
                native_audio_per_scene.append(False)
                logger.info(
                    "[Phase 3] Scene %d image saved: %s (%d bytes)",
                    scene_id,
                    frame_path,
                    len(png_bytes),
                )
        except Exception as e:
            logger.warning(
                "[Phase 3] Gen failed for scene %d: %s — placeholder", scene_id, e
            )
            if use_video:
                from PIL import Image as PILImage

                placeholder = os.path.join(
                    output_dir, f"{chapter_title}_scene_{scene_id}.png"
                )
                PILImage.new(
                    "RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT), (128, 128, 128)
                ).save(placeholder)
                scene_videos.append(placeholder)
                native_audio_per_scene.append(False)
            FusionMemoryGuardian.purge_memory()

        # TTS for dialogue lines (concatenate all lines into one audio)
        audio_path = ""
        if tts_engine and dialogue:
            try:
                full_line = (
                    " ".join(str(d) for d in dialogue)
                    if isinstance(dialogue, list)
                    else str(dialogue)
                )
                wav_bytes = await tts_engine.tts_synthesize(text=full_line, speed=1.0)
                audio_path = os.path.join(
                    output_dir, f"{chapter_title}_scene_{scene_id}.wav"
                )
                with open(audio_path, "wb") as f:
                    f.write(wav_bytes)
                logger.info("[Phase 3] Scene %d TTS saved: %s", scene_id, audio_path)
            except Exception as e:
                logger.warning("[Phase 3] TTS failed for scene %d: %s", scene_id, e)
                audio_path = ""
        scene_audios.append(audio_path)

        FusionMemoryGuardian.purge_memory()

    # Unload generation model + TTS after all scenes
    logger.info("[Phase 3.5] Unloading model(s) to free memory...")
    await model.stop()
    if tts_engine:
        await tts_engine.stop()
    FusionMemoryGuardian.purge_memory()

    # Clean continuity temp frames (intermediate, not final output)
    for frame_path in continuity_frames:
        try:
            os.unlink(frame_path)
        except OSError:
            pass
    logger.info("[Phase 3.5] Cleaned %d continuity temp frames", len(continuity_frames))

    # Phase 4: Assemble scenes
    logger.info("[Phase 4] Assembling chapter video...")
    assembled_scenes = []
    for idx, vid in enumerate(scene_videos):
        if not os.path.exists(vid):
            logger.warning("[Phase 4] Missing scene: %s", vid)
            continue
        audio = scene_audios[idx] if idx < len(scene_audios) else ""
        native_audio = native_audio_per_scene[idx] if idx < len(native_audio_per_scene) else False
        subtitle = scenes[idx].get("description_cn", "") if idx < len(scenes) else ""
        scene_dur = (
            float(scenes[idx].get("duration_seconds", 5)) if idx < len(scenes) else 5.0
        )
        try:
            assembler = SceneVideoAssembler()
            result = await assembler.execute(
                video_path=vid,
                audio_path=audio if audio and os.path.exists(audio) else "",
                subtitle_text=subtitle,
                subtitle_font_size=28,
                subtitle_y_offset=40,
                duration=scene_dur,
                native_audio=native_audio,
            )
            assembled_scenes.append(result[0])
        except Exception as e:
            logger.warning("[Phase 4] Assemble failed for scene %d: %s", idx, e)

    if assembled_scenes:
        concat = ChapterVideoConcat()
        final_result = await concat.execute(
            video_paths=",".join(assembled_scenes),
            chapter_title=chapter_title,
        )
        final_path = final_result[0]
    else:
        final_path = ""
        logger.warning("[Phase 4] No scenes to concatenate")

    # Phase 5: Cleanup
    logger.info("[Phase 5] Final cleanup...")
    await model.stop()
    await unload_all_fusion_engines()

    # Report timing
    summary = NodeTimer.summary()
    logger.info("\n%s", summary)

    csv_path = os.path.join(output_dir, f"{chapter_title}_timing.csv")
    csv_content = NodeTimer.export_csv()
    with open(csv_path, "w") as f:
        f.write(csv_content)
    logger.info("Timing data exported to %s", csv_path)

    logger.info("=== Pipeline complete: %s ===", final_path)
    return final_path


def main():
    chapter_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    drama_path = os.environ.get("DRAMA_TEXT", "script.txt")

    if not os.path.exists(drama_path):
        logger.error("script.txt not found at %s", drama_path)
        sys.exit(1)

    with open(drama_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chapters = full_text.split("第")[1:]
    if chapter_num > len(chapters):
        logger.error("Chapter %d not found (total %d)", chapter_num, len(chapters))
        sys.exit(1)

    chapter_text = "第" + chapters[chapter_num - 1]
    chapter_title = f"第{chapter_num}回"

    asyncio.run(run_chapter(chapter_text, chapter_title))


if __name__ == "__main__":
    main()
