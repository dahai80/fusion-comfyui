import asyncio
import logging
import os
import sys

from fusion_comfyui.core.timer import NodeTimer
from fusion_comfyui.core.engine_wrapper import FusionEngineWrapper, unload_all_fusion_engines
from fusion_comfyui.core.lifecycle import FusionMemoryGuardian
from fusion_comfyui.nodes.xiyouji.vlm import XiyoujiChapterParser
from fusion_comfyui.nodes.xiyouji.assemble import SceneVideoAssembler, ChapterVideoConcat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("xiyouji_pipeline")

CHARACTER_PORTRAITS = {
    "sunwukong": os.environ.get("PORTRAIT_SUNWUKONG", ""),
    "tangseng": os.environ.get("PORTRAIT_TANGSENG", ""),
    "zhubajie": os.environ.get("PORTRAIT_ZHUBAJIE", ""),
    "shaseng": os.environ.get("PORTRAIT_SHASENG", ""),
    "bailongma": os.environ.get("PORTRAIT_BAILONGMA", ""),
}

DEFAULT_MODEL = os.environ.get("XIYOUJI_MODEL", "FLUX.2-klein-base-4B")
DEFAULT_VLM = os.environ.get("XIYOUJI_VLM", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
DEFAULT_TTS = os.environ.get("XIYOUJI_TTS", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
DEFAULT_LIPSYNC_MODEL = os.environ.get("XIYOUJI_LIPSYNC_DIR", "")
DEFAULT_STEPS = int(os.environ.get("XIYOUJI_STEPS", "4"))
DEFAULT_WIDTH = int(os.environ.get("XIYOUJI_WIDTH", "512"))
DEFAULT_HEIGHT = int(os.environ.get("XIYOUJI_HEIGHT", "512"))


async def run_chapter(chapter_text: str, chapter_title: str = "chapter"):
    logger.info("=== Xiyouji Pipeline: %s ===", chapter_title)

    output_dir = os.environ.get("FUSION_OUTPUT_DIR", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Phase 0: Unload all engines
    logger.info("[Phase 0] Unloading all fusion-mlx engines...")
    await unload_all_fusion_engines()

    # Phase 1: Parse chapter into scenes (programmatic, no VLM to save memory)
    logger.info("[Phase 1] Parsing chapter into scenes (programmatic)...")
    parser = XiyoujiChapterParser()
    scenes = parser.split_only(chapter_text)
    logger.info("[Phase 1] Got %d scenes", len(scenes))

    # Phase 2: Pre-compute character identity embeddings
    logger.info("[Phase 2] Pre-computing character identity embeddings...")
    pulid_model_dir = os.environ.get("PULID_MODEL_DIR", "")
    skip_pulid = not bool(pulid_model_dir)
    if skip_pulid:
        logger.info("[Phase 2] PULID_MODEL_DIR not set, skipping PuLID identity extraction")

    # Phase 3: Generate scene images via monolithic generate()
    logger.info("[Phase 3] Generating scene images...")
    scene_videos = []
    scene_audios = []

    model = FusionEngineWrapper(DEFAULT_MODEL, offload_strategy="sequential", quant_bit="none")

    for idx, scene in enumerate(scenes):
        scene_id = int(scene.get("scene_id", idx + 1))
        desc_en = scene.get("description_en", "")
        desc_cn = scene.get("description_cn", "")
        scene.get("characters", [])
        scene.get("dialogue", [])
        float(scene.get("duration_seconds", 5))

        logger.info("[Phase 3] Scene %d: %s", scene_id, desc_cn[:60])

        frame_path = os.path.join(output_dir, f"{chapter_title}_scene_{scene_id}.png")

        try:
            png_bytes = await model.generate_image(
                prompt=desc_en,
                negative_prompt="low quality, blurry, deformed, watermark, text",
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                steps=DEFAULT_STEPS,
                cfg=3.5,
                seed=int(scene_id) * 1000,
            )
            with open(frame_path, "wb") as f:
                f.write(png_bytes)
            logger.info("[Phase 3] Scene %d image saved: %s (%d bytes)", scene_id, frame_path, len(png_bytes))
        except Exception as e:
            logger.warning("[Phase 3] Image gen failed for scene %d: %s — using placeholder", scene_id, e)
            from PIL import Image as PILImage
            pil_img = PILImage.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT), (128, 128, 128))
            pil_img.save(frame_path)
            FusionMemoryGuardian.purge_memory()

        # Image → video: for image-only models, the image IS the frame
        scene_videos.append(frame_path)
        FusionMemoryGuardian.purge_memory()

        # TTS for dialogue — skip for now (VLM+FLUX+TTS = OOM)
        scene_audios.append("")

    # Unload FLUX model after all scene images generated
    logger.info("[Phase 3.5] Unloading FLUX model to free memory...")
    await model.stop()
    FusionMemoryGuardian.purge_memory()

    # Phase 4: Assemble scenes
    logger.info("[Phase 4] Assembling chapter video...")
    assembled_scenes = []
    for idx, vid in enumerate(scene_videos):
        if not os.path.exists(vid):
            logger.warning("[Phase 4] Missing scene: %s", vid)
            continue
        audio = scene_audios[idx] if idx < len(scene_audios) else ""
        subtitle = scenes[idx].get("description_cn", "") if idx < len(scenes) else ""
        scene_dur = float(scenes[idx].get("duration_seconds", 5)) if idx < len(scenes) else 5.0
        try:
            assembler = SceneVideoAssembler()
            result = await assembler.execute(
                video_path=vid,
                audio_path=audio if audio and os.path.exists(audio) else "",
                subtitle_text=subtitle,
                subtitle_font_size=28,
                subtitle_y_offset=40,
                duration=scene_dur,
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
    xiyouji_path = os.environ.get("XIYOUJI_TEXT", "西游记.txt")

    if not os.path.exists(xiyouji_path):
        logger.error("西游记.txt not found at %s", xiyouji_path)
        sys.exit(1)

    with open(xiyouji_path, "r", encoding="utf-8") as f:
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
