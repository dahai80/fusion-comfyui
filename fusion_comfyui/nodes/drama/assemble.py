import logging
import os
import subprocess

from fusion_comfyui.nodes.base import BaseNode
from fusion_comfyui.core.timer import NodeTimer

logger = logging.getLogger("fusion_comfyui.nodes.drama.assemble")

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _is_image(path: str) -> bool:
    return path.lower().endswith(_IMAGE_EXTS)


def _burn_subtitles(image_path: str, text: str, font_size: int = 28, y_offset: int = 40) -> str:
    """Burn subtitle text into an image using PIL, return new image path."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = None
    for font_path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - tw) // 2
    y = img.height - th - y_offset

    padding = 6
    draw.rectangle(
        [x - padding, y - padding, x + tw + padding, y + th + padding],
        fill=(0, 0, 0, 180),
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    result = Image.alpha_composite(img, overlay).convert("RGB")
    out_path = image_path.rsplit(".", 1)[0] + "_sub.png"
    result.save(out_path, quality=95)
    logger.info("_burn_subtitles: %s -> %s", image_path, out_path)
    return out_path


class SceneVideoAssembler(BaseNode):
    RETURN_TYPES = ("VIDEO_PATH",)
    CATEGORY = "fusion-mlx/drama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "audio_path": ("STRING", {"default": ""}),
                "subtitle_text": ("STRING", {"default": ""}),
                "subtitle_font_size": ("INT", {"default": 28, "min": 12, "max": 72}),
                "subtitle_y_offset": ("INT", {"default": 40, "min": 0, "max": 200}),
                "duration": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0}),
            }
        }

    async def execute(
        self, video_path, audio_path="", subtitle_text="",
        subtitle_font_size=28, subtitle_y_offset=40, duration=5.0,
    ):
        async with NodeTimer.timed("SceneVideoAssembler", "full"):
            output_dir = os.environ.get("FUSION_OUTPUT_DIR", "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"scene_{abs(hash(video_path)) % 100000}.mp4"
            )

            # For images: burn subtitles with PIL, then convert to video
            if _is_image(video_path):
                source_img = video_path
                if subtitle_text:
                    source_img = _burn_subtitles(
                        video_path, subtitle_text,
                        font_size=subtitle_font_size, y_offset=subtitle_y_offset,
                    )

                img_cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", source_img,
                    "-t", str(duration),
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-preset", "medium",
                    "-r", "24",
                    output_path,
                ]
                async with NodeTimer.timed("SceneVideoAssembler", "image_to_video"):
                    result = subprocess.run(img_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error("image->video ffmpeg failed: %s", result.stderr[:500])
                        raise RuntimeError(f"image->video failed: {result.stderr[:200]}")

                # Cleanup subtitle overlay image
                if source_img != video_path and os.path.exists(source_img):
                    os.unlink(source_img)

                logger.info("SceneVideoAssembler: image->video %s -> %s", video_path, output_path)
                return (output_path,)

            # For video files: add audio if present
            cmd = ["ffmpeg", "-y", "-i", video_path]
            if audio_path and os.path.exists(audio_path):
                cmd += ["-i", audio_path]
            cmd += [
                "-c:v", "libx264", "-preset", "medium",
                "-c:a", "aac", "-shortest", output_path,
            ]
            async with NodeTimer.timed("SceneVideoAssembler", "ffmpeg"):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error("SceneVideoAssembler ffmpeg failed: %s", result.stderr[:500])
                    raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")

            logger.info("SceneVideoAssembler: output=%s", output_path)
            return (output_path,)


class ChapterVideoConcat(BaseNode):
    RETURN_TYPES = ("VIDEO_PATH",)
    CATEGORY = "fusion-mlx/drama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_paths": ("STRING", {"default": ""}),
                "chapter_title": ("STRING", {"default": ""}),
            }
        }

    async def execute(self, video_paths, chapter_title=""):
        async with NodeTimer.timed("ChapterVideoConcat", "full", chapter=chapter_title):
            output_dir = os.environ.get("FUSION_OUTPUT_DIR", "output")
            os.makedirs(output_dir, exist_ok=True)
            safe_title = chapter_title.replace(" ", "_") if chapter_title else "chapter"
            output_path = os.path.join(output_dir, f"{safe_title}.mp4")

            paths = [p.strip() for p in video_paths.split(",") if p.strip()]
            if not paths:
                raise ValueError("ChapterVideoConcat: no video paths provided")

            for p in paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"ChapterVideoConcat: missing {p}")

            concat_path = output_path.replace(".mp4", "_concat.txt")
            with open(concat_path, "w", encoding="utf-8") as f:
                for p in paths:
                    abs_p = os.path.abspath(p)
                    f.write(f"file '{abs_p}'\n")

            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_path,
                "-c:v", "libx264", "-preset", "medium",
                "-c:a", "aac", output_path,
            ]

            async with NodeTimer.timed("ChapterVideoConcat", "ffmpeg_concat"):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error("ChapterVideoConcat ffmpeg failed: %s", result.stderr)
                    raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

            if os.path.exists(concat_path):
                os.unlink(concat_path)

            logger.info("ChapterVideoConcat: output=%s (%d scenes)", output_path, len(paths))
            return (output_path,)
