import logging
import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("fusion_comfyui.nodes.postprocess")


class FusionSubtitleOverlayNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "font_size": ("INT", {"default": 36, "min": 12, "max": 120}),
                "position": (["bottom", "top", "center"], {"default": "bottom"}),
                "margin": ("INT", {"default": 40, "min": 0, "max": 200}),
                "font_color": ("STRING", {"default": "white"}),
                "stroke_color": ("STRING", {"default": "black"}),
                "stroke_width": ("INT", {"default": 2, "min": 0, "max": 10}),
                "bg_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_width_ratio": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "font_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "overlay"
    CATEGORY = "Fusion-MLX/PostProcess"

    def overlay(self, images, text, font_size=36, position="bottom", margin=40,
                font_color="white", stroke_color="black", stroke_width=2,
                bg_opacity=0.5, max_width_ratio=0.9, font_path=""):
        from core.bridge import to_numpy

        if not text:
            return (images,)

        arr = to_numpy(images).astype(np.float32)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        font = self._load_font(font_path, font_size)
        color_rgb = self._parse_color(font_color)
        stroke_rgb = self._parse_color(stroke_color)

        result_frames = []
        for i, frame in enumerate(arr):
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            elif frame.ndim == 3 and frame.shape[0] in (3, 4):
                frame = frame.transpose(1, 2, 0)[:, :, :3]

            img = Image.fromarray(frame)
            img = self._draw_subtitle(img, text, font, color_rgb, stroke_rgb,
                                      stroke_width, position, margin, bg_opacity,
                                      max_width_ratio)
            result_frames.append(np.array(img).astype(np.float32) / 255.0)

        result = np.stack(result_frames, axis=0)
        logger.info(
            "FusionSubtitleOverlay: text='%s...' frames=%d output=%s",
            text[:30], len(result_frames), result.shape,
        )
        return (result,)

    def _load_font(self, font_path, font_size):
        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception as e:
                logger.warning("FusionSubtitleOverlay: font load failed: %s, using default", e)

        for path in [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except Exception:
                    continue

        try:
            return ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            logger.warning("FusionSubtitleOverlay: no truetype font found, using default")
            return ImageFont.load_default()

    def _parse_color(self, color_str):
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
        }
        color_str = color_str.strip().lower()
        if color_str in color_map:
            return color_map[color_str]
        if color_str.startswith("#") and len(color_str) == 7:
            return tuple(int(color_str[i:i+2], 16) for i in (1, 3, 5))
        return (255, 255, 255)

    def _draw_subtitle(self, img, text, font, color_rgb, stroke_rgb,
                        stroke_width, position, margin, bg_opacity,
                        max_width_ratio):
        draw = ImageDraw.Draw(img, "RGBA")
        w, h = img.size
        max_text_w = int(w * max_width_ratio)

        lines = textwrap.wrap(text, width=max(10, int(max_text_w / (font.size * 0.6))))
        if not lines:
            return img.convert("RGB")

        line_spacing = font.size // 3
        line_heights = []
        for line in lines:
            bbox = font.getbbox(line)
            line_h = bbox[3] - bbox[1]
            line_heights.append(line_h)

        total_h = sum(line_heights) + line_spacing * (len(lines) - 1)
        total_w = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines)

        if position == "bottom":
            y_start = h - total_h - margin * 2
        elif position == "top":
            y_start = margin
        else:
            y_start = (h - total_h) // 2

        if bg_opacity > 0:
            bg_box = [
                (w - total_w) // 2 - 10,
                y_start - 10,
                (w + total_w) // 2 + 10,
                y_start + total_h + 10,
            ]
            bg_color = (0, 0, 0, int(255 * bg_opacity))
            draw.rectangle(bg_box, fill=bg_color)

        y = y_start
        for i, line in enumerate(lines):
            bbox = font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            x = (w - line_w) // 2

            if stroke_width > 0:
                for dx in range(-stroke_width, stroke_width + 1):
                    for dy in range(-stroke_width, stroke_width + 1):
                        if dx * dx + dy * dy <= stroke_width * stroke_width:
                            draw.text((x + dx, y + dy), line, fill=stroke_rgb + (255,), font=font)

            draw.text((x, y), line, fill=color_rgb + (255,), font=font)
            y += line_heights[i] + line_spacing

        return img.convert("RGB")
