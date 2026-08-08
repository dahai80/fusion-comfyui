#!/usr/bin/env python3
import logging
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("make_icon")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(SCRIPT_DIR, "AppIcon.iconset")
ICNS_PATH = os.path.join(SCRIPT_DIR, "AppIcon.icns")

SIZES = [
    (16, 1), (16, 2),
    (32, 1), (32, 2),
    (128, 1), (128, 2),
    (256, 1), (256, 2),
    (512, 1), (512, 2),
]


def render(px: int) -> Image.Image:
    s = px
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(s * 0.22)
    pad = int(s * 0.05)
    box = [pad, pad, s - pad, s - pad]
    d.rounded_rectangle(box, radius=r, fill=(26, 27, 38, 255))
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gp = int(s * 0.16)
    gd.ellipse([s * 0.55, -gp, s + gp, s * 0.5], fill=(255, 138, 80, 180))
    gd.ellipse([-gp, s * 0.55, s * 0.45, s + gp], fill=(96, 165, 250, 160))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=int(s * 0.08)))
    img = Image.alpha_composite(img, glow)
    d = ImageDraw.Draw(img)
    bw = max(2, int(s * 0.075))
    cx, cy = s * 0.5, s * 0.5
    h = int(s * 0.46)
    top = cy - h / 2
    mid = cy
    d.line([cx, top, cx, top + h], fill=(240, 240, 245, 255), width=bw)
    d.line([cx, top, cx + h * 0.42, top], fill=(240, 240, 245, 255), width=bw)
    d.line([cx, mid - h * 0.08, cx + h * 0.34, mid - h * 0.08],
           fill=(240, 240, 245, 255), width=bw)
    return img


def main() -> int:
    os.makedirs(ASSET_DIR, exist_ok=True)
    for base, scale in SIZES:
        px = base * scale
        img = render(px)
        if scale == 1:
            name = f"icon_{base}x{base}.png"
        else:
            name = f"icon_{base}x{base}@2x.png"
        out = os.path.join(ASSET_DIR, name)
        img.save(out)
        log.info("wrote %s (%dx%d)", name, px, px)
    res = subprocess.run(
        ["iconutil", "-c", "icns", ASSET_DIR, "-o", ICNS_PATH],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        log.error("iconutil failed: %s", res.stderr.strip())
        return 1
    log.info("icns ready: %s (%d bytes)", ICNS_PATH, os.path.getsize(ICNS_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
