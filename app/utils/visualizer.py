"""
SafeSight AI — Visual Forensics Bounding Box Overlay Utility
============================================================
Draws precision bounding-box overlays, tactical corner brackets, and
confidence badge pills onto media images to visually highlight localized
AI-generated regions, inpainting, and synthetic manipulations.

Features:
  - Supports PIL Images, file paths, and raw image inputs
  - High-visibility color-coding:
      * Red (#FF3B30)   -> AI-generated media, deepfake faces, synthetic text
      * Amber (#FFCC00) -> Inpainting, localized edits, lower confidence tampering
  - Clean badge pills with region label and calibrated confidence percentage
  - Automatic directory creation under 'static/reports/annotated_<filename>.png'
  - Return resolved filesystem Path for CLI and API consumers
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Union

from PIL import Image, ImageDraw, ImageFont

from app.logging_config import get_logger
from app.schemas import DetectedTamperedRegion

logger = get_logger("visualizer")

# Default reports directory
DEFAULT_REPORTS_DIR = Path("static/reports")

# Color palette (RGB)
COLOR_RED = (255, 59, 48)       # High-severity synthetic / deepfake
COLOR_AMBER = (255, 204, 0)     # Partial edit / inpainting
COLOR_TEXT_LIGHT = (255, 255, 255)
COLOR_TEXT_DARK = (20, 20, 20)


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Load default truetype font or fall back to system bitmap font."""
    font_names = [
        "arialbd.ttf",
        "segoeuib.ttf",
        "DejaVuSans-Bold.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_annotated_tampering_image(
    image_input: Union[Path, str, Image.Image],
    tampered_regions: list[DetectedTamperedRegion],
    output_path: Path | str | None = None,
    box_width: int = 4,
) -> Path:
    """
    Render bounding boxes and label badges onto the image for all detected tampered regions.

    Args:
        image_input: Path to original image or PIL Image object.
        tampered_regions: List of DetectedTamperedRegion objects with box_2d in [ymin, xmin, ymax, xmax] 0-1000 scale.
        output_path: Target save path. If None, saves to 'static/reports/annotated_<filename>.png'.
        box_width: Thickness of bounding box borders.

    Returns:
        Resolved Path to the saved annotated image.
    """
    # 1. Normalize image to RGB PIL Image
    if isinstance(image_input, (str, Path)):
        src_path = Path(image_input)
        img = Image.open(src_path).convert("RGB")
        base_name = src_path.name
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        base_name = f"scan_{int(time.time() * 1000)}.png"
    else:
        raise ValueError(f"Unsupported image_input type: {type(image_input)}")

    w, h = img.size

    # 2. Determine target output path
    if output_path is None:
        clean_name = base_name if not base_name.startswith("annotated_") else base_name[10:]
        out_file = DEFAULT_REPORTS_DIR / f"annotated_{clean_name}"
    else:
        out_file = Path(output_path)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # If no tampered regions exist, save a copy with no boxes
    if not tampered_regions:
        img.save(out_file, format="PNG")
        logger.debug("No tampered regions to draw; saved original copy", path=str(out_file))
        return out_file.resolve()

    # 3. Create drawing canvas with RGBA overlay for smooth alpha blending
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(img)

    font_size = max(12, int(min(w, h) * 0.022))
    font = _get_font(font_size)

    # 4. Iterate and render each region
    for region in tampered_regions:
        box = region.box_2d
        if not box or len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = box

        # Convert normalized 0-1000 scale to absolute pixel dimensions
        x1 = int(max(0, min(1000, xmin)) / 1000.0 * w)
        y1 = int(max(0, min(1000, ymin)) / 1000.0 * h)
        x2 = int(max(0, min(1000, xmax)) / 1000.0 * w)
        y2 = int(max(0, min(1000, ymax)) / 1000.0 * h)

        # Enforce valid box orientation
        if x2 <= x1:
            x2 = min(w, x1 + 20)
        if y2 <= y1:
            y2 = min(h, y1 + 20)

        # Select color based on category and severity
        label_lower = region.region_label.lower()
        is_high_risk = (
            "synthetic" in label_lower
            or "deepfake" in label_lower
            or "adversarial" in label_lower
            or region.confidence >= 0.85
        )
        box_rgb = COLOR_RED if is_high_risk else COLOR_AMBER
        tint_rgba = (*box_rgb, 35)  # 14% translucent fill tint

        # Draw translucent interior tint
        draw_overlay.rectangle([x1, y1, x2, y2], fill=tint_rgba)

        # Draw outer bounding box border
        for offset in range(box_width):
            draw_base.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=box_rgb,
            )

        # Draw tactical corner brackets
        corner_len = max(10, int(min(x2 - x1, y2 - y1) * 0.18))
        corner_w = box_width + 2
        # Top-left corner
        draw_base.line([(x1, y1), (x1 + corner_len, y1)], fill=box_rgb, width=corner_w)
        draw_base.line([(x1, y1), (x1, y1 + corner_len)], fill=box_rgb, width=corner_w)
        # Top-right corner
        draw_base.line([(x2, y1), (x2 - corner_len, y1)], fill=box_rgb, width=corner_w)
        draw_base.line([(x2, y1), (x2, y1 + corner_len)], fill=box_rgb, width=corner_w)
        # Bottom-left corner
        draw_base.line([(x1, y2), (x1 + corner_len, y2)], fill=box_rgb, width=corner_w)
        draw_base.line([(x1, y2), (x1, y2 - corner_len)], fill=box_rgb, width=corner_w)
        # Bottom-right corner
        draw_base.line([(x2, y2), (x2 - corner_len, y2)], fill=box_rgb, width=corner_w)
        draw_base.line([(x2, y2), (x2, y2 - corner_len)], fill=box_rgb, width=corner_w)

        # Prepare badge text
        badge_text = f"{region.region_label.replace('_', ' ').upper()} · {int(region.confidence * 100)}%"

        # Measure text size
        try:
            bbox = font.getbbox(badge_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(badge_text) * (font_size // 2)
            text_h = font_size

        pad_x, pad_y = 6, 4
        badge_w = text_w + (pad_x * 2)
        badge_h = text_h + (pad_y * 2)

        # Determine badge placement (above box if space permits, else inside top)
        if y1 - badge_h - 4 >= 0:
            badge_y1 = y1 - badge_h - 2
            badge_y2 = y1 - 2
        else:
            badge_y1 = y1 + 2
            badge_y2 = y1 + badge_h + 2

        badge_x1 = max(0, min(w - badge_w, x1))
        badge_x2 = badge_x1 + badge_w

        # Draw badge background pill
        draw_base.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=box_rgb)

        # Draw badge label text
        text_color = COLOR_TEXT_LIGHT if is_high_risk else COLOR_TEXT_DARK
        draw_base.text(
            (badge_x1 + pad_x, badge_y1 + pad_y),
            badge_text,
            fill=text_color,
            font=font,
        )

    # 5. Composite overlay with base image
    img_rgba = img.convert("RGBA")
    composited = Image.alpha_composite(img_rgba, overlay).convert("RGB")

    # 6. Save final annotated image
    composited.save(out_file, format="PNG")
    logger.info(
        "Rendered visual forensic tampering map",
        path=str(out_file),
        regions=len(tampered_regions),
    )
    return out_file.resolve()
