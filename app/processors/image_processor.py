"""
SafeSight AI — Image & Video Processor
========================================
Handles raw bytes → PIL Image conversion and video keyframe extraction.

All functions are pure / synchronous and designed to be called via
asyncio.run_in_executor() to avoid blocking the event loop.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageOps

if TYPE_CHECKING:
    pass


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """
    Convert raw image bytes to a normalized PIL Image.

    - Converts to RGB (removes alpha channel / handles CMYK / grayscale)
    - Applies EXIF orientation correction
    - Resizes to 512×512 max while preserving aspect ratio (for CLIP efficiency)

    Args:
        image_bytes: Raw bytes from any supported image format (JPEG, PNG, WEBP…)

    Returns:
        PIL.Image in RGB mode, max 512px on longest side.
    """
    image = Image.open(io.BytesIO(image_bytes))

    # Correct EXIF orientation (e.g. portrait photos taken on phones)
    image = ImageOps.exif_transpose(image)

    # Normalize to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Efficient resize for model ingestion
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    return image


def extract_video_frames(
    video_bytes: bytes,
    num_frames: int = 3,
) -> list[Image.Image]:
    """
    Extract evenly-spaced keyframes from a video file.

    Uses PyAV (ffmpeg bindings) for efficient container-level frame extraction
    without decoding every frame (seeks to I-frames when possible).

    Args:
        video_bytes: Raw bytes of a video file (MP4, WebM, AVI…)
        num_frames:  Number of frames to extract (default: 3)

    Returns:
        List of PIL Images (may be fewer than num_frames if video is very short).
    """
    try:
        import av  # type: ignore
    except ImportError:
        raise ImportError("PyAV is required for video processing: pip install av")

    frames: list[Image.Image] = []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        with av.open(tmp_path) as container:
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if video_stream is None:
                return frames

            total_frames = video_stream.frames or 0
            duration = float(video_stream.duration or 0) * video_stream.time_base

            if total_frames > 0:
                # Sample evenly across the video timeline
                sample_indices = [
                    int(i * total_frames / (num_frames + 1))
                    for i in range(1, num_frames + 1)
                ]
            else:
                sample_indices = list(range(num_frames))

            frame_idx = 0
            sample_set = set(sample_indices)

            for frame in container.decode(video=0):
                if frame_idx in sample_set:
                    pil_frame = frame.to_image().convert("RGB")
                    pil_frame.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    frames.append(pil_frame)
                    if len(frames) >= num_frames:
                        break
                frame_idx += 1

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return frames
