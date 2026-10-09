"""
SafeSight AI — Advanced Error Level Analysis (ELA) Detector
============================================================
A specialized computer vision utility for digital image forensics.

Error Level Analysis (ELA) works by intentionally re-saving an image at a
known high JPEG quality level (default 95%) and computing the pixel-wise
absolute difference against the original image.

Key Principles:
  1. Real Photos: Have undergone natural compression; the difference map shows
     consistent, low-level error across all natural textures (paper, table, background).
  2. Synthetic / AI-Generated Images (Diffusion / GANs): Often lack standard camera
     sensor noise and authentic multi-generation JPEG quantization tables, resulting in
     either an abnormally uniform "glow" or sudden frequency boundary anomalies.
  3. Digital Inpainting / Overlaid Text: Text digitally placed onto real paper exhibits
     drastically different error compression levels than the underlying paper fibers.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from app.logging_config import get_logger

logger = get_logger("ela_detector")


@dataclass
class ELAResult:
    """
    Structured outcome of Error Level Analysis on an image.
    """
    ela_image: Image.Image
    ela_bytes: bytes
    ela_base64: str
    mean_error: float
    max_error: float
    variance: float
    high_error_ratio: float
    glow_uniformity_score: float
    quadrant_variances: dict[str, float]
    text_vs_background_discrepancy: float
    is_synthetic_suspicious: bool
    synthetic_likelihood: float
    summary_description: str
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_error": round(self.mean_error, 2),
            "max_error": round(self.max_error, 2),
            "variance": round(self.variance, 2),
            "high_error_ratio": round(self.high_error_ratio, 4),
            "glow_uniformity_score": round(self.glow_uniformity_score, 3),
            "quadrant_variances": {k: round(v, 2) for k, v in self.quadrant_variances.items()},
            "text_vs_background_discrepancy": round(self.text_vs_background_discrepancy, 3),
            "is_synthetic_suspicious": self.is_synthetic_suspicious,
            "synthetic_likelihood": round(self.synthetic_likelihood, 3),
            "summary_description": self.summary_description,
            "latency_ms": round(self.latency_ms, 2),
        }


def perform_error_level_analysis(
    image_input: Image.Image | bytes,
    quality: int = 95,
    scale: float = 10.0,
    high_error_threshold: int = 35,
) -> ELAResult:
    """
    Execute Error Level Analysis on a PIL Image or raw bytes.

    Args:
        image_input: PIL Image or image bytes.
        quality: JPEG compression quality for re-saving (default: 95).
        scale: Enhancement multiplier for difference amplification (default: 10.0).
        high_error_threshold: Pixel intensity threshold to count as high error.

    Returns:
        ELAResult containing the ELA visualization map, numerical metrics,
        and forensic anomaly indicators.
    """
    t0 = time.perf_counter()

    # 1. Normalize input to RGB PIL Image
    if isinstance(image_input, bytes):
        orig_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        orig_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    w, h = orig_img.size

    # 2. Re-save at known target JPEG quality into memory
    recompressed_buffer = io.BytesIO()
    orig_img.save(recompressed_buffer, format="JPEG", quality=quality)
    recompressed_buffer.seek(0)
    recompressed_img = Image.open(recompressed_buffer).convert("RGB")

    # 3. Calculate pixel-wise absolute difference
    diff = ImageChops.difference(orig_img, recompressed_img)

    # 4. Amplify difference for visual inspection and feature extraction
    diff_enhanced = ImageEnhance.Brightness(diff).enhance(scale)

    # Convert difference to numpy array for fast vectorized statistics
    diff_arr = np.array(diff_enhanced, dtype=np.float32)
    gray_diff = np.mean(diff_arr, axis=2)  # 2D luminance error map

    # 5. Compute global error metrics
    mean_error = float(np.mean(gray_diff))
    max_error = float(np.max(gray_diff))
    variance = float(np.var(gray_diff))

    high_error_pixels = int(np.sum(gray_diff > high_error_threshold))
    total_pixels = w * h
    high_error_ratio = float(high_error_pixels / max(1, total_pixels))

    # 6. Spatial Quadrant Analysis (detect localized edits / overlays)
    quadrant_variances: dict[str, float] = {}
    mid_x, mid_y = w // 2, h // 2
    quadrants = {
        "top-left": gray_diff[0:mid_y, 0:mid_x],
        "top-right": gray_diff[0:mid_y, mid_x:w],
        "bottom-left": gray_diff[mid_y:h, 0:mid_x],
        "bottom-right": gray_diff[mid_y:h, mid_x:w],
        "center": gray_diff[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4],
    }

    for q_name, q_arr in quadrants.items():
        if q_arr.size > 0:
            quadrant_variances[q_name] = float(np.var(q_arr))
        else:
            quadrant_variances[q_name] = 0.0

    # 7. Foreground Text vs Background Paper Discrepancy
    # High-frequency text edges vs low-frequency background
    orig_gray = np.array(orig_img.convert("L"), dtype=np.float32)
    grad_y, grad_x = np.gradient(orig_gray)
    gradient_mag = np.hypot(grad_x, grad_y)

    text_mask = gradient_mag > np.percentile(gradient_mag, 85)
    bg_mask = gradient_mag < np.percentile(gradient_mag, 40)

    if np.sum(text_mask) > 100 and np.sum(bg_mask) > 100:
        text_ela_mean = float(np.mean(gray_diff[text_mask]))
        bg_ela_mean = float(np.mean(gray_diff[bg_mask]))
        # Ratio of error in text regions relative to background paper
        text_vs_background_discrepancy = float(
            abs(text_ela_mean - bg_ela_mean) / (bg_ela_mean + 1e-5)
        )
    else:
        text_vs_background_discrepancy = 0.0

    # 8. Diffusion / 3D Render Glow Uniformity Score
    # Synthetic diffusion models often generate uniformly smooth or uniformly elevated
    # high-frequency noise profiles without camera sensor noise distributions.
    quad_var_vals = list(quadrant_variances.values())
    quad_std = float(np.std(quad_var_vals)) if quad_var_vals else 0.0
    quad_mean = float(np.mean(quad_var_vals)) if quad_var_vals else 1.0
    # Coefficient of variation across regions:
    # Extremely low variation across all regions suggests synthetic generation
    glow_uniformity_score = float(1.0 - min(1.0, quad_std / (quad_mean + 1e-5)))

    # 9. Synthetic Likelihood & Heuristic Assessment
    synthetic_likelihood = 0.0
    anomaly_notes: list[str] = []

    # Heuristic A: Unnaturally smooth ELA (diffusion signature)
    if variance < 45.0:
        synthetic_likelihood += 0.40
        anomaly_notes.append(
            f"Abnormally low compression variance ({variance:.1f}), consistent with diffusion synthesis."
        )

    # Heuristic B: Discrepancy between text strokes and background paper
    if text_vs_background_discrepancy > 1.25:
        synthetic_likelihood += 0.35
        anomaly_notes.append(
            f"Elevated text-to-background compression discrepancy ({text_vs_background_discrepancy:.2f}), "
            "indicating digital text overlay on simulated paper."
        )

    # Heuristic C: Global synthetic glow uniformity
    if glow_uniformity_score > 0.85 and high_error_ratio > 0.12:
        synthetic_likelihood += 0.25
        anomaly_notes.append(
            f"Uniform high-frequency compression glow (uniformity={glow_uniformity_score:.2f})."
        )

    synthetic_likelihood = min(1.0, max(0.0, synthetic_likelihood))
    is_suspicious = synthetic_likelihood >= 0.50

    if not anomaly_notes:
        summary_description = (
            f"ELA map shows natural compression consistency (mean error={mean_error:.1f}, "
            f"variance={variance:.1f}). No anomalous digital overlays or synthetic glow detected."
        )
    else:
        summary_description = " ".join(anomaly_notes)

    # 10. Generate ELA artifact bytes and base64 string
    ela_out_buffer = io.BytesIO()
    diff_enhanced.save(ela_out_buffer, format="JPEG", quality=quality)
    ela_bytes = ela_out_buffer.getvalue()
    ela_base64 = base64.b64encode(ela_bytes).decode("utf-8")

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.debug(
        "ELA completed",
        mean_error=round(mean_error, 2),
        variance=round(variance, 2),
        discrepancy=round(text_vs_background_discrepancy, 3),
        synthetic_likelihood=round(synthetic_likelihood, 2),
        latency_ms=round(latency_ms, 2),
    )

    return ELAResult(
        ela_image=diff_enhanced,
        ela_bytes=ela_bytes,
        ela_base64=ela_base64,
        mean_error=mean_error,
        max_error=max_error,
        variance=variance,
        high_error_ratio=high_error_ratio,
        glow_uniformity_score=glow_uniformity_score,
        quadrant_variances=quadrant_variances,
        text_vs_background_discrepancy=text_vs_background_discrepancy,
        is_synthetic_suspicious=is_suspicious,
        synthetic_likelihood=synthetic_likelihood,
        summary_description=summary_description,
        latency_ms=latency_ms,
    )


async def perform_ela_async(
    image_input: Image.Image | bytes,
    quality: int = 95,
    scale: float = 10.0,
    high_error_threshold: int = 35,
) -> ELAResult:
    """
    Non-blocking async wrapper for Error Level Analysis.
    Offloads CPU-bound array transformations to default thread pool.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        perform_error_level_analysis,
        image_input,
        quality,
        scale,
        high_error_threshold,
    )
