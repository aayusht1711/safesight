"""
SafeSight AI — Advanced Differential Error Level Analysis (ELA) Detector
==========================================================================
A specialized signal-processing and computer vision utility for digital image forensics.

Upgraded to Differential ELA (Compression-Aware Multi-Stage Analysis):
  1. Primary Pass (High Quality 95%): Re-saves at Q95 to quantify high-frequency noise variance.
  2. Secondary Pass (Standard Quality 75%): Re-saves at Q75 matching social media platforms.
  3. Differential Map (|ELA_95 - ELA_75|):
     - In authentic photos (even heavily compressed via WhatsApp/Facebook), error levels
       between 95% and 75% are spatially uniform across natural background textures.
     - In tampered photos where an uncompressed or generative AI element (e.g. pasted birds,
       synthetic face, inpainting patch) has been composited onto a compressed background,
       the pasted object undergoes dramatic first-generation quantization loss and lights up
       starkly in the differential map while the authentic background remains dark.
  4. Compression Awareness: Integrates with compression fingerprinting so that benign
     WhatsApp JPEG blockiness is never misdiagnosed as malicious generative AI.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from app.logging_config import get_logger
from app.utils.compression_fingerprint import (
    CompressionFingerprint,
    get_image_fingerprint,
)

logger = get_logger("ela_detector")


@dataclass
class ELAResult:
    """
    Structured outcome of Differential Error Level Analysis on an image.
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
    # Differential ELA attributes
    ela_75_image: Image.Image | None = None
    differential_image: Image.Image | None = None
    differential_bytes: bytes | None = None
    differential_base64: str | None = None
    differential_variance: float = 0.0
    differential_mean: float = 0.0
    splicing_anomaly_score: float = 0.0
    splicing_detected: bool = False
    splicing_region_box: list[int] | None = None
    fingerprint: CompressionFingerprint | None = None
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
            "differential_variance": round(self.differential_variance, 2),
            "differential_mean": round(self.differential_mean, 2),
            "splicing_anomaly_score": round(self.splicing_anomaly_score, 2),
            "splicing_detected": self.splicing_detected,
            "splicing_region_box": self.splicing_region_box,
            "fingerprint_label": self.fingerprint.fingerprint_label if self.fingerprint else None,
            "latency_ms": round(self.latency_ms, 2),
        }


def perform_error_level_analysis(
    image_input: Union[Image.Image, bytes],
    quality: int = 95,
    quality_low: int = 75,
    scale: float = 10.0,
    high_error_threshold: int = 35,
    fingerprint: CompressionFingerprint | None = None,
) -> ELAResult:
    """
    Execute Differential Error Level Analysis (95% vs 75% quality passes).

    Args:
        image_input: PIL Image or image bytes.
        quality: High JPEG compression quality for primary ELA pass (default: 95).
        quality_low: Secondary JPEG quality for differential pass (default: 75).
        scale: Enhancement multiplier for difference amplification (default: 10.0).
        high_error_threshold: Pixel intensity threshold to count as high error.
        fingerprint: Optional pre-computed CompressionFingerprint. If omitted, computed on the fly.

    Returns:
        ELAResult with primary 95% ELA map, differential ELA map, and splicing detection metrics.
    """
    t0 = time.perf_counter()

    # 1. Extract or reuse compression fingerprint from raw image_input
    if fingerprint is None:
        fingerprint = get_image_fingerprint(image_input)

    # 2. Normalize input to RGB PIL Image
    if isinstance(image_input, bytes):
        raw_bytes = image_input
        orig_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        # Preserve DQT tables on orig_img if present
        orig_quant = getattr(image_input, "quantization", None)
        orig_format = getattr(image_input, "format", None)
        orig_img = image_input.convert("RGB")
        if orig_quant:
            orig_img.quantization = orig_quant
        if orig_format:
            orig_img.format = orig_format
        buf = io.BytesIO()
        orig_img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    w, h = orig_img.size

    # 3. High-Quality ELA Pass (Quality = 95)
    buf_95 = io.BytesIO()
    orig_img.save(buf_95, format="JPEG", quality=quality)
    buf_95.seek(0)
    recompressed_95 = Image.open(buf_95).convert("RGB")
    diff_95 = ImageChops.difference(orig_img, recompressed_95)
    diff_95_enhanced = ImageEnhance.Brightness(diff_95).enhance(scale)

    diff_arr_95 = np.array(diff_95_enhanced, dtype=np.float32)
    gray_diff_95 = np.mean(diff_arr_95, axis=2)  # 2D luminance error map

    # 4. Low-Quality ELA Pass (Quality = 75, WhatsApp standard baseline)
    buf_75 = io.BytesIO()
    orig_img.save(buf_75, format="JPEG", quality=quality_low)
    buf_75.seek(0)
    recompressed_75 = Image.open(buf_75).convert("RGB")
    diff_75 = ImageChops.difference(orig_img, recompressed_75)
    diff_75_enhanced = ImageEnhance.Brightness(diff_75).enhance(scale)

    # 5. Differential ELA Map: Absolute difference between 95% and 75% error responses
    diff_differential = ImageChops.difference(diff_95, diff_75)
    diff_differential_enhanced = ImageEnhance.Brightness(diff_differential).enhance(scale * 1.5)

    diff_arr_diff = np.array(diff_differential_enhanced, dtype=np.float32)
    gray_diff_delta = np.mean(diff_arr_diff, axis=2)  # 2D differential error map

    # 6. Compute Global Primary ELA Metrics
    mean_error = float(np.mean(gray_diff_95))
    max_error = float(np.max(gray_diff_95))
    variance = float(np.var(gray_diff_95))

    high_error_pixels = int(np.sum(gray_diff_95 > high_error_threshold))
    total_pixels = w * h
    high_error_ratio = float(high_error_pixels / max(1, total_pixels))

    # 7. Spatial Quadrant Analysis (95% pass)
    quadrant_variances: dict[str, float] = {}
    mid_x, mid_y = w // 2, h // 2
    quadrants = {
        "top-left": gray_diff_95[0:mid_y, 0:mid_x],
        "top-right": gray_diff_95[0:mid_y, mid_x:w],
        "bottom-left": gray_diff_95[mid_y:h, 0:mid_x],
        "bottom-right": gray_diff_95[mid_y:h, mid_x:w],
        "center": gray_diff_95[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4],
    }
    for q_name, q_arr in quadrants.items():
        quadrant_variances[q_name] = float(np.var(q_arr)) if q_arr.size > 0 else 0.0

    # 8. Foreground Text vs Background Paper Discrepancy
    orig_gray = np.array(orig_img.convert("L"), dtype=np.float32)
    grad_y, grad_x = np.gradient(orig_gray)
    gradient_mag = np.hypot(grad_x, grad_y)

    text_mask = gradient_mag > np.percentile(gradient_mag, 85)
    bg_mask = gradient_mag < np.percentile(gradient_mag, 40)

    if np.sum(text_mask) > 100 and np.sum(bg_mask) > 100:
        text_ela_mean = float(np.mean(gray_diff_95[text_mask]))
        bg_ela_mean = float(np.mean(gray_diff_95[bg_mask]))
        text_vs_background_discrepancy = float(
            abs(text_ela_mean - bg_ela_mean) / (bg_ela_mean + 1e-5)
        )
    else:
        text_vs_background_discrepancy = 0.0

    # 9. Diffusion / 3D Render Glow Uniformity Score
    quad_var_vals = list(quadrant_variances.values())
    quad_std = float(np.std(quad_var_vals)) if quad_var_vals else 0.0
    quad_mean = float(np.mean(quad_var_vals)) if quad_var_vals else 1.0
    glow_uniformity_score = float(1.0 - min(1.0, quad_std / (quad_mean + 1e-5)))

    # 10. Differential ELA Signal Analysis (Splicing Detection)
    diff_mean = float(np.mean(gray_diff_delta))
    diff_var = float(np.var(gray_diff_delta))
    diff_max = float(np.max(gray_diff_delta))

    # Evaluate spatial concentration of differential error
    diff_bg_median = float(np.median(gray_diff_delta))
    diff_peak_p99 = float(np.percentile(gray_diff_delta, 99.0))
    splicing_anomaly_score = float((diff_peak_p99 - diff_bg_median) / (diff_bg_median + 1.0))

    splicing_detected = False
    splicing_box: list[int] | None = None

    # High differential threshold: localized splices concentrate in the upper tail (98.5th percentile)
    thresh = max(8.0, float(np.percentile(gray_diff_delta, 98.5)))
    anomaly_coords = np.where(gray_diff_delta >= thresh)

    num_anomaly_pixels = len(anomaly_coords[0])
    if num_anomaly_pixels >= max(20, int(total_pixels * 0.0002)):
        y_min_px, y_max_px = int(np.min(anomaly_coords[0])), int(np.max(anomaly_coords[0]))
        x_min_px, x_max_px = int(np.min(anomaly_coords[1])), int(np.max(anomaly_coords[1]))

        box_w = x_max_px - x_min_px
        box_h = y_max_px - y_min_px

        has_significant_differential = (diff_peak_p99 >= 7.0 or diff_max >= 15.0)
        is_compact_2d_cluster = (
            10 <= box_w < int(w * 0.70)
            and 10 <= box_h < int(h * 0.70)
            and (box_w * box_h) < int(total_pixels * 0.45)
        )

        if has_significant_differential and is_compact_2d_cluster and splicing_anomaly_score >= 1.5:
            splicing_detected = True
            ymin_norm = int(max(0, min(1000, (y_min_px / h) * 1000)))
            xmin_norm = int(max(0, min(1000, (x_min_px / w) * 1000)))
            ymax_norm = int(max(0, min(1000, (y_max_px / h) * 1000)))
            xmax_norm = int(max(0, min(1000, (x_max_px / w) * 1000)))
            splicing_box = [ymin_norm, xmin_norm, ymax_norm, xmax_norm]

    # 11. Compression-Aware Synthetic Likelihood Calibration
    synthetic_likelihood = 0.0
    anomaly_notes: list[str] = []

    is_social = fingerprint.is_social_media_compressed

    if is_social:
        # In social-media compressed images (WhatsApp/Facebook), expect blockiness & high error ratio.
        # Do NOT penalize global variance or high error ratio!
        if splicing_detected:
            synthetic_likelihood += 0.85
            anomaly_notes.append(
                f"Differential ELA detected high-contrast localized compression divergence "
                f"(splicing score: {splicing_anomaly_score:.2f}) in region {splicing_box}. "
                "Background exhibits standard social media compression, but spliced region indicates "
                "secondary digital insertion."
            )
        elif text_vs_background_discrepancy > 1.35:
            synthetic_likelihood += 0.40
            anomaly_notes.append(
                f"Elevated text-to-background discrepancy ({text_vs_background_discrepancy:.2f}) "
                "detected on top of social media compression."
            )
        else:
            synthetic_likelihood = 0.05
            anomaly_notes.append(
                f"Forensic analysis confirmed benign {fingerprint.fingerprint_label} compression artifacts. "
                "Differential ELA (95% vs 75%) shows consistent compression history across all image segments "
                "with no digital splicing or generative AI overlays."
            )
    else:
        # Standard camera or uncompressed profile
        if splicing_detected:
            synthetic_likelihood += 0.80
            anomaly_notes.append(
                f"Differential ELA localized a high-confidence digital splice at {splicing_box}."
            )
        if variance < 45.0:
            synthetic_likelihood += 0.35
            anomaly_notes.append(
                f"Abnormally low compression variance ({variance:.1f}), consistent with diffusion synthesis."
            )
        if text_vs_background_discrepancy > 1.25:
            synthetic_likelihood += 0.35
            anomaly_notes.append(
                f"Elevated text-to-background compression discrepancy ({text_vs_background_discrepancy:.2f})."
            )
        if glow_uniformity_score > 0.85 and high_error_ratio > 0.12:
            synthetic_likelihood += 0.25
            anomaly_notes.append(
                f"Uniform high-frequency compression glow (uniformity={glow_uniformity_score:.2f})."
            )

    synthetic_likelihood = min(1.0, max(0.0, synthetic_likelihood))
    is_suspicious = synthetic_likelihood >= 0.50
    summary_description = " ".join(anomaly_notes)

    # 12. Serialize Output Images (Primary 95% ELA and Differential ELA)
    ela_out_buf = io.BytesIO()
    diff_95_enhanced.save(ela_out_buf, format="JPEG", quality=quality)
    ela_bytes = ela_out_buf.getvalue()
    ela_base64 = base64.b64encode(ela_bytes).decode("utf-8")

    diff_out_buf = io.BytesIO()
    diff_differential_enhanced.save(diff_out_buf, format="JPEG", quality=quality)
    differential_bytes = diff_out_buf.getvalue()
    differential_base64 = base64.b64encode(differential_bytes).decode("utf-8")

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.debug(
        "Differential ELA completed",
        fingerprint=fingerprint.fingerprint_label,
        mean_error=round(mean_error, 2),
        differential_var=round(diff_var, 2),
        splicing_detected=splicing_detected,
        synthetic_likelihood=round(synthetic_likelihood, 2),
        latency_ms=round(latency_ms, 2),
    )

    return ELAResult(
        ela_image=diff_95_enhanced,
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
        ela_75_image=diff_75_enhanced,
        differential_image=diff_differential_enhanced,
        differential_bytes=differential_bytes,
        differential_base64=differential_base64,
        differential_variance=diff_var,
        differential_mean=diff_mean,
        splicing_anomaly_score=splicing_anomaly_score,
        splicing_detected=splicing_detected,
        splicing_region_box=splicing_box,
        fingerprint=fingerprint,
        latency_ms=latency_ms,
    )


async def perform_ela_async(
    image_input: Union[Image.Image, bytes],
    quality: int = 95,
    quality_low: int = 75,
    scale: float = 10.0,
    high_error_threshold: int = 35,
    fingerprint: CompressionFingerprint | None = None,
) -> ELAResult:
    """
    Non-blocking async wrapper for Differential Error Level Analysis.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        perform_error_level_analysis,
        image_input,
        quality,
        quality_low,
        scale,
        high_error_threshold,
        fingerprint,
    )
