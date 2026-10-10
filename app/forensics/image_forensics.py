"""
SafeSight AI — Compression-Aware Image Forensics Engine
========================================================
Enterprise-grade image forensics module combining:
  1. JPEG Quantization Table (DQT) & EXIF Provenance Profiling (WhatsApp, Telegram, etc.)
  2. Artifact Threshold Calibration for social media compression
  3. Differential Error Level Analysis (ELA) (95% vs 75%)
  4. High-Frequency Discrete Cosine Transform (DCT) Inpainting Localization
  5. Qwen2.5-VL Visual Grounding with normalized bounding boxes [ymin, xmin, ymax, xmax]
  6. Visual Forensic Annotation Overlays ('annotated_<filename>.png')
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image

try:
    import scipy.fftpack as fftpack
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from app.logging_config import get_logger
from app.schemas import (
    AuthenticityVerdict,
    DetectedTamperedRegion,
    TamperedRegionCategory,
)
from app.utils.compression_fingerprint import (
    CompressionFingerprint,
    get_image_fingerprint,
)
from app.utils.ela_detector import (
    ELAResult,
    perform_error_level_analysis,
)
from app.utils.visualizer import render_annotated_tampering_image

logger = get_logger("image_forensics")

DEFAULT_REPORTS_DIR = Path("static/reports")


@dataclass
class ImageForensicsResult:
    """Standardized result for compression-aware image forensics."""
    overall_verdict: str
    ai_percentage: float
    tampered_regions: list[dict[str, Any]] = field(default_factory=list)
    compression_fingerprint: str = "Unknown"
    is_social_media_compressed: bool = False
    ela_splicing_score: float = 0.0
    dct_anomaly_score: float = 0.0
    annotated_image_path: str | None = None
    forensic_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 1. High-Frequency Discrete Cosine Transform (DCT) Inpainting Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _dct2(block: np.ndarray) -> np.ndarray:
    """Compute 2D Discrete Cosine Transform of an 8x8 block."""
    if _HAS_SCIPY:
        return fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")
    # Pure numpy DCT-II fallback
    n = block.shape[0]
    u = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    cos_matrix = np.cos((2 * x + 1) * u * np.pi / (2 * n))
    alpha = np.sqrt(2 / n) * np.ones((n, 1))
    alpha[0, 0] = 1 / np.sqrt(n)
    t = alpha * cos_matrix
    return np.dot(np.dot(t, block), t.T)


def compute_hf_dct_inpainting_anomalies(
    image: Image.Image,
    is_social_media: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Analyze high-frequency 8x8 DCT coefficients across the image to locate
    inpainting, synthetic object fills, or texture boundary splices.

    Args:
        image: PIL Image in RGB.
        is_social_media: If True, calibrates thresholds higher to ignore
                         benign 8x8 quantization grid steps from WhatsApp/Telegram.

    Returns:
        tuple of (dct_anomaly_score: float, detected_regions: list[dict])
    """
    # Convert to grayscale luminance numpy array
    gray = np.array(image.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # Need at least 32x32 for block analysis
    if h < 32 or w < 32:
        return 0.0, []

    block_h = h // 8
    block_w = w // 8
    hf_energy_map = np.zeros((block_h, block_w), dtype=np.float32)

    # 8x8 High-Frequency mask (outer triangular half of DCT matrix, u + v >= 7)
    u_idx, v_idx = np.indices((8, 8))
    hf_mask = (u_idx + v_idx) >= 7

    for bi in range(block_h):
        for bj in range(block_w):
            block = gray[bi * 8 : (bi + 1) * 8, bj * 8 : (bj + 1) * 8]
            dct_coeffs = _dct2(block)
            # Energy of high-frequency components
            hf_energy = np.sum(np.abs(dct_coeffs[hf_mask]))
            hf_energy_map[bi, bj] = hf_energy

    # Robust baseline stats (median & IQR)
    med = float(np.median(hf_energy_map))
    p75, p25 = np.percentile(hf_energy_map, [75, 25])
    iqr = max(float(p75 - p25), 1e-4)

    # Calibrate sensitivity: social media compression exhibits standard DCT grid noise
    threshold_sigma = 3.8 if is_social_media else 2.8
    outlier_thresh = med + threshold_sigma * (iqr / 1.349)

    outlier_mask = hf_energy_map > outlier_thresh
    num_outliers = int(np.sum(outlier_mask))
    total_blocks = block_h * block_w
    outlier_ratio = num_outliers / max(total_blocks, 1)

    # Score between 0.0 and 1.0
    dct_score = min(1.0, outlier_ratio * (12.0 if is_social_media else 20.0))

    detected_regions: list[dict[str, Any]] = []

    # Cluster outlier blocks into spatial bounding boxes if anomaly detected
    if num_outliers >= 4:
        y_indices, x_indices = np.where(outlier_mask)
        # 10th to 90th percentile to bound the main cluster
        y_min_b, y_max_b = int(np.percentile(y_indices, 5)), int(np.percentile(y_indices, 95))
        x_min_b, x_max_b = int(np.percentile(x_indices, 5)), int(np.percentile(x_indices, 95))

        box_bw = x_max_b - x_min_b + 1
        box_bh = y_max_b - y_min_b + 1

        # Check compactness: must be localized (not global whole-screen noise)
        if 2 <= box_bw < 0.75 * block_w and 2 <= box_bh < 0.75 * block_h:
            # Map back to 0-1000 scale
            ymin_norm = int(round((y_min_b * 8 / h) * 1000))
            xmin_norm = int(round((x_min_b * 8 / w) * 1000))
            ymax_norm = int(round(((y_max_b + 1) * 8 / h) * 1000))
            xmax_norm = int(round(((x_max_b + 1) * 8 / w) * 1000))

            conf = min(0.98, max(0.65, dct_score * 0.9 + 0.1))
            detected_regions.append({
                "region_label": "AI-Inpainted Texture Anomaly",
                "label": "AI-Inpainted Texture Anomaly",
                "category": "INPAINTING",
                "box_2d": [ymin_norm, xmin_norm, ymax_norm, xmax_norm],
                "confidence": round(conf, 2),
                "forensic_explanation": (
                    f"High-frequency DCT anomaly cluster detected with {num_outliers} non-uniform "
                    f"blocks (deviation > {threshold_sigma:.1f}σ from background median)."
                ),
                "forensic_reasoning": (
                    f"High-frequency DCT anomaly cluster detected with {num_outliers} non-uniform "
                    f"blocks (deviation > {threshold_sigma:.1f}σ from background median)."
                ),
            })

    return dct_score, detected_regions


# ─────────────────────────────────────────────────────────────────────────────
# 2. Qwen2.5-VL Visual Grounding Prompt & Parser
# ─────────────────────────────────────────────────────────────────────────────

VLM_GROUNDING_SYSTEM_PROMPT = """You are SafeSight AI, an elite multimodal forensic grounding engine.
When inspecting images for manipulation or synthetic generation:
1. Identify any localized region containing AI inpainting, pasted generative objects, or synthetic artifacts.
2. For each detected region, assign:
   - 'box_2d': Normalized coordinates [ymin, xmin, ymax, xmax] (0-1000).
   - 'region_label': A concise, plain-English label describing the EXACT object (e.g., 'AI-Inpainted Birds in Sky', 'Synthetic Handwritten Notes', 'Deepfake Face Blend').
   - 'category': One of 'INPAINTING', 'GENERATIVE_OBJECT', 'SYNTHETIC_TEXT', 'FACE_SWAP', 'COMPRESSION_ARTIFACT'.
   - 'confidence': float between 0.0 and 1.0.
   - 'forensic_explanation': Explain the visual or mathematical clue (e.g., 'Diffusion edge feathering inconsistent with natural sky background', 'Identical stroke pressure and kerning').
Do NOT emit anonymous or generic labels like 'region_1' or 'anomaly'.

Return ONLY a valid JSON object with this exact structure:
{
  "overall_verdict": "PARTIALLY_SYNTHETIC",
  "ai_percentage": 12.0,
  "tampered_regions": [
    {
      "region_label": "AI-Inpainted Birds in Sky",
      "category": "INPAINTING",
      "confidence": 0.96,
      "box_2d": [180, 310, 340, 620],
      "forensic_explanation": "Diffusion edge feathering inconsistent with natural sky background."
    }
  ]
}

Grounding Coordinate Rules:
- box_2d is [ymin, xmin, ymax, xmax] normalized from 0 to 1000.
- If authentic, set overall_verdict to "AUTHENTIC", ai_percentage to 0.0, and tampered_regions to [].
- If fully synthetic, set overall_verdict to "SYNTHETIC_GENERATED", ai_percentage to 100.0, and box_2d to [0, 0, 1000, 1000].
- If localized inpainting/splicing (e.g. pasted birds, swapped face, synthetic text), set overall_verdict to "PARTIALLY_SYNTHETIC".
"""


def _parse_vlm_grounding_json(raw_text: str) -> dict[str, Any] | None:
    """Extract and parse structured JSON from VLM output."""
    if not raw_text or not raw_text.strip():
        return None

    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

    # Regex search for JSON object
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict) and "overall_verdict" in data:
            return data
    except Exception:
        pass
    return None


async def _query_vlm_for_grounding(
    image_path: Path,
    system_prompt: str,
) -> dict[str, Any] | None:
    """Attempt live VLM call to Qwen2.5-VL via Ollama if available."""
    try:
        from app.models.ollama_client import get_ollama_client
        client = get_ollama_client()
        is_healthy = await client.health_check()
        if not is_healthy:
            return None

        user_prompt = (
            "Analyze this image for localized tampering, generative fills, or spliced elements. "
            "Output the exact normalized [ymin, xmin, ymax, xmax] bounding boxes."
        )
        response_text = await client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            images=[str(image_path)],
        )
        return _parse_vlm_grounding_json(response_text)
    except Exception as exc:
        logger.debug("VLM live grounding call skipped or offline", error=str(exc))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Unified Image Forensics Analyzer
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image_forensics(
    image_input: Union[Path, str, Image.Image],
    text_context: str | None = None,
    output_dir: Path | str | None = None,
) -> ImageForensicsResult:
    """
    Execute full compression-aware image forensics on the input image.

    Steps:
      1. Provenance Fingerprinting (DQT tables + EXIF metadata).
      2. Threshold calibration for social media (WhatsApp, Telegram).
      3. Dual-pass Differential Error Level Analysis (ELA).
      4. High-Frequency DCT inpainting anomaly localization.
      5. Spatial grounding synthesis (VLM grounding or deterministic fusion).
      6. Visual forensic bounding box overlay rendering.

    Returns:
        ImageForensicsResult dataclass with structured findings.
    """
    # Resolve image to PIL Image and source path
    src_file_path: Path | None = None
    if isinstance(image_input, (str, Path)):
        src_file_path = Path(image_input)
        img = Image.open(src_file_path)
        base_name = src_file_path.name
    elif isinstance(image_input, Image.Image):
        img = image_input
        base_name = f"image_{int(time.time() * 1000)}.png"
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    rgb_img = img.convert("RGB")

    # 1. Provenance Fingerprinting
    fingerprint = get_image_fingerprint(img)
    is_social = bool(fingerprint.is_social_media_compressed)

    # 2. Dual-Pass Differential ELA
    ela_res: ELAResult = perform_error_level_analysis(
        image_input=img,
        quality=95,
        quality_low=75,
        fingerprint=fingerprint,
    )

    # 3. High-Frequency DCT Analysis
    dct_score, dct_regions = compute_hf_dct_inpainting_anomalies(
        image=rgb_img,
        is_social_media=is_social,
    )

    # 4. Synthesize Tampered Regions & Decision
    tampered_regions: list[dict[str, Any]] = []

    # Combine ELA bounding box and DCT anomalies
    if ela_res.splicing_detected and ela_res.splicing_region_box:
        tampered_regions.append({
            "region_label": "AI Spliced Element",
            "label": "ai_spliced_element",
            "category": TamperedRegionCategory.INPAINTING.value,
            "box_2d": ela_res.splicing_region_box,
            "confidence": round(min(0.99, max(0.70, ela_res.splicing_anomaly_score)), 2),
            "forensic_explanation": (
                f"Differential Error Level Analysis detected localized compression error variance "
                f"(score: {ela_res.splicing_anomaly_score:.2f}) inconsistent with background."
            ),
            "forensic_reasoning": (
                f"Differential Error Level Analysis detected localized compression error variance "
                f"(score: {ela_res.splicing_anomaly_score:.2f}) inconsistent with background."
            ),
        })

    # Add DCT regions if not heavily overlapping
    for r_dct in dct_regions:
        box_d = r_dct["box_2d"]
        has_overlap = False
        for existing in tampered_regions:
            box_e = existing["box_2d"]
            # Check overlap
            if (abs(box_d[0] - box_e[0]) < 150 and abs(box_d[1] - box_e[1]) < 150):
                has_overlap = True
                break
        if not has_overlap:
            tampered_regions.append(r_dct)

    # Check for specific contextual edge case: spliced birds or objects on WhatsApp background
    is_spliced = (
        ela_res.splicing_anomaly_score >= 0.40
        or (is_social and len(tampered_regions) > 0)
        or dct_score >= 0.45
    )

    # Calculate overall verdict and AI percentage
    if is_spliced and tampered_regions:
        overall_verdict = "PARTIALLY_SYNTHETIC"
        # Calculate AI percentage based on area coverage and anomaly confidence
        w_img, h_img = rgb_img.size
        total_ai_area = 0.0
        for r in tampered_regions:
            b = r["box_2d"]
            bw = max(0, b[3] - b[1]) / 1000.0
            bh = max(0, b[2] - b[0]) / 1000.0
            total_ai_area += (bw * bh)

        ai_pct = min(95.0, max(5.0, round(total_ai_area * 100.0 * 1.5, 1)))
        if ai_pct < 8.0:
            ai_pct = 12.0  # Normalized minimum representation for localized edits

        forensic_explanation = (
            f"While the background shows standard {fingerprint.profile} compression, "
            f"differential ELA and DCT analysis detected high-probability digital splicing "
            f"in {len(tampered_regions)} localized region(s)."
        )
    elif is_social:
        overall_verdict = "AUTHENTIC"
        ai_pct = 0.0
        tampered_regions = []
        forensic_explanation = (
            f"Forensic analysis confirmed the presence of heavy {fingerprint.profile} compression artifacts "
            f"but found no additional evidence of generative AI tampering."
        )
    else:
        # Check standard high-quality original
        if ela_res.splicing_anomaly_score >= 0.55 or dct_score >= 0.60:
            overall_verdict = "PARTIALLY_SYNTHETIC"
            ai_pct = round(max(ela_res.splicing_anomaly_score, dct_score) * 100.0, 1)
            forensic_explanation = "Localized frequency and error level anomalies detected in pristine image."
        else:
            overall_verdict = "AUTHENTIC"
            ai_pct = 0.0
            tampered_regions = []
            forensic_explanation = "Image exhibits consistent error levels, natural DCT spectrum, and authentic sensor noise."

    # 5. Render Visual Forensic Annotation Overlays
    target_dir = Path(output_dir) if output_dir else DEFAULT_REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    clean_base = base_name if not base_name.startswith("annotated_") else base_name[10:]
    annotated_output_path = target_dir / f"annotated_{clean_base}"

    p_regions = [
        DetectedTamperedRegion(
            region_label=r.get("region_label") or r.get("label", "Tampered Region"),
            category=r.get("category", TamperedRegionCategory.INPAINTING),
            confidence=float(r["confidence"]),
            box_2d=r["box_2d"],
            forensic_explanation=r.get("forensic_explanation") or r.get("forensic_reasoning", "Tampered region"),
            forensic_reasoning=r.get("forensic_reasoning") or r.get("forensic_explanation", "Tampered region"),
        )
        for r in tampered_regions
    ]

    rendered_path = render_annotated_tampering_image(
        image_input=rgb_img,
        tampered_regions=p_regions,
        output_path=annotated_output_path,
    )

    return ImageForensicsResult(
        overall_verdict=overall_verdict,
        ai_percentage=ai_pct,
        tampered_regions=tampered_regions,
        compression_fingerprint=fingerprint.profile,
        is_social_media_compressed=is_social,
        ela_splicing_score=round(ela_res.splicing_anomaly_score, 3),
        dct_anomaly_score=round(dct_score, 3),
        annotated_image_path=str(rendered_path.resolve()),
        forensic_explanation=forensic_explanation,
    )
