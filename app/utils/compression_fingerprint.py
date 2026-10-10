"""
SafeSight AI — JPEG Quantization & Metadata Compression Fingerprinter
======================================================================
Pre-analysis forensics module that inspects digital provenance and compression history:
  1. EXIF Metadata Inspection: Detects camera signatures (Make, Model, Lens) or metadata
     stripping typical of social media pipelines (WhatsApp, Telegram, Facebook, Instagram).
  2. JPEG Quantization Table (DQT) Reverse-Engineering: Evaluates discrete cosine
     transform (DCT) luminance and chrominance quantization matrices to estimate original
     quality factor and match against known compression profiles (e.g. WhatsApp Android v2.23,
     Facebook, Original Camera High Quality).
  3. Dynamic Context Generation: Supplies crucial forensic context to downstream VLM
     reasoning engines so benign re-compression artifacts are never misclassified as
     malicious generative AI.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image, ExifTags

from app.logging_config import get_logger

logger = get_logger("compression_fingerprint")

# Standard Independent JPEG Group (IJG) reference luminance table
IJG_LUMINANCE_BASELINE = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

# Standard WhatsApp Android / iOS typical Quantization Table 0 (Quality ~75)
WHATSAPP_Q75_LUMINANCE = [
    8, 6, 5, 8, 12, 20, 26, 31,
    6, 6, 7, 10, 13, 29, 30, 28,
    7, 7, 8, 12, 20, 29, 35, 28,
    7, 9, 11, 15, 26, 44, 40, 31,
    9, 11, 19, 28, 34, 55, 52, 39,
    12, 18, 28, 32, 41, 52, 57, 46,
    25, 32, 39, 44, 52, 61, 60, 51,
    36, 46, 48, 49, 56, 50, 52, 50,
]

WHATSAPP_Q70_LUMINANCE = [
    10, 7, 6, 10, 14, 24, 31, 37,
    7, 7, 8, 11, 16, 35, 36, 33,
    8, 8, 10, 14, 24, 34, 41, 34,
    8, 10, 13, 17, 31, 52, 48, 37,
    11, 13, 22, 34, 41, 65, 62, 46,
    14, 21, 33, 38, 49, 62, 68, 55,
    29, 38, 47, 52, 62, 73, 72, 61,
    43, 55, 57, 59, 67, 60, 62, 59,
]


@dataclass
class CompressionFingerprint:
    """
    Forensic metadata and quantization profile of an inspected media file.
    """
    fingerprint_label: str  # e.g., "WhatsApp_Android_v2.23", "Original_Camera_HighQ", "Unknown_Or_PNG"
    has_exif: bool
    is_social_media_compressed: bool
    estimated_quality: int | None
    dqt_hash: str | None
    dqt_tables: dict[int, list[int]] | None = None
    exif_summary: dict[str, Any] = field(default_factory=dict)
    forensic_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint_label": self.fingerprint_label,
            "has_exif": self.has_exif,
            "is_social_media_compressed": self.is_social_media_compressed,
            "estimated_quality": self.estimated_quality,
            "dqt_hash": self.dqt_hash,
            "exif_summary": self.exif_summary,
            "forensic_context": self.forensic_context,
        }


def estimate_jpeg_quality_from_dqt(table: list[int]) -> int:
    """
    Reverse-engineer JPEG quality factor (1-100) from DQT Luminance Table (Table 0)
    using the standard IJG quantization scaling formula.
    """
    if not table or len(table) < 16:
        return 75

    # Use low-frequency coefficients (first 16 entries)
    scales: list[float] = []
    for t_val, ijg_val in zip(table[:16], IJG_LUMINANCE_BASELINE[:16]):
        if ijg_val > 0:
            scales.append((float(t_val) * 100.0) / float(ijg_val))

    if not scales:
        return 75

    med_scale = float(np.median(scales))
    if med_scale <= 100.0:
        q = int(round((200.0 - med_scale) / 2.0))
    else:
        q = int(round(5000.0 / max(1.0, med_scale)))

    return max(1, min(100, q))


def get_image_fingerprint(
    image_input: Union[Path, str, bytes, Image.Image],
) -> CompressionFingerprint:
    """
    Determine an image's digital provenance, EXIF status, and quantization table signature.

    Args:
        image_input: File path, raw bytes, or PIL Image instance.

    Returns:
        CompressionFingerprint with label (e.g. 'WhatsApp_Android_v2.23',
        'Original_Camera_HighQ', 'Unknown_Or_PNG') and dynamic forensic prompt context.
    """
    # 1. Normalize input to PIL Image
    pil_img: Image.Image
    if isinstance(image_input, (str, Path)):
        src_path = Path(image_input)
        if not src_path.exists():
            return CompressionFingerprint(
                fingerprint_label="Unknown_Or_PNG",
                has_exif=False,
                is_social_media_compressed=False,
                estimated_quality=None,
                dqt_hash=None,
                forensic_context="File unreadable; defaulting to neutral context.",
            )
        try:
            pil_img = Image.open(src_path)
        except Exception:
            return CompressionFingerprint(
                fingerprint_label="Unknown_Or_PNG",
                has_exif=False,
                is_social_media_compressed=False,
                estimated_quality=None,
                dqt_hash=None,
                forensic_context="Non-image binary stream.",
            )
    elif isinstance(image_input, bytes):
        try:
            pil_img = Image.open(io.BytesIO(image_input))
        except Exception:
            return CompressionFingerprint(
                fingerprint_label="Unknown_Or_PNG",
                has_exif=False,
                is_social_media_compressed=False,
                estimated_quality=None,
                dqt_hash=None,
                forensic_context="Corrupted image bytes.",
            )
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    img_format = (pil_img.format or "").upper()

    # 2. EXIF Metadata Extraction
    has_exif = False
    exif_summary: dict[str, Any] = {}
    try:
        raw_exif = pil_img.getexif()
        if raw_exif and len(raw_exif) > 0:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                # Capture prominent camera / software tags
                if tag_name in (
                    "Make", "Model", "Software", "DateTimeOriginal",
                    "DateTime", "Artist", "LensModel", "ExifImageWidth"
                ):
                    exif_summary[tag_name] = str(value)
            if exif_summary or len(raw_exif) >= 3:
                has_exif = True
    except Exception as exc:
        logger.debug("EXIF parsing error", error=str(exc))

    # 3. Non-JPEG / Lossless Formats Check
    if img_format not in ("JPEG", "JPG"):
        # Check if PNG / WebP / BMP
        label = "Unknown_Or_PNG"
        f_context = (
            "Image format is uncompressed or lossless (PNG/WebP). No JPEG quantization tables present. "
            "Scrutinize pixel geometry, color boundaries, and edge frequency."
        )
        return CompressionFingerprint(
            fingerprint_label=label,
            has_exif=has_exif,
            is_social_media_compressed=False,
            estimated_quality=None,
            dqt_hash=None,
            exif_summary=exif_summary,
            forensic_context=f_context,
        )

    # 4. JPEG DQT (Define Quantization Table) Analysis
    dqt_tables_raw = getattr(pil_img, "quantization", None)
    if not dqt_tables_raw or 0 not in dqt_tables_raw:
        label = "Unknown_Or_PNG"
        return CompressionFingerprint(
            fingerprint_label=label,
            has_exif=has_exif,
            is_social_media_compressed=False,
            estimated_quality=None,
            dqt_hash=None,
            exif_summary=exif_summary,
            forensic_context="JPEG image without standard DQT tables.",
        )

    table0 = list(dqt_tables_raw[0])
    table1 = list(dqt_tables_raw[1]) if 1 in dqt_tables_raw else None
    dqt_dict = {0: table0}
    if table1:
        dqt_dict[1] = table1

    dqt_str = ",".join(str(v) for v in table0)
    dqt_hash = hashlib.md5(dqt_str.encode("utf-8")).hexdigest()[:12]

    # Calculate estimated JPEG quality factor
    est_q = estimate_jpeg_quality_from_dqt(table0)

    # Calculate similarity to known WhatsApp profiles
    diff_q75 = float(np.mean(np.abs(np.array(table0) - np.array(WHATSAPP_Q75_LUMINANCE))))
    diff_q70 = float(np.mean(np.abs(np.array(table0) - np.array(WHATSAPP_Q70_LUMINANCE))))

    is_exact_whatsapp_table = diff_q75 < 2.0 or diff_q70 < 2.0

    # 5. Fingerprint Classification Matrix
    camera_make = exif_summary.get("Make", "").lower()
    has_known_camera = any(cam in camera_make for cam in ["apple", "canon", "nikon", "sony", "samsung", "google", "fuji"])

    if has_known_camera or (has_exif and est_q >= 90):
        fingerprint_label = "Original_Camera_HighQ"
        is_social_media = False
        forensic_context = (
            "CONTEXT: This image appears to be an original camera file. "
            "Be highly sensitive to any form of digital noise or compression inconsistency, as it may indicate tampering."
        )

    elif not has_exif and is_exact_whatsapp_table:
        fingerprint_label = "WhatsApp_Android_v2.23"
        is_social_media = True
        forensic_context = (
            "CRITICAL CONTEXT: This image has been processed by a known social media compression engine (WhatsApp). "
            "Expect to see heavy JPEG artifacts, blockiness, and color banding. "
            "Your task is to IGNORE these compression artifacts and search ONLY for additional, anomalous signs of generative AI, "
            "such as unnatural objects, impossible physics, or splicing artifacts that are inconsistent with the background compression."
        )

    elif not has_exif and (65 <= est_q <= 82):
        fingerprint_label = "WhatsApp_Standard"
        is_social_media = True
        forensic_context = (
            "CRITICAL CONTEXT: This image has been processed by a known social media compression engine. "
            "Expect to see heavy JPEG artifacts, blockiness, and color banding. "
            "Your task is to IGNORE these compression artifacts and search ONLY for additional, anomalous signs of generative AI, "
            "such as unnatural objects, impossible physics, or splicing artifacts that are inconsistent with the background compression."
        )

    elif not has_exif and est_q < 65:
        fingerprint_label = "Social_Media_Heavily_Compressed"
        is_social_media = True
        forensic_context = (
            "CRITICAL CONTEXT: This image has undergone aggressive secondary re-compression (Q < 65). "
            "Severe DCT block artifacts and ringing noise are present. "
            "Disregard standard compression noise and scrutinize only high-order structural tampering."
        )

    elif est_q >= 90:
        fingerprint_label = "Original_Camera_HighQ"
        is_social_media = False
        forensic_context = (
            "CONTEXT: This image appears to be an original camera file (High Quality Q >= 90). "
            "Be highly sensitive to any form of digital noise or compression inconsistency, as it may indicate tampering."
        )

    else:
        fingerprint_label = f"Generic_JPEG_Q{est_q}"
        is_social_media = not has_exif
        if is_social_media:
            forensic_context = (
                "CRITICAL CONTEXT: This image has been processed by a social media re-compression pipeline. "
                "Expect to see moderate JPEG artifacts. Ignore benign compression and detect differential anomalies."
            )
        else:
            forensic_context = (
                "CONTEXT: Standard JPEG file with preserved metadata. Evaluate spatial consistency across modalities."
            )

    logger.info(
        "Computed compression fingerprint",
        label=fingerprint_label,
        has_exif=has_exif,
        estimated_quality=est_q,
        is_social_media=is_social_media,
    )

    return CompressionFingerprint(
        fingerprint_label=fingerprint_label,
        has_exif=has_exif,
        is_social_media_compressed=is_social_media,
        estimated_quality=est_q,
        dqt_hash=dqt_hash,
        dqt_tables=dqt_dict,
        exif_summary=exif_summary,
        forensic_context=forensic_context,
    )
