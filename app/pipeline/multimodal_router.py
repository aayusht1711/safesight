"""
SafeSight AI — Unified Multimodal Forensic Router
===================================================
Auto-routes media files to specialized forensic engines:
  - IMAGE: Compression-aware DQT, Differential ELA, High-Frequency DCT, Qwen2.5-VL Grounding
  - AUDIO: Mel-Spectrogram, Spectral Phase Consistency, Breath Cadence, Timestamp Splicing
  - VIDEO: 4 FPS Face Tracking, Boundary Blending, Optical Flow, Frame Range Intervals
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.forensics.audio_forensics import (
    AudioForensicsResult,
    SyntheticAudioSegment,
    analyze_audio_forensics,
    load_audio_signal,
    render_annotated_spectrogram,
)
from app.forensics.image_forensics import (
    ImageForensicsResult,
    analyze_image_forensics,
    compute_hf_dct_inpainting_anomalies,
)
from app.forensics.video_forensics import (
    ManipulatedFrameRange,
    VideoForensicsResult,
    analyze_video_forensics,
    render_annotated_video_frame,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def detect_media_modality(file_path: Path | str) -> str:
    """Auto-detect media modality (IMAGE / AUDIO / VIDEO) from file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "IMAGE"
    elif suffix in AUDIO_EXTENSIONS:
        return "AUDIO"
    elif suffix in VIDEO_EXTENSIONS:
        return "VIDEO"
    return "IMAGE"


def run_multimodal_forensics(
    file_path: Path | str,
    media_type: str = "auto",
    text_context: str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Unified Multimodal Forensic Router.

    Dispatches input media to the appropriate specialized pipeline:
      - IMAGE: DQT calibration + Differential ELA + High-frequency DCT + Visual Grounding
      - AUDIO: Mel-Spectrogram + Phase Consistency + Breath Cadence + Spliced Intervals
      - VIDEO: 4 FPS Face Tracking + Temporal Jitter + Boundary Blending + Optical Flow

    Generates visual forensic proof:
      - For Images: 'annotated_<filename>.png' (red bounding box overlays)
      - For Audio: 'annotated_<filename>_spectrogram.png' (highlighted frequency bands)
      - For Video: 'annotated_<filename>_frame_<N>.png' (suspect frame with deepfake mask)

    Returns:
        Structured dictionary with modality, verdict, exact AI component,
        coordinate/timestamp table, and visual evidence path.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Media file not found: {p}")

    modality = (
        media_type.upper()
        if media_type and media_type.lower() != "auto"
        else detect_media_modality(p)
    )

    if modality == "IMAGE":
        res_img = analyze_image_forensics(p, text_context=text_context, output_dir=output_dir)
        if res_img.tampered_regions:
            first_label = res_img.tampered_regions[0]["label"].replace("_", " ").title()
            ai_comp = first_label
        elif res_img.overall_verdict == "AUTHENTIC":
            ai_comp = "None (Authentic Image)"
        else:
            ai_comp = "Synthetic Image Generation"

        conf = (
            max([float(r["confidence"]) for r in res_img.tampered_regions], default=0.0)
            if res_img.tampered_regions
            else (1.0 if res_img.overall_verdict == "AUTHENTIC" else 0.85)
        )

        return {
            "modality": "IMAGE",
            "verdict": res_img.overall_verdict,
            "ai_component": ai_comp,
            "ai_percentage": res_img.ai_percentage,
            "confidence": round(conf, 2),
            "coordinates_or_timestamps": res_img.tampered_regions,
            "evidence_file_path": res_img.annotated_image_path,
            "forensic_explanation": res_img.forensic_explanation,
            "forensics_details": res_img.to_dict(),
        }

    elif modality == "AUDIO":
        res_aud = analyze_audio_forensics(p, output_dir=output_dir)
        if res_aud.synthetic_segments:
            first_seg = res_aud.synthetic_segments[0]
            first_type = first_seg["type"].replace("_", " ").title()
            ai_comp = f"{first_type} Segment"
        elif res_aud.audio_verdict == "AUTHENTIC":
            ai_comp = "None (Authentic Speech)"
        else:
            ai_comp = "Full Voice Clone / TTS"

        conf = (
            max([float(s["confidence"]) for s in res_aud.synthetic_segments], default=0.0)
            if res_aud.synthetic_segments
            else (1.0 if res_aud.audio_verdict == "AUTHENTIC" else 0.90)
        )

        return {
            "modality": "AUDIO",
            "verdict": res_aud.audio_verdict,
            "ai_component": ai_comp,
            "ai_percentage": res_aud.synthetic_percentage,
            "confidence": round(conf, 2),
            "coordinates_or_timestamps": res_aud.synthetic_segments,
            "evidence_file_path": res_aud.annotated_spectrogram_path,
            "forensic_explanation": res_aud.forensic_explanation,
            "forensics_details": res_aud.to_dict(),
        }

    elif modality == "VIDEO":
        res_vid = analyze_video_forensics(p, output_dir=output_dir)
        if res_vid.manipulated_frames:
            first_mf = res_vid.manipulated_frames[0]
            first_anom = first_mf["anomaly"].replace("_", " ").title()
            ai_comp = first_anom
        elif res_vid.video_verdict == "AUTHENTIC":
            ai_comp = "None (Authentic Video)"
        else:
            ai_comp = "Suspect Video Tampering"

        conf = (
            max([float(m["confidence"]) for m in res_vid.manipulated_frames], default=0.0)
            if res_vid.manipulated_frames
            else (1.0 if res_vid.video_verdict == "AUTHENTIC" else 0.85)
        )

        ai_pct = (
            round((len(res_vid.manipulated_frames) / max(res_vid.total_frames_analyzed, 1)) * 100.0, 1)
            if res_vid.manipulated_frames
            else 0.0
        )

        return {
            "modality": "VIDEO",
            "verdict": res_vid.video_verdict,
            "ai_component": ai_comp,
            "ai_percentage": ai_pct,
            "confidence": round(conf, 2),
            "coordinates_or_timestamps": res_vid.manipulated_frames,
            "evidence_file_path": res_vid.annotated_video_frame_path,
            "forensic_explanation": res_vid.forensic_explanation,
            "forensics_details": res_vid.to_dict(),
        }

    else:
        raise ValueError(f"Unsupported media modality: {modality}")
