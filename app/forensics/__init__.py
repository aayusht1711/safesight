"""
SafeSight AI — Multimodal Forensics Suite
==========================================
Modular forensic analyzers for Image, Audio, and Video media:
  - image_forensics: Compression-aware JPEG DQT, differential ELA, high-frequency DCT, Qwen2.5-VL grounding.
  - audio_forensics: Temporal audio forensics, Mel-spectrograms, phase consistency, breath cadence, voice cloning localization.
  - video_forensics: Spatiotemporal video forensics, 4 FPS face tracking, temporal jitter, boundary color blending, optical flow discontinuity.
"""

from app.forensics.image_forensics import (
    ImageForensicsResult,
    analyze_image_forensics,
    compute_hf_dct_inpainting_anomalies,
)
from app.forensics.audio_forensics import (
    AudioForensicsResult,
    SyntheticAudioSegment,
    analyze_audio_forensics,
    load_audio_signal,
    render_annotated_spectrogram,
)
from app.forensics.video_forensics import (
    ManipulatedFrameRange,
    VideoForensicsResult,
    analyze_video_forensics,
    render_annotated_video_frame,
)

__all__ = [
    "ImageForensicsResult",
    "analyze_image_forensics",
    "compute_hf_dct_inpainting_anomalies",
    "AudioForensicsResult",
    "SyntheticAudioSegment",
    "analyze_audio_forensics",
    "load_audio_signal",
    "render_annotated_spectrogram",
    "ManipulatedFrameRange",
    "VideoForensicsResult",
    "analyze_video_forensics",
    "render_annotated_video_frame",
]
