"""
SafeSight AI — Multimodal Forensics Unit & Integration Test Suite
==================================================================
Tests for:
  1. Compression-Aware Image Forensics (DQT, Differential ELA, High-Freq DCT, Visual Grounding)
  2. Temporal Audio Forensics (Mel-Spectrogram, Spectral Phase Consistency, Breath Cadence, Spliced Timestamps)
  3. Spatiotemporal Video Forensics (4 FPS Face Tracking, Boundary Blending, Optical Flow, Frame Intervals)
  4. Unified Multimodal Router (run_multimodal_forensics)
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest

from app.forensics.audio_forensics import (
    analyze_audio_forensics,
    compute_stft_and_melspectrogram,
    compute_spectral_phase_consistency,
    analyze_breath_cadence,
    render_annotated_spectrogram,
    SyntheticAudioSegment,
)
from app.forensics.image_forensics import (
    analyze_image_forensics,
    compute_hf_dct_inpainting_anomalies,
    _parse_vlm_grounding_json,
    VLM_GROUNDING_SYSTEM_PROMPT,
)
from app.forensics.video_forensics import (
    analyze_video_forensics,
    render_annotated_video_frame,
    ManipulatedFrameRange,
)
from app.pipeline import (
    run_multimodal_forensics,
    detect_media_modality,
)
from app.utils.compression_fingerprint import (
    get_image_fingerprint,
    WHATSAPP_Q75_LUMINANCE,
)


class TestImageForensicsSuite:
    """Tests for compression-aware image forensics engine."""

    def test_dqt_whatsapp_and_telegram_calibration(self, tmp_path: Path):
        # Create an image re-compressed at WhatsApp standard quality (75)
        img = Image.new("RGB", (320, 240), color=(135, 206, 235))
        draw = ImageDraw.Draw(img)
        for y in range(0, 240, 8):
            draw.line([(0, y), (320, y)], fill=(120, 150, 180 + y // 4), width=2)

        wa_path = tmp_path / "whatsapp_test.jpg"
        img.save(wa_path, format="JPEG", quality=75)

        fingerprint = get_image_fingerprint(wa_path)
        assert fingerprint.is_social_media_compressed is True
        assert "WhatsApp" in fingerprint.fingerprint_label or fingerprint.estimated_quality in range(65, 85)

        # Run image forensics
        res = analyze_image_forensics(wa_path, output_dir=tmp_path)
        assert res.overall_verdict == "AUTHENTIC"
        assert res.ai_percentage == 0.0
        assert res.is_social_media_compressed is True
        assert Path(res.annotated_image_path).exists()

    def test_hf_dct_and_differential_ela_detects_inpainted_patch(self, tmp_path: Path):
        # Base background compressed at Q75
        base_img = Image.new("RGB", (400, 400), color=(100, 150, 200))
        bg_buf = io.BytesIO()
        base_img.save(bg_buf, format="JPEG", quality=75)
        bg_buf.seek(0)
        spliced_img = Image.open(bg_buf).convert("RGB")

        # Paste an anomalous high-frequency synthetic patch (birds/object) at [120, 140, 260, 300]
        patch_w = 160
        patch_h = 140
        np_noise = np.random.RandomState(42).randint(0, 255, (patch_h, patch_w, 3), dtype=np.uint8)
        patch_img = Image.fromarray(np_noise)
        spliced_img.paste(patch_img, (140, 120))

        spliced_path = tmp_path / "spliced_birds_image.jpg"
        spliced_img.save(spliced_path, format="JPEG", quality=92)

        res = analyze_image_forensics(spliced_path, output_dir=tmp_path)
        assert res.overall_verdict in ("PARTIALLY_SYNTHETIC", "TAMPERED (PARTIAL_SYNTHETIC)")
        assert len(res.tampered_regions) > 0
        assert res.ai_percentage > 0.0
        assert Path(res.annotated_image_path).exists()

    def test_vlm_grounding_json_parser(self):
        valid_response = """```json
{
  "overall_verdict": "PARTIALLY_SYNTHETIC",
  "ai_percentage": 12.0,
  "tampered_regions": [
    {
      "label": "ai_inpainted_birds",
      "box_2d": [180, 310, 340, 620],
      "confidence": 0.96
    }
  ]
}
```"""
        parsed = _parse_vlm_grounding_json(valid_response)
        assert parsed is not None
        assert parsed["overall_verdict"] == "PARTIALLY_SYNTHETIC"
        assert parsed["ai_percentage"] == 12.0
        assert len(parsed["tampered_regions"]) == 1
        assert parsed["tampered_regions"][0]["label"] == "ai_inpainted_birds"
        assert parsed["tampered_regions"][0]["box_2d"] == [180, 310, 340, 620]


class TestAudioForensicsSuite:
    """Tests for temporal audio forensics engine."""

    def _generate_wav(self, path: Path, sr: int = 16000, duration: float = 6.0, is_synthetic: bool = False):
        import scipy.io.wavfile as wavfile
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

        if not is_synthetic:
            # Authentic speech simulation: harmonics + periodic natural pauses/breaths
            signal = 0.4 * np.sin(2 * np.pi * 180 * t) + 0.2 * np.sin(2 * np.pi * 360 * t)
            # Add respiration energy drop at 3.0s - 3.4s
            breath_mask = (t >= 3.0) & (t <= 3.35)
            signal[breath_mask] *= 0.05
            signal += 0.01 * np.random.randn(len(t)).astype(np.float32)
        else:
            # Synthetic vocoder simulation: high-frequency phase noise and continuous cadence
            signal = 0.4 * np.sin(2 * np.pi * 200 * t)
            # High-frequency phase jitter noise
            high_freq_noise = 0.25 * np.random.randn(len(t)).astype(np.float32)
            signal += high_freq_noise

        signal_int16 = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(str(path), sr, signal_int16)

    def test_authentic_audio_analysis(self, tmp_path: Path):
        wav_path = tmp_path / "authentic_speech.wav"
        self._generate_wav(wav_path, duration=5.0, is_synthetic=False)

        res = analyze_audio_forensics(wav_path, output_dir=tmp_path)
        assert res.total_duration_sec >= 4.9
        assert res.spectral_phase_consistency >= 0.50
        assert Path(res.annotated_spectrogram_path).exists()

    def test_spliced_audio_interval_detection(self, tmp_path: Path):
        # Create audio with clean speech 0-3s, and synthetic noise 3-7s
        sr = 16000
        dur_total = 8.0
        t = np.linspace(0, dur_total, int(sr * dur_total), endpoint=False, dtype=np.float32)

        signal = 0.3 * np.sin(2 * np.pi * 150 * t)
        # Spliced voice clone between 3.0s and 6.5s
        synth_slice = (t >= 3.0) & (t <= 6.5)
        signal[synth_slice] += 0.35 * np.random.randn(int(np.sum(synth_slice))).astype(np.float32)

        wav_path = tmp_path / "spliced_speech.wav"
        import scipy.io.wavfile as wavfile
        wavfile.write(str(wav_path), sr, (signal * 32767).astype(np.int16))

        res = analyze_audio_forensics(wav_path, output_dir=tmp_path)
        assert res.audio_verdict in ("SPLICED_AUDIO", "SYNTHETIC_SPEECH")
        assert len(res.synthetic_segments) > 0
        seg = res.synthetic_segments[0]
        assert seg["type"] == "voice_clone"
        assert seg["confidence"] >= 0.70
        assert Path(res.annotated_spectrogram_path).exists()


class TestVideoForensicsSuite:
    """Tests for spatiotemporal video forensics engine."""

    def _generate_test_video(self, path: Path, num_frames: int = 16, is_deepfake: bool = False):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 4.0, (240, 240))

        for i in range(num_frames):
            frame = np.full((240, 240, 3), 180, dtype=np.uint8)
            # Draw synthetic head oval
            cv2.ellipse(frame, (120, 120), (60, 80), 0, 0, 360, (210, 190, 170), -1)

            if is_deepfake and 4 <= i <= 12:
                # Introduce facial boundary color mismatch and boundary seam
                cv2.circle(frame, (120, 115), 45, (80, 80, 240), -1)  # Reddish mismatched face mask
                # Add temporal jitter displacement
                jitter_offset = int((i % 2) * 12)
                cv2.circle(frame, (120 + jitter_offset, 115), 10, (20, 20, 20), -1)
            else:
                # Authentic facial features
                cv2.circle(frame, (100, 105), 8, (50, 50, 50), -1)
                cv2.circle(frame, (140, 105), 8, (50, 50, 50), -1)

            writer.write(frame)
        writer.release()

    def test_authentic_video_analysis(self, tmp_path: Path):
        vid_path = tmp_path / "authentic_interview.mp4"
        self._generate_test_video(vid_path, num_frames=12, is_deepfake=False)

        res = analyze_video_forensics(vid_path, output_dir=tmp_path)
        assert res.total_frames_analyzed >= 10
        assert res.video_verdict in ("AUTHENTIC", "SUSPECT_TAMPERING")

    def test_deepfake_frame_range_detection(self, tmp_path: Path):
        vid_path = tmp_path / "deepfake_face_swap.mp4"
        self._generate_test_video(vid_path, num_frames=16, is_deepfake=True)

        res = analyze_video_forensics(vid_path, output_dir=tmp_path)
        assert res.video_verdict == "DEEPFAKE_DETECTED"
        assert len(res.manipulated_frames) > 0
        mf = res.manipulated_frames[0]
        assert mf["anomaly"] == "face_swap_boundary_mismatch"
        assert mf["confidence"] >= 0.75
        assert Path(res.annotated_video_frame_path).exists()


class TestMultimodalRouter:
    """Tests for the unified multimodal forensic router."""

    def test_router_image_dispatch(self, tmp_path: Path):
        img_path = tmp_path / "nature.jpg"
        img = Image.new("RGB", (200, 200), color=(50, 120, 80))
        img.save(img_path, format="JPEG", quality=90)

        out = run_multimodal_forensics(img_path, output_dir=tmp_path)
        assert out["modality"] == "IMAGE"
        assert "verdict" in out
        assert "ai_component" in out
        assert "evidence_file_path" in out
        assert Path(out["evidence_file_path"]).exists()

    def test_router_audio_dispatch(self, tmp_path: Path):
        wav_path = tmp_path / "voice_sample.wav"
        sr = 16000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False, dtype=np.float32)
        signal = 0.3 * np.sin(2 * np.pi * 200 * t)
        import scipy.io.wavfile as wavfile
        wavfile.write(str(wav_path), sr, (signal * 32767).astype(np.int16))

        out = run_multimodal_forensics(wav_path, output_dir=tmp_path)
        assert out["modality"] == "AUDIO"
        assert "verdict" in out
        assert "ai_component" in out
        assert "evidence_file_path" in out
        assert Path(out["evidence_file_path"]).exists()

    def test_modality_detection(self):
        assert detect_media_modality("photo.png") == "IMAGE"
        assert detect_media_modality("audio.m4a") == "AUDIO"
        assert detect_media_modality("recording.wav") == "AUDIO"
        assert detect_media_modality("clip.mp4") == "VIDEO"
        assert detect_media_modality("stream.webm") == "VIDEO"
