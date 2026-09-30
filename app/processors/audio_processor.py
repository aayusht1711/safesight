"""
SafeSight AI — Audio Processor
================================
Handles audio bytes → text transcription using OpenAI Whisper-Tiny.

All functions are synchronous and must be called via run_in_executor.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path


def transcribe_audio(audio_bytes: bytes, whisper_model: object) -> str:
    """
    Transcribe audio bytes to text using OpenAI Whisper.

    Supports WAV, MP3, FLAC, M4A, OGG (any format ffmpeg can decode).
    Writes bytes to a temp file and passes the path to whisper.transcribe()
    since Whisper's Python API accepts file paths.

    Args:
        audio_bytes:   Raw audio file bytes.
        whisper_model: A loaded whisper.Model instance.

    Returns:
        Transcribed text string (may be empty if audio is silent/inaudible).
    """
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(
            tmp_path,
            language=None,        # Auto-detect language
            fp16=False,           # Safe default — avoids half-precision issues on CPU
            verbose=False,
        )
        transcript: str = result.get("text", "").strip()
        return transcript

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def normalize_audio_bytes(audio_bytes: bytes) -> bytes:
    """
    Normalize audio to 16kHz mono WAV — the format Whisper expects.

    Uses soundfile + numpy for pure-Python resampling when librosa is available,
    otherwise returns the raw bytes unchanged (Whisper can handle most formats
    via ffmpeg internally).

    Args:
        audio_bytes: Raw audio bytes in any format.

    Returns:
        Normalized WAV bytes at 16kHz mono, or the original bytes on failure.
    """
    try:
        import numpy as np
        import soundfile as sf
        import librosa

        # Load from bytes
        audio_io = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_io, dtype="float32", always_2d=False)

        # Resample to 16kHz if needed
        if sr != 16_000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16_000)

        # Convert stereo to mono
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Serialize back to WAV bytes
        out_io = io.BytesIO()
        sf.write(out_io, y, 16_000, format="WAV", subtype="PCM_16")
        return out_io.getvalue()

    except Exception:
        # Graceful fallback — Whisper handles raw bytes via ffmpeg
        return audio_bytes
