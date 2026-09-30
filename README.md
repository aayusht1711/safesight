# SafeSight AI 🛡️
### Real-Time Multimodal Content Moderation — Production Backend

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

SafeSight AI is a **production-grade, two-tier cascaded content moderation system** that processes text, images, video frames, and audio files in real time. It evaluates two independent dimensions:

| Dimension | What it detects |
|---|---|
| **Authenticity** | Deepfakes, AI-generated media, face/voice cloning |
| **Harm & Policy** | Hate speech, self-harm, explicit content, cyberbullying, violent threats |

---

## Architecture

```
 Request (text + image + audio + video)
         │
         ▼
┌────────────────────────────────────────────────┐
│         Media Pre-Processing (async)            │
│  ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ Image    │ │ Whisper  │ │ Text Normalizer│  │
│  │ Proc.    │ │ ASR      │ │ (NFKC + clean) │  │
│  └──────────┘ └──────────┘ └────────────────┘  │
└────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────┐  ← < 50ms
│  TIER 1 — Fast-Path Screening Gate             │
│                                                │
│  1. Hard-block regex keyword list              │
│  2. CLIP/SigLIP visual embedding similarity   │
│     (safe anchors vs unsafe anchors)          │
│  3. DeBERTa-v3 toxicity classification        │
│  4. Weighted score fusion + threshold routing │
└────────────────────────────────────────────────┘
         │                    │
    SAFE / FLAGGED        BORDERLINE
    (short-circuit)           │
         │                    ▼
         │   ┌──────────────────────────────────┐  ← < 700ms
         │   │  TIER 2 — VLM Reasoning Engine   │
         │   │                                  │
         │   │  Qwen2.5-VL-3B-Instruct          │
         │   │  (4-bit NF4 quantization)         │
         │   │                                  │
         │   │  Bimodal text+image prompt →      │
         │   │  Structured JSON moderation       │
         │   │  decision with explanations       │
         │   └──────────────────────────────────┘
         │                    │
         └────────┬───────────┘
                  ▼
    ┌─────────────────────────────┐
    │   ModerationResult (JSON)   │
    │   decision + confidence     │
    │   violations + reasoning    │
    └─────────────────────────────┘
```

---

## Project Layout

```
SafeSight/
├── app/
│   ├── main.py                   # FastAPI app factory + lifespan
│   ├── config.py                 # Pydantic-Settings configuration
│   ├── schemas.py                # All Pydantic request/response models
│   ├── logging_config.py         # structlog setup
│   ├── models/
│   │   └── registry.py           # Model singleton + parallel loading
│   ├── pipeline/
│   │   ├── orchestrator.py       # Master async pipeline orchestrator
│   │   ├── tier1_gate.py         # Fast-path screening (<50ms)
│   │   └── tier2_vlm.py          # Qwen2.5-VL reasoning engine (<700ms)
│   ├── processors/
│   │   ├── image_processor.py    # PIL image + PyAV video frames
│   │   ├── audio_processor.py    # Whisper transcription
│   │   └── text_processor.py     # NFKC normalization + cleaning
│   └── routes/
│       └── moderation.py         # FastAPI route handlers
├── tests/
│   ├── mock_pipeline.py          # Deterministic mock (no ML needed)
│   ├── test_single.py            # CLI single-request test tool
│   ├── test_suite.py             # Batch 10-case F1 validation suite
│   └── test_unit.py              # pytest unit tests
├── run_server.py                 # Uvicorn entry point
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Clone & Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env — set DEVICE=cpu if no GPU
```

### 2. Run Tests (No GPU / Models Required)

```bash
# Unit tests — schema, text processor, mock pipeline
pytest tests/test_unit.py -v

# Single request test (mock mode — no server needed)
python tests/test_single.py --text "Hello world, have a nice day!" --mock

# Single request with image + text
python tests/test_single.py --text "Check this out" --image path/to/image.jpg --mock

# Batch 10-case validation suite with F1 report
python tests/test_suite.py

# Verbose mode (shows failure analysis)
python tests/test_suite.py --verbose
```

### 3. Start the API Server

```bash
# Development (auto-reload)
python run_server.py --reload

# Production
python run_server.py --host 0.0.0.0 --port 8000 --workers 1

# Or directly via uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Model health check |
| `POST` | `/api/v1/moderate/multimodal` | Full multimodal (form-data) |
| `POST` | `/api/v1/moderate/image` | Image + caption |
| `POST` | `/api/v1/moderate/text` | Text-only (JSON body) |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

### 5. Example API Request

```bash
# Text + Image moderation
curl -X POST http://localhost:8000/api/v1/moderate/multimodal \
  -F "text=Look at this suspicious content" \
  -F "image=@/path/to/image.jpg"

# Text-only (JSON)
curl -X POST http://localhost:8000/api/v1/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world, this is a safe post!"}'
```

---

## Response Schema

```json
{
  "decision": "APPROVED | FLAGGED | HUMAN_AUDIT_REQUIRED",
  "confidence_score": 0.94,
  "detected_violations": [
    {
      "category": "HATE_SPEECH_AND_HARASSMENT",
      "policy_code": "MLC_H1.1",
      "severity": "HIGH",
      "confidence": 0.95
    }
  ],
  "reasoning": {
    "text_explanation": "Text contains explicit hate speech targeting a protected group.",
    "visual_explanation": "Image contains hate symbols consistent with extremist imagery.",
    "bimodal_synergy": "The combination of text + image amplifies harm — the image provides context that makes the text more threatening."
  },
  "pipeline_tier_used": "TIER2",
  "total_latency_ms": 312.4,
  "media_types_processed": ["text", "image"]
}
```

---

## Policy Violation Categories

| Category | Policy Code | Description |
|----------|-------------|-------------|
| `HATE_SPEECH_AND_HARASSMENT` | `MLC_H1.1` | Slurs, targeted hate, protected groups |
| `CYBERBULLYING` | `MLC_H1.2` | Personal attacks, doxxing |
| `SELF_HARM` | `MLC_S2.1` | Suicide, self-injury, eating disorders |
| `EXPLICIT_CONTENT` | `MLC_E3.1` | Sexual/nudity content |
| `DEEPFAKE` | `MLC_D4.1` | Synthetic faces, voice cloning |
| `VIOLENT_THREATS` | `MLC_V5.1` | Explicit threats, weapons glorification |
| `MISINFORMATION` | `MLC_M6.1` | Dangerous false claims |
| `SPAM_AND_MANIPULATION` | `MLC_SP7.1` | Coordinated inauthentic behavior |

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace model ID for Tier-2 |
| `VLM_LOAD_IN_4BIT` | `true` | Enable NF4 4-bit quantization |
| `WHISPER_MODEL_SIZE` | `tiny` | Whisper model size |
| `TIER1_CONFIDENCE_THRESHOLD` | `0.82` | Auto-approve above this score |
| `TIER1_SUSPICIOUS_THRESHOLD` | `0.45` | Auto-flag below this score |
| `TIER2_TIMEOUT_MS` | `700` | VLM inference timeout budget |
| `DEVICE` | `cuda` | `cuda` / `cpu` / `mps` |
| `REDIS_URL` | `redis://localhost:6379/0` | Result cache (optional) |

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| VRAM (Tier-2 VLM) | 4 GB (4-bit) | 8 GB |
| Storage (models) | ~8 GB | ~15 GB |

> **CPU Mode**: Set `DEVICE=cpu` and `VLM_LOAD_IN_4BIT=false`.  
> Tier-2 latency will be significantly higher (~5-30s) but everything runs.

---

## Design Decisions

- **Why Qwen2.5-VL-3B?** Best-in-class small VLM for vision-language reasoning at 3B params. 4-bit NF4 quantization brings it under 4GB VRAM while preserving >95% quality.
- **Why cascaded tiers?** ~70% of posts are clearly safe. Skipping VLM inference for these saves 10-20× GPU compute — critical for throughput at scale.
- **Why DeBERTa for Tier-1?** DeBERTa-v3 outperforms BERT/RoBERTa on toxicity benchmarks while being small enough for sub-50ms inference.
- **Why asyncio throughout?** I/O bound operations (file reads, Redis, HTTP) and CPU-bound model inference (thread pool) are both optimized — the event loop never blocks.
