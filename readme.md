# Healthcare Voice Documentation Assistant

An **AI-assisted pipeline** that turns patient speech into **structured clinical-style notes**: record or upload audio → **Whisper** transcription → **LLM** (Groq or OpenAI-compatible) generates sectioned JSON → optional **PDF** export. Supports **multiple speech languages** with configurable **note language** (e.g. Telugu speech → English note).

> **Disclaimer:** This is a research / prototype tool. It is **not** a medical device and must **not** replace professional documentation or clinical judgment. Do not use real patient identifiers on public deployments without proper compliance review.

## Features

- Browser **microphone** recording or **audio file** upload  
- **OpenAI Whisper** for speech-to-text (language auto or fixed)  
- **Structured clinical note** fields (chief complaint, HPI, timeline, ROS, etc.) via LLM, with a **heuristic fallback** when no API is configured  
- **PDF download** (Unicode-friendly fonts)  
- **Multilingual** UI options driven by `src/languages.py`

## Stack

- **Backend:** Python 3, Flask  
- **ASR:** `openai-whisper`, PyTorch  
- **LLM:** Groq (`groq`) and/or OpenAI-compatible APIs (`openai` client)  
- **PDF:** ReportLab  

## Prerequisites

- **Python 3.10+** (recommended)  
- **ffmpeg** installed and on your `PATH` (required by Whisper to read many audio formats)  
  - macOS: `brew install ffmpeg`  
  - Ubuntu: `sudo apt install ffmpeg`  

## Local setup

```bash
cd /path/to/MAN
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set at least GROQ_API_KEY (or OpenAI keys — see .env.example)
python app.py
```



### Environment variables

Copy `.env.example` to `.env` and configure:

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Recommended: fast LLM notes via Groq |
| `GROQ_MODEL` | e.g. `llama-3.1-8b-instant` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | Alternative: OpenAI or compatible API |
| `CLINICAL_NOTE_PROVIDER` | `groq` \| `openai` \| `local` \| `heuristic` |
| `WHISPER_MODEL` | `base`, `small`, `medium`, … (quality vs speed / RAM) |
| `WHISPER_DEVICE` | `cpu`, `cuda`, `mps`, or empty for auto |
| `WHISPER_FAST`, `WHISPER_BEAM_SIZE` | Speed vs decoding quality |
| `WHISPER_NO_SPEECH_THRESHOLD` | Lower if quiet speech is dropped |
| `FLASK_DEBUG`, `FLASK_HOST`, `PORT` | Local server behavior |

See **`.env.example`** for full comments.

## Deploy (Render)

This repo includes **`render.yaml`** for a **Render** web service:

- Build: `pip install -r requirements.txt`  
- Start: `gunicorn` with extended timeout (Whisper can be slow on first request)  

On the Render dashboard, add **`GROQ_API_KEY`** (and optional Whisper overrides). Free-tier instances may sleep when idle.

## Project layout (core app)

```
MAN/
├── app.py                 # Flask app, routes
├── requirements.txt
├── render.yaml            # optional Render blueprint
├── templates/index.html   # UI
├── static/fonts/          # PDF Unicode fonts
└── src/
    ├── audio_pipeline.py      # Whisper transcription
    ├── clinical_note_generator.py  # LLM / heuristic notes
    ├── languages.py           # speech & note language options
    └── pdf_export.py          # PDF generation
```

## License / usage

Use responsibly. API keys and patient audio must be protected; prefer private deployment and access controls for anything beyond demos.
