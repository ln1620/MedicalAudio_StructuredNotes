# Hospital Intake — Audio to Clinical Note

I built this as a **hospital-style intake web app** that helps turn **patient speech** into a **reviewable transcript** and a **structured draft clinical note**.

It works in two main modes:
- **Audio → Draft note**: upload/record audio → speech-to-text → structured note + optional PDF
- **Speak to a doctor (AI)**: an interactive intake conversation where the AI asks follow-ups, then generates a visit summary and safe suggestions

> **Disclaimer:** This is a prototype documentation aid. It is **not** a medical device and does **not** replace professional clinical judgment. Do not use real patient identifiers in demos or public deployments.

## What I can do with it
- **Multilingual speech-to-text** (I use Google Cloud Speech-to-Text by default, with local Whisper fallback)
- **Any-language patient speech → English translation panel** (optional) for clinician readability
- **Structured clinical-style note** (chief complaint, HPI, timeline, ROS, etc.) via Groq/OpenAI-compatible LLMs, with a heuristic fallback if no API key is set
- **PDF export** of transcript + structured sections
- **AI doctor conversation**: follow-up questions + “Finish visit” summary + safety guardrails

## Tech stack
- **Backend**: Python + Flask
- **Speech-to-text (ASR)**:
  - Google Cloud Speech-to-Text (recommended)
  - Local Whisper fallback (`openai-whisper`)
- **Translation**: Google Cloud Translation (optional)
- **LLM notes & chat**: Groq or OpenAI-compatible APIs
- **PDF**: ReportLab

## Prerequisites
- **Python 3.10+** (I use 3.11 locally)
- Optional for local Whisper: **ffmpeg**
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`

## Run locally

```bash
cd /path/to/MAN
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Then open `http://127.0.0.1:5001`.

## Configure `.env` (high-signal variables)
I keep real secrets in `.env` (and **do not commit** `.env` or `keys/`).

### LLM (notes + AI doctor chat)
- `GROQ_API_KEY` (recommended)
- `GROQ_MODEL` (example: `llama-3.1-8b-instant`)

Alternative:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_MODEL`

### Speech-to-text (multilingual)
Recommended (cloud):
- `ASR_ENGINE=google`
- `GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json`
- `GOOGLE_CLOUD_PROJECT=...` (needed for translation)

Optional:
- `TRANSLATE_TO_ENGLISH=1` (shows English translation panel when transcript isn’t English)

Local fallback (Whisper) tuning (optional):
- `WHISPER_MODEL=base|small|medium|...`
- `WHISPER_DEVICE=cpu|mps|cuda`
- `WHISPER_FAST=1` and/or `WHISPER_BEAM_SIZE=...`

See `.env.example` for the full list and comments.

## How I use it
- If I’m speaking Telugu/Hindi/Tamil, I **select the language explicitly** for best accuracy.
- I review the transcript first, then use the structured note as a draft.
- In “Speak to a doctor (AI)”, I click **Finish visit** to generate a summary and next-step suggestions (still clinician-reviewed).

## Project layout

```
MAN/
├── app.py
├── templates/index.html
├── static/
│   ├── images/                 # hospital UI illustrations
│   └── fonts/                  # PDF Unicode fonts
└── src/
    ├── audio_pipeline.py       # ASR dispatcher + caching
    ├── doctor_chat.py          # AI doctor conversation + visit summary
    ├── translate_google.py     # English translation panel (optional)
    ├── clinical_note_generator.py
    ├── pdf_export.py
    ├── languages.py
    └── asr/
        ├── base.py
        ├── google_stt.py
        └── whisper_local.py
```

## Responsible usage
I treat this as a **documentation helper** only:
- don’t enter real identifiers in demos
- keep API keys private
- always have a licensed clinician review output
