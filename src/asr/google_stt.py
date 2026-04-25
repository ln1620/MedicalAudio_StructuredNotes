from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from asr.base import ASRResult, ASREngine
from languages import to_google_speech_locale, google_alternative_locales

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AudioFormat:
    encoding: str
    sample_rate_hz: int | None = None


def _guess_format(audio_path: str) -> _AudioFormat | None:
    low = audio_path.lower()
    if low.endswith(".wav"):
        return _AudioFormat("LINEAR16")
    if low.endswith(".flac"):
        return _AudioFormat("FLAC")
    if low.endswith(".mp3"):
        return _AudioFormat("MP3")
    if low.endswith(".ogg"):
        return _AudioFormat("OGG_OPUS")
    if low.endswith(".webm"):
        return _AudioFormat("WEBM_OPUS")
    # m4a/mp4/aac often require explicit conversion; we’ll let Google try AUTO in those cases
    return None


class GoogleSpeechEngine(ASREngine):
    name = "google"

    def __init__(self) -> None:
        # Import lazily so environments without Google deps don’t break local mode.
        from google.cloud import speech

        self._speech = speech
        self._client = speech.SpeechClient()

    def transcribe(self, audio_path: str, *, language: str | None) -> ASRResult:
        """
        Uses Google Cloud Speech-to-Text.

        - If `language` is provided (ISO 639-1 from UI), we set a matching BCP-47 locale.
        - If `language` is None (auto), we enable language identification by providing a set
          of alternative locales commonly used in this app.
        """
        speech = self._speech

        with open(audio_path, "rb") as f:
            content = f.read()

        fmt = _guess_format(audio_path)

        chosen_locale = to_google_speech_locale(language) if language else "en-US"
        alt_locales = google_alternative_locales()
        if language:
            # If user explicitly chose a language, focus on it for best accuracy.
            alt_locales = []

        cfg_kwargs: dict = {
            "language_code": chosen_locale,
            "alternative_language_codes": alt_locales[:20],  # API has limits; keep it small+useful
            "enable_automatic_punctuation": True,
            # For typical short clips (~1 min), prefer latest_short.
            "model": "latest_short",
            "use_enhanced": True,
        }

        if fmt is not None:
            cfg_kwargs["encoding"] = getattr(speech.RecognitionConfig.AudioEncoding, fmt.encoding)
            if fmt.sample_rate_hz:
                cfg_kwargs["sample_rate_hertz"] = fmt.sample_rate_hz

        config = speech.RecognitionConfig(**cfg_kwargs)
        audio = speech.RecognitionAudio(content=content)

        # For ~60s audio, sync recognition is usually fine and simplest.
        resp = self._client.recognize(config=config, audio=audio)

        parts: list[str] = []
        detected: str | None = None

        for r in resp.results:
            if not r.alternatives:
                continue
            alt = r.alternatives[0]
            if alt.transcript:
                parts.append(alt.transcript.strip())
            # Some responses can include language code on result; tolerate absence.
            if getattr(r, "language_code", None):
                detected = r.language_code

        text = " ".join([p for p in parts if p]).strip()
        if not text:
            logger.info("Google STT returned empty transcript")

        return ASRResult(text=text, engine="google", detected_language=detected or None)


def google_available() -> bool:
    if (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip():
        return True
    # Could also work with metadata server / ADC, but this is the most common local setup.
    return False

