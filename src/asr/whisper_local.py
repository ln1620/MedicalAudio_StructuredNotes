from __future__ import annotations

import logging
import os

from asr.base import ASRResult, ASREngine

logger = logging.getLogger(__name__)

_whisper_model = None
_whisper_model_id: str | None = None


def _pick_device() -> str:
    override = (os.getenv("WHISPER_DEVICE") or "").strip().lower()
    if override in ("cpu", "cuda", "mps"):
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _fp16_for_device() -> bool:
    return _pick_device() == "cuda"


def _get_whisper_model():
    global _whisper_model, _whisper_model_id
    name = (os.getenv("WHISPER_MODEL") or "small").strip() or "small"
    if _whisper_model is None or _whisper_model_id != name:
        device = _pick_device()
        try:
            import whisper

            _whisper_model = whisper.load_model(name, device=device)
        except Exception as e:
            logger.warning("Whisper load on %s failed (%s); falling back to CPU.", device, e)
            import whisper

            _whisper_model = whisper.load_model(name, device="cpu")
            device = "cpu"
        _whisper_model_id = name
        logger.info("Whisper model=%s device=%s", name, device)
    return _whisper_model


class WhisperLocalEngine(ASREngine):
    name = "whisper"

    def transcribe(self, audio_path: str, *, language: str | None) -> ASRResult:
        """
        Local Whisper transcription.

        `language` is Whisper ISO 639-1 code (e.g. te) or None for auto-detect.
        """
        model = _get_whisper_model()

        beam_default = (
            "1"
            if (os.getenv("WHISPER_FAST", "").lower() in ("1", "true", "yes"))
            else "2"
        )
        beam = int(os.getenv("WHISPER_BEAM_SIZE", beam_default))
        no_speech = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.45"))

        decode_kwargs: dict = {
            "verbose": False,
            "fp16": _fp16_for_device(),
            "beam_size": max(1, beam),
            "task": "transcribe",
            "no_speech_threshold": max(0.0, min(1.0, no_speech)),
        }
        if os.getenv("WHISPER_CONDITION_ON_PREVIOUS", "").lower() in ("0", "false", "no"):
            decode_kwargs["condition_on_previous_text"] = False
        if language:
            decode_kwargs["language"] = language

        if language == "te":
            decode_kwargs["initial_prompt"] = os.getenv(
                "WHISPER_TE_INITIAL_PROMPT",
                "ఆరోగ్యం, నొప్పి, లక్షణాలు, మందులు, ఎప్పటి నుంచో.",
            )
        elif language == "hi":
            decode_kwargs["initial_prompt"] = os.getenv(
                "WHISPER_HI_INITIAL_PROMPT",
                "स्वास्थ्य, दर्द, लक्षण, दवाइयाँ, कब से।",
            )

        result = model.transcribe(audio_path, **decode_kwargs)
        return ASRResult(
            text=(result.get("text") or "").strip(),
            engine="whisper",
            detected_language=result.get("language"),
        )

