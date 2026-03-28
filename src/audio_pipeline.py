import os

import torch
import whisper

# "small" is much better than "base" for Telugu and other Indian languages (larger download).
# Override with WHISPER_MODEL=base if CPU is too slow.
_whisper_model = None
_whisper_model_id: str | None = None


def _get_whisper_model():
    global _whisper_model, _whisper_model_id
    name = (os.getenv("WHISPER_MODEL") or "small").strip() or "small"
    if _whisper_model is None or _whisper_model_id != name:
        _whisper_model = whisper.load_model(name)
        _whisper_model_id = name
    return _whisper_model


def audio_to_text(audio_path, language=None):
    """
    Transcribe audio. `language` is Whisper ISO 639-1 code (e.g. 'te', 'hi') or None for auto-detect.

    Uses beam search and optional language-specific prompt for better Indic transcription.
    Set WHISPER_MODEL=medium for even better quality (slower / more RAM).
    """
    model = _get_whisper_model()
    fp16 = torch.cuda.is_available()

    decode_kwargs = {
        "verbose": False,
        "fp16": fp16,
        "beam_size": int(os.getenv("WHISPER_BEAM_SIZE", "5")),
        "task": "transcribe",
    }
    if language:
        decode_kwargs["language"] = language

    # Short in-language prompt nudges the decoder (especially helpful for Telugu script).
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
    return result["text"]
