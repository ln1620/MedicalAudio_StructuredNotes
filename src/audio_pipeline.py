import hashlib
import logging
import os
from collections import OrderedDict

from asr.base import ASRResult

logger = logging.getLogger(__name__)


def _engine_name() -> str:
    return (os.getenv("ASR_ENGINE") or "").strip().lower() or "google"


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

_ASR_CACHE: "OrderedDict[str, ASRResult]" = OrderedDict()
_ASR_CACHE_MAX = int(os.getenv("ASR_CACHE_MAX", "32"))


def _cache_get(key: str) -> ASRResult | None:
    hit = _ASR_CACHE.get(key)
    if hit is None:
        return None
    _ASR_CACHE.move_to_end(key)
    return hit


def _cache_put(key: str, val: ASRResult) -> None:
    _ASR_CACHE[key] = val
    _ASR_CACHE.move_to_end(key)
    while len(_ASR_CACHE) > max(1, _ASR_CACHE_MAX):
        _ASR_CACHE.popitem(last=False)


def _transcribe_uncached(audio_path: str, *, language: str | None) -> ASRResult:
    engine = _engine_name()

    if engine == "google":
        try:
            from asr.google_stt import GoogleSpeechEngine, google_available

            if google_available():
                return GoogleSpeechEngine().transcribe(audio_path, language=language)
            logger.info("Google STT selected but credentials not found; falling back to Whisper.")
        except Exception as e:
            logger.warning("Google STT failed; falling back to Whisper (%s)", e, exc_info=False)

    # Default / fallback: local Whisper (lazy import)
    from asr.whisper_local import WhisperLocalEngine

    return WhisperLocalEngine().transcribe(audio_path, language=language)


def audio_to_text_result(audio_path: str, language: str | None = None) -> ASRResult:
    """
    Transcribe audio using the configured engine.

    Returns ASRResult(text, detected_language, engine).
    """
    try:
        fp = os.path.abspath(audio_path)
        digest = _file_sha256(fp)
    except Exception:
        fp = os.path.abspath(audio_path)
        digest = f"{os.path.getmtime(fp)}:{os.path.getsize(fp)}"

    key = f"{_engine_name()}:{language or 'auto'}:{digest}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    res = _transcribe_uncached(fp, language=language)
    _cache_put(key, res)
    return res


def audio_to_text(audio_path: str, language: str | None = None) -> str:
    """
    Backwards-compatible helper: returns transcript text only.
    """
    return audio_to_text_result(audio_path, language=language).text
