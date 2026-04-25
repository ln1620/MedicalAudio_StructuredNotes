from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=128)
def translate_to_english(text: str, *, source_language: str | None = None) -> str | None:
    """
    Translate text to English using Google Cloud Translation.

    Returns translated string, or None if translation is not configured/available.
    """
    if not text or not text.strip():
        return ""

    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    if not project:
        return None

    src = (source_language or "").strip()
    # If it’s already English (en or en-US etc.), don’t translate.
    if src:
        base = src.split("-", 1)[0].lower()
        if base == "en":
            return None

    try:
        from google.cloud import translate_v3 as translate
    except Exception:
        return None

    client = translate.TranslationServiceClient()
    parent = f"projects/{project}/locations/global"

    kwargs: dict = {
        "parent": parent,
        "contents": [text],
        "mime_type": "text/plain",
        "target_language_code": "en",
    }
    # Don’t force source language by default; let the API auto-detect.
    # (Passing "en-us" while targeting "en" can error; auto-detect is robust.)

    resp = client.translate_text(request=translate.TranslateTextRequest(**kwargs))
    if not resp.translations:
        return None
    return (resp.translations[0].translated_text or "").strip()

