"""
Languages for Whisper transcription (ISO 639-1) and clinical note output.

Whisper language list: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
"""

from __future__ import annotations

# (code, display label) — code "auto" skips passing language to Whisper (auto-detect)
SPEECH_LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Auto-detect"),
    ("en", "English"),
    ("hi", "Hindi (हिन्दी)"),
    ("te", "Telugu (తెలుగు)"),
    ("ta", "Tamil (தமிழ்)"),
    ("kn", "Kannada (ಕನ್ನಡ)"),
    ("ml", "Malayalam (മലയാളം)"),
    ("mr", "Marathi (मराठी)"),
    ("bn", "Bengali (বাংলা)"),
    ("gu", "Gujarati (ગુજરાતી)"),
    ("pa", "Punjabi (ਪੰਜਾਬੀ)"),
    ("ur", "Urdu (اردو)"),
    ("es", "Spanish (Español)"),
    ("fr", "French (Français)"),
    ("de", "German (Deutsch)"),
    ("it", "Italian (Italiano)"),
    ("pt", "Portuguese (Português)"),
    ("zh", "Chinese (中文)"),
    ("ja", "Japanese (日本語)"),
    ("ko", "Korean (한국어)"),
    ("ar", "Arabic (العربية)"),
    ("ru", "Russian (Русский)"),
    ("tr", "Turkish (Türkçe)"),
    ("pl", "Polish (Polski)"),
    ("nl", "Dutch (Nederlands)"),
    ("vi", "Vietnamese (Tiếng Việt)"),
    ("th", "Thai (ไทย)"),
    ("id", "Indonesian (Bahasa Indonesia)"),
    ("ms", "Malay (Bahasa Melayu)"),
    ("sw", "Swahili (Kiswahili)"),
    ("el", "Greek (Ελληνικά)"),
    ("he", "Hebrew (עברית)"),
    ("fa", "Persian (فارسی)"),
    ("uk", "Ukrainian (Українська)"),
]

# Note output: no "auto" — pick explicit language for the written note
NOTE_LANGUAGE_OPTIONS: list[tuple[str, str]] = [x for x in SPEECH_LANGUAGE_OPTIONS if x[0] != "auto"]

# English name for LLM prompts (must match model understanding)
NOTE_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "ru": "Russian",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "sw": "Swahili",
    "el": "Greek",
    "he": "Hebrew",
    "fa": "Persian",
    "uk": "Ukrainian",
}

# Section headings shown in the UI (per note language); fallback English
SECTION_TITLES_EN: dict[str, str] = {
    "chief_complaint": "Chief complaint",
    "history_of_present_illness": "History of present illness",
    "context_aggravating_or_relief": "Context / aggravating & relieving factors",
    "associated_symptoms": "Associated symptoms",
    "timeline_and_duration": "Timeline & duration",
    "review_of_systems_pertinent": "Review of systems (pertinent)",
    "pertinent_negatives": "Pertinent negatives (if stated)",
    "summary_for_clinician": "Brief clinical summary",
    "documentation_note": "Documentation note",
}

SECTION_TITLES_I18N: dict[str, dict[str, str]] = {
    "es": {
        "chief_complaint": "Motivo de consulta",
        "history_of_present_illness": "Historia de la enfermedad actual",
        "context_aggravating_or_relief": "Contexto / factores agravantes y de alivio",
        "associated_symptoms": "Síntomas asociados",
        "timeline_and_duration": "Cronología y duración",
        "review_of_systems_pertinent": "Revisión de sistemas (pertinente)",
        "pertinent_negatives": "Negativos pertinentes (si se indicaron)",
        "summary_for_clinician": "Resumen clínico breve",
        "documentation_note": "Nota de documentación",
    },
    "hi": {
        "chief_complaint": "मुख्य शिकायत",
        "history_of_present_illness": "वर्तमान बीमारी का इतिहास",
        "context_aggravating_or_relief": "संदर्भ / बिगाड़ने और राहत के कारक",
        "associated_symptoms": "संबंधित लक्षण",
        "timeline_and_duration": "समयरेखा और अवधि",
        "review_of_systems_pertinent": "तंत्र की समीक्षा (संबंधित)",
        "pertinent_negatives": "प्रासंगिक नकारात्मक (यदि कहा गया)",
        "summary_for_clinician": "संक्षिप्त नैदानिक सारांश",
        "documentation_note": "दस्तावेज़ी नोट",
    },
    "te": {
        "chief_complaint": "ప్రధాన ఫిర్యాదు",
        "history_of_present_illness": "ప్రస్తుత అనారోగ్య చరిత్ర",
        "context_aggravating_or_relief": "సందర్భం / తీవ్రత పెంచే మరియు ఉపశమన కారకాలు",
        "associated_symptoms": "సంబంధిత లక్షణాలు",
        "timeline_and_duration": "కాలరేఖ మరియు వ్యవధి",
        "review_of_systems_pertinent": "వ్యవస్థల సమీక్ష (సంబంధిత)",
        "pertinent_negatives": "సంబంధిత ప్రతికూలాలు (తెలిపితే)",
        "summary_for_clinician": "సంక్షిప్త క్లినికల్ సారాంశం",
        "documentation_note": "డాక్యుమెంటేషన్ గమనిక",
    },
    "fr": {
        "chief_complaint": "Motif de consultation",
        "history_of_present_illness": "Histoire de la maladie actuelle",
        "context_aggravating_or_relief": "Contexte / facteurs aggravants et de soulagement",
        "associated_symptoms": "Symptômes associés",
        "timeline_and_duration": "Chronologie et durée",
        "review_of_systems_pertinent": "Revue des systèmes (pertinente)",
        "pertinent_negatives": "Négatifs pertinents (si mentionnés)",
        "summary_for_clinician": "Bref résumé clinique",
        "documentation_note": "Note de documentation",
    },
    "de": {
        "chief_complaint": "Hauptbeschwerde",
        "history_of_present_illness": "Aktuelle Anamnese",
        "context_aggravating_or_relief": "Kontext / verschlimmernde und lindernde Faktoren",
        "associated_symptoms": "Begleitsymptome",
        "timeline_and_duration": "Zeitverlauf und Dauer",
        "review_of_systems_pertinent": "Systemüberblick (relevant)",
        "pertinent_negatives": "Relevante Negativangaben (falls genannt)",
        "summary_for_clinician": "Kurze klinische Zusammenfassung",
        "documentation_note": "Dokumentationshinweis",
    },
    "zh": {
        "chief_complaint": "主诉",
        "history_of_present_illness": "现病史",
        "context_aggravating_or_relief": "背景 / 加重与缓解因素",
        "associated_symptoms": "伴随症状",
        "timeline_and_duration": "时间线与病程",
        "review_of_systems_pertinent": "系统回顾（相关）",
        "pertinent_negatives": "阴性症状（如提及）",
        "summary_for_clinician": "简要临床摘要",
        "documentation_note": "文书说明",
    },
    "ar": {
        "chief_complaint": "الشكوى الرئيسية",
        "history_of_present_illness": "تاريخ المرض الحالي",
        "context_aggravating_or_relief": "السياق / عوامل التفاقم والتخفيف",
        "associated_symptoms": "الأعراض المصاحبة",
        "timeline_and_duration": "الجدول الزمني والمدة",
        "review_of_systems_pertinent": "مراجعة الأجهزة (ذات الصلة)",
        "pertinent_negatives": "السلبيات ذات الصلة (إن ذُكرت)",
        "summary_for_clinician": "ملخص سريري موجز",
        "documentation_note": "ملاحظة توثيق",
    },
    "pt": {
        "chief_complaint": "Queixa principal",
        "history_of_present_illness": "História da doença atual",
        "context_aggravating_or_relief": "Contexto / fatores agravantes e de alívio",
        "associated_symptoms": "Sintomas associados",
        "timeline_and_duration": "Linha do tempo e duração",
        "review_of_systems_pertinent": "Revisão de sistemas (pertinente)",
        "pertinent_negatives": "Negativos pertinentes (se mencionados)",
        "summary_for_clinician": "Resumo clínico breve",
        "documentation_note": "Nota de documentação",
    },
}


def normalize_speech_language(code: str | None) -> str | None:
    """Return Whisper language code or None for auto-detect."""
    if not code or code.strip().lower() in ("auto", "", "none"):
        return None
    c = code.strip().lower()
    valid = {x[0] for x in SPEECH_LANGUAGE_OPTIONS}
    if c not in valid or c == "auto":
        return None
    return c


def normalize_note_language(code: str | None) -> str:
    """Return note language code; default English."""
    if not code:
        return "en"
    c = code.strip().lower()
    if c in NOTE_LANGUAGE_NAMES:
        return c
    return "en"


def get_note_language_display_name(code: str) -> str:
    return NOTE_LANGUAGE_NAMES.get(code, "English")


def get_speech_language_display_name(code: str | None) -> str:
    """English label for LLM prompts (speech dropdown / Whisper)."""
    if not code or str(code).strip().lower() in ("auto", "", "none"):
        return "unspecified (auto-detected; language may vary)"
    c = str(code).strip()
    # Allow BCP-47 locale codes (e.g. te-IN) by reducing to base language.
    if "-" in c:
        c = c.split("-", 1)[0]
    c = c.lower()
    return NOTE_LANGUAGE_NAMES.get(c, c)


def get_section_titles(note_lang_code: str) -> dict[str, str]:
    base = dict(SECTION_TITLES_EN)
    overrides = SECTION_TITLES_I18N.get(note_lang_code, {})
    base.update(overrides)
    return base


# --- Cloud speech locale helpers (Google) ---

_GOOGLE_SPEECH_LOCALE: dict[str, str] = {
    "en": "en-US",
    "hi": "hi-IN",
    "te": "te-IN",
    "ta": "ta-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "ur": "ur-IN",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-PT",
    "zh": "zh-CN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "ar": "ar-SA",
    "ru": "ru-RU",
    "tr": "tr-TR",
    "pl": "pl-PL",
    "nl": "nl-NL",
    "vi": "vi-VN",
    "th": "th-TH",
    "id": "id-ID",
    "ms": "ms-MY",
    "sw": "sw-KE",
    "el": "el-GR",
    "he": "he-IL",
    "fa": "fa-IR",
    "uk": "uk-UA",
}


def to_google_speech_locale(code: str | None) -> str:
    """
    Convert UI ISO 639-1 code (e.g. 'te') to Google Speech-to-Text locale (BCP-47).
    """
    if not code:
        return "en-US"
    c = code.strip().lower()
    return _GOOGLE_SPEECH_LOCALE.get(c, "en-US")


def google_alternative_locales() -> list[str]:
    """
    Locales to enable language identification when the user selects Auto.
    Keep this list small-ish to reduce confusion and latency.
    """
    # Prefer languages in the dropdown first.
    codes = [c for (c, _) in SPEECH_LANGUAGE_OPTIONS if c != "auto"]
    out: list[str] = []
    for c in codes:
        loc = to_google_speech_locale(c)
        if loc not in out:
            out.append(loc)
    # Ensure English is first.
    if "en-US" in out:
        out.remove("en-US")
    return ["en-US"] + out
