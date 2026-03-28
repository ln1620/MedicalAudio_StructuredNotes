"""
Convert patient monologue (from speech-to-text) into clinician-style documentation.

Preferred: OpenAI-compatible chat API or Groq (fast, best quality).
Fallback: structured heuristic note (no API; honest limitations).

Set env vars (optional .env in project root — see STEP_BY_STEP in project or docs).
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

from languages import (
    get_note_language_display_name,
    get_speech_language_display_name,
    normalize_note_language,
)

# JSON fields expected from LLM and used in the UI (same keys for heuristic path).
NOTE_SCHEMA_KEYS = [
    "chief_complaint",
    "history_of_present_illness",
    "context_aggravating_or_relief",
    "associated_symptoms",
    "timeline_and_duration",
    "review_of_systems_pertinent",
    "pertinent_negatives",
    "summary_for_clinician",
    "documentation_note",
]

CLINICIAN_JSON_INSTRUCTIONS = """You are an experienced clinician writing the *subjective* portion of a clinical encounter note.
Your ONLY source of information is the patient transcript below. The patient spoke naturally (possibly disorganized); your job is to reorganize it the way a careful clinician documents—not keyword tagging.

Rules:
1. Write in third person ("The patient reports...", "States...", "Describes..."). Do not use "I" for the patient.
2. Do NOT invent symptoms, durations, prior history, medications, allergies, negatives, or diagnoses that are not clearly implied by the transcript.
3. If something is unclear or not mentioned, say so briefly (e.g. "Duration not specified") rather than guessing.
4. Do NOT output ICD codes or definitive diagnoses. Descriptive language only.
5. Separate *story/context* (what happened before symptoms) from the symptom description when helpful.
6. "pertinent_negatives" must list only denials or negatives the patient actually said (e.g. "denies fever"). If none said, use: "Not explicitly addressed in transcript."
7. "documentation_note" is a short line such as: subjective only; examination, investigations, and medical decision-making to be documented separately—unless the patient only mentioned self-care they already tried (then summarize those facts only).

Return a SINGLE JSON object with exactly these keys (string values, use empty string only if truly nothing applies):
- chief_complaint
- history_of_present_illness
- context_aggravating_or_relief
- associated_symptoms
- timeline_and_duration
- review_of_systems_pertinent
- pertinent_negatives
- summary_for_clinician
- documentation_note

Transcript:
---
{transcript}
---
"""


def _build_clinician_user_prompt(
    transcript: str,
    note_lang_code: str,
    speech_lang_code: str | None = None,
) -> str:
    """
    speech_lang_code: UI value e.g. 'te', 'en', or 'auto' (must be passed through so the LLM
    knows the transcript may be Telugu etc., especially when the note is English).
    """
    text = transcript.strip()
    note_code = normalize_note_language(note_lang_code)
    speech_raw = (speech_lang_code or "auto").strip().lower()
    speech_label = get_speech_language_display_name(
        None if speech_raw in ("auto", "", "none") else speech_raw
    )

    preamble = (
        "=== Recording language setting ===\n"
        f"The patient chose (or auto-detected) speech language context: **{speech_label}**.\n"
        "The transcript below may be in a non-Latin script (e.g. Telugu, Hindi). "
        "Read it as that language, not as English.\n"
        "=== End preamble ===\n\n"
    )

    base = preamble + CLINICIAN_JSON_INSTRUCTIONS.format(transcript=text)

    # English note from non-English speech (most common mismatch issue)
    if note_code == "en":
        extra = [
            "\n\n---\nCross-lingual output (required):\n",
            "- All JSON string values must be written in **English** (clinical style).\n",
            "- The transcript may be entirely in another language: translate the *medical meaning* faithfully.\n",
            "- Do not hallucinate symptoms or durations. If something is unintelligible, say so briefly.\n",
            "- Do not leave long passages in Telugu/Devanagari/etc. inside the JSON values unless it is a proper name or drug name best left untranslated.\n",
            "- Use third person in English (e.g. \"The patient reports…\", \"States…\").\n",
        ]
        if speech_raw not in ("auto", "", "none") and speech_raw != "en":
            extra.append(
                f"- The transcript is expected to be primarily **{get_note_language_display_name(speech_raw)}**. "
                "Map concepts (pain, duration, body part) carefully into English.\n"
            )
        return base + "".join(extra)

    # Non-English note
    name = get_note_language_display_name(note_code)
    return (
        base
        + "\n\nOutput language (required):\n"
        f"- Write **every** JSON string value entirely in **{name}**.\n"
        "- Use standard clinical documentation style in that language (third-person).\n"
        "- Preserve factual content from the transcript; translate or rephrase as needed for that language.\n"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group())


def _normalize_note(
    raw: dict[str, Any],
    transcript: str,
    source: str,
    *,
    note_lang_code: str = "en",
    speech_lang_code: str | None = None,
) -> dict[str, Any]:
    raw = dict(raw)
    for alias, key in (
        ("pertinent_negatives_if stated", "pertinent_negatives"),
        ("pertinent_negatives_if_stated", "pertinent_negatives"),
    ):
        if alias in raw and key not in raw:
            raw[key] = raw.pop(alias)
    out: dict[str, Any] = {
        "_source": source,
        "_transcript_length": len(transcript),
        "_note_language": note_lang_code,
        "_speech_language": speech_lang_code or "auto",
    }
    for key in NOTE_SCHEMA_KEYS:
        val = raw.get(key)
        if val is None:
            out[key] = ""
        elif isinstance(val, list):
            out[key] = "; ".join(str(x) for x in val)
        else:
            out[key] = str(val).strip()
    return out


# --- LLM backends ---


def _generate_openai_compatible(
    transcript: str,
    note_lang_code: str = "en",
    speech_lang_code: str | None = None,
) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed")
        return None

    base_url = os.getenv("OPENAI_BASE_URL")  # None = default OpenAI; or e.g. OpenRouter
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key, base_url=base_url)
    user_content = _build_clinician_user_prompt(transcript, note_lang_code, speech_lang_code)
    lang = get_note_language_display_name(normalize_note_language(note_lang_code))
    sys_msg = "You write accurate clinical documentation and answer with JSON only."
    if normalize_note_language(note_lang_code) == "en":
        sys_msg += (
            " Transcripts may be in Telugu, Hindi, or other languages; every JSON string value must be in English."
        )
    else:
        sys_msg += f" All JSON string values must be in {lang}."

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
    }
    low = model.lower()
    if base_url is None or "openai.com" in (base_url or ""):
        if "gpt-4" in low or "gpt-3.5-turbo" in low or "gpt-5" in low:
            kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or "{}"
    data = _extract_json_object(content)
    return _normalize_note(
        data,
        transcript,
        f"openai:{model}",
        note_lang_code=normalize_note_language(note_lang_code),
        speech_lang_code=speech_lang_code,
    )


def _generate_groq(
    transcript: str,
    note_lang_code: str = "en",
    speech_lang_code: str | None = None,
) -> dict[str, Any] | None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        from groq import Groq
    except ImportError:
        logger.warning("groq package not installed")
        return None

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    client = Groq(api_key=key)
    user_content = _build_clinician_user_prompt(transcript, note_lang_code, speech_lang_code)
    lang = get_note_language_display_name(normalize_note_language(note_lang_code))
    sys_msg = "You write accurate clinical documentation. Reply with JSON only, no markdown."
    if normalize_note_language(note_lang_code) == "en":
        sys_msg += (
            " The transcript may be in Telugu or other languages; write every JSON value in English only, translating faithfully."
        )
    else:
        sys_msg += f" All JSON string values must be in {lang}."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or "{}"
    data = _extract_json_object(content)
    return _normalize_note(
        data,
        transcript,
        f"groq:{model}",
        note_lang_code=normalize_note_language(note_lang_code),
        speech_lang_code=speech_lang_code,
    )


@lru_cache(maxsize=1)
def _local_llm_pipe():
    """Small local instruct model; first load downloads weights (~1GB). Optional."""
    if os.getenv("USE_LOCAL_CLINICAL_LLM", "").lower() not in ("1", "true", "yes"):
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        return None

    model_id = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    return {"tokenizer": tok, "model": model, "name": model_id}


def _generate_local_qwen(
    transcript: str,
    note_lang_code: str = "en",
    speech_lang_code: str | None = None,
) -> dict[str, Any] | None:
    bundle = _local_llm_pipe()
    if bundle is None:
        return None
    import torch

    tok, model, name = bundle["tokenizer"], bundle["model"], bundle["name"]
    user_content = _build_clinician_user_prompt(transcript, note_lang_code, speech_lang_code)
    sys_msg = "You reply with JSON only."
    if normalize_note_language(note_lang_code) == "en":
        sys_msg += " Transcript may be non-English; JSON values must be English."
    else:
        sys_msg += f" Write all string values in {get_note_language_display_name(normalize_note_language(note_lang_code))}."
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_content},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt")
    out = model.generate(
        **inputs,
        max_new_tokens=700,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    content = tok.decode(gen, skip_special_tokens=True)
    try:
        data = _extract_json_object(content)
    except (json.JSONDecodeError, ValueError):
        return None
    return _normalize_note(
        data,
        transcript,
        f"local:{name}",
        note_lang_code=normalize_note_language(note_lang_code),
        speech_lang_code=speech_lang_code,
    )


# --- Heuristic fallback (no API) ---

_SYMPTOM_CUE = re.compile(
    r"\b(pain|hurt|aching|ache|sore|burning|stiff|swell|tender|numb|tingle|nausea|vomit|"
    r"dizzy|fever|cough|breath|wheez|fatigue|weak|cramp|throb|headache|rash|itch)\w*\b",
    re.I,
)

_TIME_CUE = re.compile(
    r"\b(?:for|since|about|over|during|after|before|started|began)\s+[^.!?\n]{0,80}?"
    r"(?:\d+\s*)?(?:day|week|month|year|hour|minute|min|hr)s?\b|\b\d+\s*(?:day|week|month|year|hour|minute)s?\b",
    re.I,
)


def _pick_chief_complaint(sentences: list[str], full_text: str) -> str:
    for s in sentences:
        if _SYMPTOM_CUE.search(s):
            return s
    return sentences[0] if sentences else full_text[:500]


def _heuristic_note(
    transcript: str,
    note_lang_code: str = "en",
    speech_lang_code: str | None = None,
) -> dict[str, Any]:
    code = normalize_note_language(note_lang_code)
    t = transcript.strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    cc = _pick_chief_complaint(sentences, t)
    times = _TIME_CUE.findall(t)
    timeline = "; ".join(times) if times else "Not clearly specified in the transcript."

    hpi = (
        f"The patient provides a narrative in their own words. The subjective report is summarized as follows: "
        f"{t} "
        f"This documentation reflects speech-to-text only; details should be confirmed with the patient."
    )

    raw = {
        "chief_complaint": cc,
        "history_of_present_illness": hpi,
        "context_aggravating_or_relief": "See narrative. (Heuristic mode—no language model; extract aggravating/relief factors manually if needed.)",
        "associated_symptoms": "Not separately parsed in heuristic mode; see full transcript.",
        "timeline_and_duration": timeline,
        "review_of_systems_pertinent": "Not systematically reviewed in transcript format; see narrative.",
        "pertinent_negatives": "Not explicitly parsed in heuristic mode; review transcript for denials.",
        "summary_for_clinician": (
            f"Subjective only (automated heuristic). Key time expressions noted: {timeline}. "
            "Correlate with exam, vitals, and history as appropriate."
        ),
        "documentation_note": (
            "Heuristic draft—no clinical LLM configured. For practitioner-quality notes, set GROQ_API_KEY or OPENAI_API_KEY "
            "(see setup instructions)."
        ),
    }
    if code != "en":
        name = get_note_language_display_name(code)
        raw["documentation_note"] += f" Note: headings/heuristic text may be partly English; full {name} notes need the LLM API."
    return _normalize_note(
        raw,
        transcript,
        "heuristic",
        note_lang_code=code,
        speech_lang_code=speech_lang_code,
    )


def generate_clinical_document(
    transcript: str,
    note_language: str = "en",
    speech_language: str | None = None,
) -> dict[str, Any]:
    """
    Produce structured clinical-style documentation from raw transcript.
    Tries: provider override → Groq → OpenAI-compatible → local Qwen → heuristic.

    `note_language`: ISO 639-1 code (e.g. en, hi, es) for the written note text.
    `speech_language`: UI selection (e.g. te, en, auto) so the LLM knows the transcript language.
    """
    note_lang_code = normalize_note_language(note_language)
    speech_raw = (speech_language or "auto").strip().lower()

    if not transcript or not transcript.strip():
        return _normalize_note(
            {
                "chief_complaint": "",
                "history_of_present_illness": "No transcript available.",
                **{k: "" for k in NOTE_SCHEMA_KEYS[2:]},
            },
            "",
            "empty",
            note_lang_code=note_lang_code,
            speech_lang_code=speech_raw,
        )

    provider = (os.getenv("CLINICAL_NOTE_PROVIDER") or "").lower().strip()
    order: list[Any] = []

    def add(fn, name):
        order.append((name, fn))

    if provider == "groq":
        add(lambda: _generate_groq(transcript, note_lang_code, speech_raw), "groq")
    elif provider == "openai":
        add(lambda: _generate_openai_compatible(transcript, note_lang_code, speech_raw), "openai")
    elif provider == "local":
        add(lambda: _generate_local_qwen(transcript, note_lang_code, speech_raw), "local")
    elif provider == "heuristic":
        order = []
    else:
        add(lambda: _generate_groq(transcript, note_lang_code, speech_raw), "groq")
        add(lambda: _generate_openai_compatible(transcript, note_lang_code, speech_raw), "openai")
        add(lambda: _generate_local_qwen(transcript, note_lang_code, speech_raw), "local")

    for _, fn in order:
        try:
            note = fn()
            if note is not None:
                return note
        except Exception as e:
            logger.warning("Clinical note generator step failed: %s", e, exc_info=False)

    return _heuristic_note(transcript, note_lang_code, speech_raw)
