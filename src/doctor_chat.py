from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from languages import get_note_language_display_name, normalize_note_language


Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


@dataclass(frozen=True)
class ChatStart:
    conversation_id: str
    assistant_message: str


@dataclass(frozen=True)
class ChatReply:
    assistant_message: str


@dataclass(frozen=True)
class VisitSummary:
    summary_for_clinician: str
    likely_problem_list: str
    red_flags: str
    self_care_advice: str
    suggested_next_steps: str
    prescription_suggestions: str
    disclaimer: str


_EMERGENCY_REGEX = re.compile(
    r"\b(chest pain|shortness of breath|difficulty breathing|faint|unconscious|seizure|"
    r"stroke|weakness on one side|slurred speech|severe bleeding|suicid|overdose|"
    r"severe allergic|anaphylaxis)\b",
    re.I,
)


def _doctor_system_prompt(output_lang_code: str) -> str:
    lang = get_note_language_display_name(normalize_note_language(output_lang_code))
    return (
        "You are an AI clinician conducting a patient intake conversation.\n"
        "Goal: Ask concise follow-up questions to understand symptoms, timeline, severity, context, and safety.\n"
        "Style: warm, professional, short questions, one at a time.\n"
        "If the patient asks for a prescription, explain that a licensed clinician must prescribe and you can only provide general information.\n\n"
        "Safety rules:\n"
        "- You are not a substitute for a licensed clinician.\n"
        "- Do NOT give definitive diagnoses.\n"
        "- If the patient reports danger signs (e.g., chest pain, severe shortness of breath, stroke symptoms, fainting), "
        "tell them to seek urgent/emergency care.\n"
        "- If the patient is a minor, pregnant, immunocompromised, or has severe/worsening symptoms, recommend clinical evaluation.\n"
        "- Do not request sensitive IDs or payment info.\n\n"
        f"Output language requirement: Write ALL assistant messages in {lang}.\n"
    )


def _summary_system_prompt(output_lang_code: str) -> str:
    lang = get_note_language_display_name(normalize_note_language(output_lang_code))
    return (
        "You are an AI clinician summarizing a patient intake chat.\n"
        "Write a brief, structured clinical summary and safe next-step guidance.\n"
        "Do not claim certainty or provide definitive diagnosis.\n"
        f"Write the output in {lang}.\n"
        "Return plain text for each requested section."
    )


def _llm_chat(messages: list[ChatMessage]) -> str:
    """
    Uses Groq if configured, else OpenAI-compatible.
    Returns assistant text.
    """
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if key:
        from groq import Groq

        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if api_key:
        from openai import OpenAI

        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    raise RuntimeError("No LLM configured. Set GROQ_API_KEY or OPENAI_API_KEY.")


def _new_conversation_id() -> str:
    return uuid.uuid4().hex


def initial_doctor_message(output_lang_code: str) -> str:
    # First assistant message should not depend on user content.
    sys_msg: ChatMessage = {"role": "system", "content": _doctor_system_prompt(output_lang_code)}
    user_msg: ChatMessage = {
        "role": "user",
        "content": "Start the visit with a short greeting and ask how you can help.",
    }
    return _llm_chat([sys_msg, user_msg])


def doctor_reply(
    history: list[ChatMessage],
    *,
    output_lang_code: str,
    user_text: str,
) -> ChatReply:
    messages: list[ChatMessage] = [{"role": "system", "content": _doctor_system_prompt(output_lang_code)}]
    messages.extend(history[-16:])  # keep it snappy
    messages.append({"role": "user", "content": user_text})

    # Quick safety nudge before LLM if obvious danger signs are present.
    if _EMERGENCY_REGEX.search(user_text):
        emergency_prefix = (
            "Safety note: The patient may have urgent danger signs. Start by advising urgent evaluation, "
            "then ask one key follow-up question.\n\n"
        )
        messages.append({"role": "system", "content": emergency_prefix})

    text = _llm_chat(messages)
    return ChatReply(assistant_message=text)


def build_visit_summary(history: list[ChatMessage], *, output_lang_code: str) -> VisitSummary:
    # Concatenate user+assistant content for summarization.
    transcript_lines: list[str] = []
    for m in history:
        if m["role"] == "system":
            continue
        who = "Patient" if m["role"] == "user" else "Clinician"
        transcript_lines.append(f"{who}: {m['content']}")
    convo = "\n".join(transcript_lines).strip()

    sys_msg: ChatMessage = {"role": "system", "content": _summary_system_prompt(output_lang_code)}
    user_msg: ChatMessage = {
        "role": "user",
        "content": (
            "From the conversation below, produce these sections in order with headings:\n"
            "1) Summary for clinician (short)\n"
            "2) Likely problem list (possibilities, not diagnosis)\n"
            "3) Red flags to watch for\n"
            "4) Self-care advice\n"
            "5) Suggested next steps (when to see a clinician, tests to consider)\n"
            "6) Prescription suggestions (only as 'discussion points' for a licensed clinician; include typical OTC options if relevant)\n"
            "Also add a final 1-line disclaimer.\n\n"
            f"Conversation:\n---\n{convo}\n---"
        ),
    }

    raw = _llm_chat([sys_msg, user_msg])

    # Very lightweight parsing: split by numbered headings; if parsing fails, keep raw in summary_for_clinician.
    def pick(label: str) -> str:
        m = re.search(rf"{label}[:\n]\s*([\s\S]*?)(?=\n\d\)|\nDisclaimer|\Z)", raw, re.I)
        return (m.group(1).strip() if m else "").strip()

    disclaimer = ""
    mdis = re.search(r"(disclaimer[:\n]\s*[\s\S]*)$", raw, re.I)
    if mdis:
        disclaimer = mdis.group(1).strip()

    return VisitSummary(
        summary_for_clinician=pick(r"1\)\s*Summary for clinician"),
        likely_problem_list=pick(r"2\)\s*Likely problem list"),
        red_flags=pick(r"3\)\s*Red flags"),
        self_care_advice=pick(r"4\)\s*Self-care advice"),
        suggested_next_steps=pick(r"5\)\s*Suggested next steps"),
        prescription_suggestions=pick(r"6\)\s*Prescription suggestions"),
        disclaimer=disclaimer or "Disclaimer: This is informational only and not medical advice. Seek licensed clinical care for diagnosis/treatment.",
    )


class ConversationStore:
    def __init__(self) -> None:
        self._conversations: dict[str, list[ChatMessage]] = {}

    def create(self) -> str:
        cid = _new_conversation_id()
        self._conversations[cid] = []
        return cid

    def get(self, cid: str) -> list[ChatMessage]:
        if cid not in self._conversations:
            self._conversations[cid] = []
        return self._conversations[cid]

    def append(self, cid: str, role: Role, content: str) -> None:
        self.get(cid).append({"role": role, "content": content})

