"""
Build a PDF of the transcript + structured clinical note (draft).
Uses Noto Sans when available under static/fonts/ for multilingual text.
"""

from __future__ import annotations

import os
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

_FONT_REGISTERED = False
_FONT_NAME = "Helvetica"


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_unicode_font() -> str:
    """Register Noto Sans if present; return ReportLab font name."""
    global _FONT_REGISTERED, _FONT_NAME
    if _FONT_REGISTERED:
        return _FONT_NAME

    candidates = [
        os.path.join(_project_root(), "static", "fonts", "NotoSans-Regular.ttf"),
        os.path.join(_project_root(), "static", "fonts", "DejaVuSans.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                name = "ClinicalUnicode"
                pdfmetrics.registerFont(TTFont(name, path))
                _FONT_NAME = name
                _FONT_REGISTERED = True
                return name
            except Exception:
                continue
    _FONT_NAME = "Helvetica"
    _FONT_REGISTERED = True
    return _FONT_NAME


def _p(text: str) -> str:
    """Escape for ReportLab Paragraph XML; preserve line breaks."""
    if not text:
        return ""
    return "<br/>".join(escape(line) for line in str(text).splitlines())


def build_clinical_pdf_bytes(
    transcription: str,
    sections: list[dict],
    *,
    speech_setting: str = "",
    note_setting: str = "",
    disclaimer: str = "Draft documentation only. Not for diagnosis or treatment. A licensed clinician must review and edit.",
) -> bytes:
    font = _ensure_unicode_font()
    buf = BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Clinical note (draft)",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CTitle",
        parent=styles["Heading1"],
        fontName=font,
        fontSize=16,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "CH2",
        parent=styles["Heading2"],
        fontName=font,
        fontSize=11,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "CBody",
        parent=styles["Normal"],
        fontName=font,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )
    small_style = ParagraphStyle(
        "CSmall",
        parent=styles["Normal"],
        fontName=font,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748b"),
    )
    warn_style = ParagraphStyle(
        "CWarn",
        parent=styles["Normal"],
        fontName=font,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#92400e"),
        backColor=colors.HexColor("#fffbeb"),
        borderPadding=6,
    )

    story: list = []

    story.append(Paragraph("Subjective encounter note (draft)", title_style))
    meta_bits = []
    if speech_setting:
        meta_bits.append(f"Speech setting: {speech_setting}")
    if note_setting:
        meta_bits.append(f"Note language: {note_setting}")
    if meta_bits:
        story.append(Paragraph(" · ".join(meta_bits), small_style))
        story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(disclaimer, warn_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Transcript</b>", h2_style))
    story.append(Paragraph(_p(transcription or "(empty)"), body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Structured note</b>", h2_style))
    for sec in sections:
        title = sec.get("title") or "Section"
        body = sec.get("body") or ""
        story.append(Paragraph(f"<b>{escape(title)}</b>", body_style))
        story.append(Paragraph(_p(body), body_style))
        story.append(Spacer(1, 0.08 * inch))

    doc.build(story)
    data = buf.getvalue()
    buf.close()
    return data
