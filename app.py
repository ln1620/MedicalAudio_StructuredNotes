from flask import Flask, jsonify, render_template, request, send_file
from io import BytesIO
import os
import sys

from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

# allow src imports
sys.path.append(os.path.abspath("src"))

from audio_pipeline import audio_to_text
from pdf_export import build_clinical_pdf_bytes
from clinical_note_generator import NOTE_SCHEMA_KEYS, generate_clinical_document
from languages import (
    NOTE_LANGUAGE_OPTIONS,
    SPEECH_LANGUAGE_OPTIONS,
    get_section_titles,
    normalize_note_language,
    normalize_speech_language,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB — room for longer mic recordings

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    transcription = ""
    clinical_note = None
    error_message = None
    selected_speech = "auto"
    selected_note = "en"

    if request.method == "POST":
        selected_speech = request.form.get("speech_language", "auto").strip() or "auto"
        raw_note = request.form.get("note_language", "en").strip() or "en"
        selected_note = normalize_note_language(raw_note)

        file = request.files.get("audio")
        if not file or not file.filename:
            error_message = "Please choose an audio file or record from the microphone."
        else:
            safe_name = secure_filename(file.filename)
            if not safe_name:
                error_message = "Invalid file name."
            else:
                filepath = os.path.join(UPLOAD_FOLDER, safe_name)
                file.save(filepath)
                try:
                    whisper_lang = normalize_speech_language(selected_speech)
                    transcription = audio_to_text(filepath, language=whisper_lang)
                    clinical_note = generate_clinical_document(
                        transcription,
                        note_language=selected_note,
                        speech_language=selected_speech,
                    )
                except Exception as exc:
                    error_message = f"Processing failed: {exc}"

    sections = None
    if clinical_note:
        note_lang = clinical_note.get("_note_language") or normalize_note_language(
            selected_note
        )
        titles = get_section_titles(note_lang)
        sections = [
            {"id": k, "title": titles.get(k, k), "body": clinical_note.get(k, "")}
            for k in NOTE_SCHEMA_KEYS
            if clinical_note.get(k)
        ]

    pdf_payload = None
    if clinical_note:
        pdf_payload = {
            "transcription": transcription or "",
            "sections": [
                {"title": s["title"], "body": s["body"]}
                for s in (sections or [])
            ],
            "speech": selected_speech,
            "note": selected_note,
        }

    return render_template(
        "index.html",
        transcription=transcription,
        clinical_note=clinical_note,
        sections=sections,
        error_message=error_message,
        speech_language_options=SPEECH_LANGUAGE_OPTIONS,
        note_language_options=NOTE_LANGUAGE_OPTIONS,
        selected_speech=selected_speech,
        selected_note=selected_note,
        pdf_payload=pdf_payload,
    )


@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    """Generate PDF from JSON body (same content as shown on the results page)."""
    data = request.get_json(silent=True) or {}
    transcription = data.get("transcription") or ""
    sections = data.get("sections")
    if not isinstance(sections, list):
        sections = []
    speech = data.get("speech") or ""
    note = data.get("note") or ""
    try:
        pdf_bytes = build_clinical_pdf_bytes(
            transcription,
            sections,
            speech_setting=str(speech),
            note_setting=str(note),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="clinical-note-draft.pdf",
    )


if __name__ == "__main__":
    # macOS often uses port 5000 for AirPlay Receiver; default to 5001 to avoid "Address already in use".
    port = int(os.environ.get("PORT", 5001))
    # Single process by default: avoids loading Whisper twice (reloader spawns a child process).
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "0").lower() in ("1", "true", "yes")
    app.run(debug=True, host="127.0.0.1", port=port, use_reloader=use_reloader)
