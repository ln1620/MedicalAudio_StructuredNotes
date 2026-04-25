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
from audio_pipeline import audio_to_text_result
from pdf_export import build_clinical_pdf_bytes
from clinical_note_generator import NOTE_SCHEMA_KEYS, generate_clinical_document
from languages import (
    NOTE_LANGUAGE_OPTIONS,
    SPEECH_LANGUAGE_OPTIONS,
    get_section_titles,
    normalize_note_language,
    normalize_speech_language,
)
from translate_google import translate_to_english
from doctor_chat import ConversationStore, build_visit_summary, doctor_reply, initial_doctor_message

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB — room for longer mic recordings

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_chat_store = ConversationStore()


@app.route("/", methods=["GET", "POST"])
def index():
    transcription = ""
    detected_language = None
    asr_engine = None
    translation_en = None
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
                    asr = audio_to_text_result(filepath, language=whisper_lang)
                    transcription = asr.text
                    detected_language = asr.detected_language
                    asr_engine = asr.engine
                    if (os.getenv("TRANSLATE_TO_ENGLISH", "1").strip().lower() in ("1", "true", "yes")):
                        try:
                            translation_en = translate_to_english(
                                transcription,
                                source_language=detected_language or whisper_lang or None,
                            )
                        except Exception:
                            # Translation is optional; don’t fail the request if it errors.
                            translation_en = None
                    clinical_note = generate_clinical_document(
                        transcription,
                        note_language=selected_note,
                        speech_language=detected_language or selected_speech,
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
        detected_language=detected_language,
        asr_engine=asr_engine,
        translation_en=translation_en,
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


@app.route("/api/chat/start", methods=["POST"])
def chat_start():
    data = request.get_json(silent=True) or {}
    out_lang = normalize_note_language(str(data.get("output_language") or "en"))
    cid = _chat_store.create()
    msg = initial_doctor_message(out_lang)
    _chat_store.append(cid, "assistant", msg)
    return jsonify({"conversation_id": cid, "assistant_message": msg})


@app.route("/api/chat/message", methods=["POST"])
def chat_message():
    data = request.get_json(silent=True) or {}
    cid = str(data.get("conversation_id") or "").strip()
    if not cid:
        return jsonify({"error": "conversation_id required"}), 400
    user_text = str(data.get("text") or "").strip()
    if not user_text:
        return jsonify({"error": "text required"}), 400
    out_lang = normalize_note_language(str(data.get("output_language") or "en"))

    _chat_store.append(cid, "user", user_text)
    hist = _chat_store.get(cid)
    reply = doctor_reply(hist, output_lang_code=out_lang, user_text=user_text)
    _chat_store.append(cid, "assistant", reply.assistant_message)
    return jsonify({"assistant_message": reply.assistant_message})


@app.route("/api/chat/audio", methods=["POST"])
def chat_audio():
    cid = (request.form.get("conversation_id") or "").strip()
    if not cid:
        return jsonify({"error": "conversation_id required"}), 400
    out_lang = normalize_note_language(str(request.form.get("output_language") or "en"))
    selected_speech = (request.form.get("speech_language") or "auto").strip() or "auto"
    whisper_lang = normalize_speech_language(selected_speech)

    file = request.files.get("audio")
    if not file or not file.filename:
        return jsonify({"error": "audio file required"}), 400
    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"error": "invalid file name"}), 400
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)

    asr = audio_to_text_result(filepath, language=whisper_lang)
    user_text = asr.text.strip()
    if not user_text:
        return jsonify({"error": "empty transcript"}), 400

    _chat_store.append(cid, "user", user_text)
    hist = _chat_store.get(cid)
    reply = doctor_reply(hist, output_lang_code=out_lang, user_text=user_text)
    _chat_store.append(cid, "assistant", reply.assistant_message)
    return jsonify(
        {
            "transcription": user_text,
            "detected_language": asr.detected_language,
            "assistant_message": reply.assistant_message,
        }
    )


@app.route("/api/chat/finish", methods=["POST"])
def chat_finish():
    data = request.get_json(silent=True) or {}
    cid = str(data.get("conversation_id") or "").strip()
    if not cid:
        return jsonify({"error": "conversation_id required"}), 400
    out_lang = normalize_note_language(str(data.get("output_language") or "en"))
    hist = _chat_store.get(cid)
    summary = build_visit_summary(hist, output_lang_code=out_lang)
    return jsonify({"summary": summary.__dict__})


if __name__ == "__main__":
    # macOS often uses port 5000 for AirPlay Receiver; default to 5001 to avoid "Address already in use".
    port = int(os.environ.get("PORT", 5001))
    # Single process by default: avoids loading Whisper twice (reloader spawns a child process).
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "0").lower() in ("1", "true", "yes")
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    app.run(debug=debug, host=host, port=port, use_reloader=use_reloader)
