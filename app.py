from flask import Flask, render_template, request
import os
import sys

from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

# allow src imports
sys.path.append(os.path.abspath("src"))

from audio_pipeline import audio_to_text
from clinical_note_generator import NOTE_SCHEMA_KEYS, generate_clinical_document

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB — room for longer mic recordings

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Human-readable section titles for the template
SECTION_TITLES = {
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


@app.route("/", methods=["GET", "POST"])
def index():
    transcription = ""
    clinical_note = None
    error_message = None

    if request.method == "POST":
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
                    transcription = audio_to_text(filepath)
                    clinical_note = generate_clinical_document(transcription)
                except Exception as exc:
                    error_message = f"Processing failed: {exc}"

    sections = None
    if clinical_note:
        sections = [
            {"id": k, "title": SECTION_TITLES[k], "body": clinical_note.get(k, "")}
            for k in NOTE_SCHEMA_KEYS
            if clinical_note.get(k)
        ]

    return render_template(
        "index.html",
        transcription=transcription,
        clinical_note=clinical_note,
        sections=sections,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
