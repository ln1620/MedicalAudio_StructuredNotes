import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from audio_pipeline import audio_to_text
from clinical_note_generator import NOTE_SCHEMA_KEYS, generate_clinical_document

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
audio_file = os.path.join(ROOT, "sample4.wav")

text = audio_to_text(audio_file)
print("Transcription:", text)

note = generate_clinical_document(text)
print("\n--- Clinical note ---")
for k in NOTE_SCHEMA_KEYS:
    if note.get(k):
        print(f"\n{k}:\n{note[k]}")
print("\n_source:", note.get("_source"))
