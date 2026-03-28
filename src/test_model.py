from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("models/")
model = AutoModelForTokenClassification.from_pretrained("models/")

# Must match training label order (id 0..4). config may still say LABEL_0; keep this in sync with training.
label_list = ["O", "B-SYMPTOM", "I-SYMPTOM", "B-DURATION", "I-DURATION"]

# NER often tags activity / setting nouns as symptoms; drop single-token false positives only.
_SYMPTOM_DENYLIST = frozenset(
    {
        "lumber",
        "yard",
        "groceries",
        "boxes",
        "furniture",
        "driveway",
        "garage",
        "stairs",
        "shopping",
        "car",
        "truck",
        "load",
        "loads",
        "lifting",
        "lifted",
        "carried",
        "carrying",
        "loading",
        "warehouse",
        "bags",
        "suitcase",
    }
)

# If the patient describes pain but the model only tagged a body area, add "pain" for clarity.
_BODY_HINTS = (
    "back",
    "neck",
    "head",
    "chest",
    "stomach",
    "abdomen",
    "throat",
    "leg",
    "legs",
    "arm",
    "arms",
    "knee",
    "ankle",
    "wrist",
    "shoulder",
    "hip",
    "foot",
    "feet",
    "hand",
    "hands",
    "ear",
    "eyes",
    "eye",
    "tooth",
    "teeth",
    "jaw",
    "ribs",
    "rib",
    "side",
    "lower",
    "upper",
)

_PAIN_PHRASES = (
    "killing me",
    "kills me",
    "hurts",
    "hurting",
    "aching",
    "ache",
    "pain",
    "painful",
    "sore",
    "soreness",
    "stiff",
    "stiffness",
    "throbbing",
    "sharp",
    "dull",
    "burning",
    "cramping",
    "cramp",
    "unbearable",
    "excruciating",
)


def _merge_wordpieces(token_label_pairs):
    """BERT WordPiece splits words (e.g. headache -> head + ##ache). Join them before building notes."""
    merged = []
    current_word = ""
    current_label = None

    for token, label in token_label_pairs:
        if token in ["[CLS]", "[SEP]", "", "[UNK]"]:
            continue
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                merged.append((current_word, current_label))
            current_word = token
            current_label = label

    if current_word:
        merged.append((current_word, current_label))

    return merged


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=2)[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    results = []
    for token, pred in zip(tokens, preds):
        if token not in ["[CLS]", "[SEP]", "", "[UNK]"]:
            results.append((token, label_list[pred.item()]))

    return _merge_wordpieces(results)


def clean_predictions(predictions):
    stopwords = {
        "i", "me", "my", "and", "or", "but",
        "the", "a", "is", "was", "are",
        "after", "before",
    }

    cleaned = []

    for word, label in predictions:
        if word in stopwords:
            continue

        if len(word) <= 2:
            continue

        cleaned.append((word, label))

    return cleaned


def enhance_predictions(predictions):
    enhanced = []

    for i, (word, label) in enumerate(predictions):

        # detect pain phrases (link prior token as symptom anchor)
        if word in ["pain", "hurt", "hurts", "aching", "ache", "sore"]:
            if i > 0:
                enhanced.append((predictions[i - 1][0], "B-SYMPTOM"))
                enhanced.append((word, "I-SYMPTOM"))
                continue

        # breathing issues
        if word in ["breath", "breathing"]:
            enhanced.append((word, "B-SYMPTOM"))
            continue

        # movement issues
        if word in ["move", "moving"]:
            enhanced.append((word, "B-SYMPTOM"))
            continue

        enhanced.append((word, label))

    return enhanced


def to_structured_output(predictions):
    symptoms = []
    symptom_chunk = []
    durations = []
    duration_chunk = []

    for word, label in predictions:
        if "SYMPTOM" in label:
            if duration_chunk:
                durations.append(" ".join(duration_chunk))
                duration_chunk = []
            symptom_chunk.append(word)
        elif "DURATION" in label:
            if symptom_chunk:
                symptoms.append(" ".join(symptom_chunk))
                symptom_chunk = []
            duration_chunk.append(word)
        else:
            if symptom_chunk:
                symptoms.append(" ".join(symptom_chunk))
                symptom_chunk = []
            if duration_chunk:
                durations.append(" ".join(duration_chunk))
                duration_chunk = []

    if symptom_chunk:
        symptoms.append(" ".join(symptom_chunk))
    if duration_chunk:
        durations.append(" ".join(duration_chunk))

    return {
        "symptoms": symptoms,
        "duration": "; ".join(durations) if durations else "",
    }


def _mentions_pain_language(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _PAIN_PHRASES)


def _enrich_symptom_phrase(phrase: str, full_text: str) -> str:
    if not full_text or not phrase:
        return phrase
    pl = phrase.lower()
    if any(x in pl for x in ("pain", "hurt", "ache", "sore", "cramp", "burn")):
        return phrase
    if not _mentions_pain_language(full_text):
        return phrase
    tokens = pl.split()
    if any(tok in _BODY_HINTS for tok in tokens):
        return f"{phrase} pain"
    return phrase


def _drop_denylist_symptoms(symptoms: list) -> list:
    kept = []
    for s in symptoms:
        if not s or not s.strip():
            continue
        tokens = s.lower().split()
        if len(tokens) == 1 and tokens[0] in _SYMPTOM_DENYLIST:
            continue
        if s.lower() in _SYMPTOM_DENYLIST:
            continue
        kept.append(s)
    return kept


def finalize_structured_notes(transcription: str, structured: dict) -> dict:
    """Filter common NER false positives and align symptom phrases with pain language in the transcript."""
    symptoms = _drop_denylist_symptoms(structured.get("symptoms") or [])
    symptoms = [_enrich_symptom_phrase(s, transcription) for s in symptoms]
    return {
        "symptoms": symptoms,
        "duration": structured.get("duration") or "",
    }
