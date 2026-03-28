"""
Export the Hugging Face NER model under models/ to a single .pkl bundle.

The bundle contains:
  - state_dict (weights)
  - config dict (architecture + labels)
  - tokenizer (PreTrainedTokenizer, same Python/transformers version recommended)
  - label_list: BIO tags used in training / inference

Usage (from project root, with .venv activated):

  python src/export_model_to_pkl.py
  python src/export_model_to_pkl.py --output outputs/medical_ner_model.pkl

Load example:

  from export_model_to_pkl import load_ner_from_pkl
  model, tokenizer, label_list = load_ner_from_pkl("outputs/medical_ner_model.pkl")

Security: only unpickle files you created (pickle can execute code if abused).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Default BIO labels (match src/test_model.py)
DEFAULT_LABEL_LIST = ["O", "B-SYMPTOM", "I-SYMPTOM", "B-DURATION", "I-DURATION"]


def export_to_pkl(
    input_dir: Path,
    output_path: Path,
    *,
    weights_only: bool = False,
    label_list: list[str] | None = None,
) -> None:
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {input_dir}")

    model = AutoModelForTokenClassification.from_pretrained(input_dir)
    model.eval()
    model.cpu()

    bundle: dict = {
        "format_version": 2,
        "state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "label_list": label_list or DEFAULT_LABEL_LIST,
    }

    if not weights_only:
        tok = AutoTokenizer.from_pretrained(input_dir)
        bundle["tokenizer"] = tok
    bundle["weights_only"] = weights_only

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MiB)")


def _config_from_dict(config_dict: dict):
    """Build a PretrainedConfig from a dict (works across transformers versions)."""
    from transformers import CONFIG_MAPPING

    model_type = config_dict.get("model_type")
    if not model_type or model_type not in CONFIG_MAPPING:
        raise ValueError(f"Unknown or missing model_type in config: {model_type!r}")
    cls = CONFIG_MAPPING[model_type]
    return cls.from_dict(config_dict)


def load_ner_from_pkl(pkl_path: str | Path):
    """
    Load model + tokenizer from a bundle created by export_to_pkl.

    Returns:
        (model, tokenizer, label_list)  — tokenizer may be None if exported with --weights-only
    """
    from transformers import AutoModelForTokenClassification

    pkl_path = Path(pkl_path).resolve()
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    if bundle.get("format_version") not in (1, 2):
        raise ValueError(f"Unknown bundle format: {bundle.get('format_version')}")

    config = _config_from_dict(bundle["config"])
    model = AutoModelForTokenClassification.from_config(config)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    tokenizer = bundle.get("tokenizer")
    label_list = bundle.get("label_list") or DEFAULT_LABEL_LIST
    return model, tokenizer, label_list


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Export models/ NER checkpoint to .pkl")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "models",
        help="Hugging Face saved model directory (default: ./models)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "models" / "medical_ner_model.pkl",
        help="Output .pkl path (default: models/medical_ner_model.pkl)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Do not pickle tokenizer (smaller; keep tokenizer JSON files next to this pkl for loading)",
    )
    args = parser.parse_args()

    try:
        export_to_pkl(args.input, args.output, weights_only=args.weights_only)
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
