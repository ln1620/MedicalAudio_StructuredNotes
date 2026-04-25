from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ASRResult:
    text: str
    engine: Literal["google", "whisper"]
    detected_language: str | None = None  # BCP-47 (preferred) or ISO code when known


class ASREngine:
    name: str

    def transcribe(self, audio_path: str, *, language: str | None) -> ASRResult:
        raise NotImplementedError()

