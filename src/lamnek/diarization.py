"""pyannote Speaker-Diarization Wrapper."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import lamnek.cuda_setup as _cuda  # noqa: F401  # side-effects on import


class DiarizationSegment:
    """Ein Diarization-Segment mit Start, Ende und Sprecher-Label."""

    def __init__(self, start: float, end: float, speaker: str) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker

    def to_dict(self) -> dict[str, object]:
        """Gibt das Segment als Dictionary zurück."""
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
        }


def ensure_hf_token() -> str:
    """Prüft, dass HF_TOKEN als Umgebungsvariable gesetzt ist.

    Returns:
        Der Token-Wert.

    Raises:
        SystemExit: Wenn HF_TOKEN fehlt.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Fehlende Umgebungsvariable: HF_TOKEN")
        print("   Bitte setze sie vor dem Start:")
        print("   export HF_TOKEN=hf_...")
        sys.exit(1)
    return token


def diarize(
    audio_path: str,
    num_speakers: int | None = None,
    model_id: str = "pyannote/speaker-diarization-3.1",
) -> list[DiarizationSegment]:
    """Führt Sprecher-Diarisierung mit pyannote durch.

    Args:
        audio_path: Pfad zur Audiodatei.
        num_speakers: Erwartete Sprecherzahl oder None.
        model_id: pyannote Pipeline-ID.

    Returns:
        Liste von DiarizationSegmenten.
    """
    print("👥 Diarisierung (Sprecher-Erkennung)...")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        print("   ❌ pyannote.audio nicht installiert.")
        print("   Installiere: pip install pyannote.audio")
        return []

    try:
        pipeline: Any = Pipeline.from_pretrained(model_id)
    except Exception as e:
        print(f"   ❌ pyannote Modell konnte nicht geladen werden: {e}")
        print("   Stelle sicher, dass HF_TOKEN gesetzt ist.")
        return []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    try:
        pipeline.to(device)
    except Exception as e:
        print(f"   ⚠️  Konnte pyannote nicht auf {device} laden: {e}")
        if device.type == "cuda":
            print("   Fallback auf CPU...")
            pipeline.to(torch.device("cpu"))

    params: dict[str, object] = {}
    if num_speakers:
        params["num_speakers"] = num_speakers

    try:
        start_time = time.time()
        diarization = pipeline(audio_path, **params)
        annotation = diarization.speaker_diarization

        segments: list[DiarizationSegment] = []
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=str(speaker),
                )
            )

        elapsed = time.time() - start_time
        print(f"   ✅ {len(segments)} Segmente, {elapsed:.1f}s")
        if segments:
            unique = sorted({s.speaker for s in segments})
            print(f"   Sprecher: {', '.join(unique)}")
        return segments
    except Exception as e:
        print(f"   ⚠️  Diarisierung fehlgeschlagen: {e}")
        print("   Fahre ohne Sprecher-Erkennung fort...")
        return []
