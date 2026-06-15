"""faster-whisper ASR-Wrapper mit CUDA-Fallback."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from faster_whisper import WhisperModel

import lamnek.cuda_setup as _cuda  # noqa: F401  # side-effects on import


@dataclass(frozen=True)
class TranscriptionInfo:
    """Minimal getypte Transkriptions-Metadaten."""

    language: str
    language_probability: float
    duration: float


@dataclass(frozen=True)
class Segment:
    """Ein ASR-Segment."""

    start: float
    end: float
    text: str


def _detect_device(device: str | None) -> str:
    """Wählt cuda, wenn verfügbar, sonst cpu."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch kann fehlen
        return device or "cpu"


def load_model(model_name: str = "large-v3-turbo", device: str | None = "cuda") -> WhisperModel:
    """Lädt ein faster-whisper Modell mit automatischem CPU-Fallback.

    Args:
        model_name: Name des Whisper-Modells.
        device: Bevorzugtes Gerät ("cuda" oder "cpu"). None → Auto.

    Returns:
        Geladenes WhisperModel.

    Raises:
        SystemExit: Wenn weder CUDA noch CPU funktionieren.
    """
    chosen_device = device if device else _detect_device(device)
    compute = "float16" if chosen_device == "cuda" else "int8"

    try:
        print(f"🔄 Lade Modell '{model_name}' ({chosen_device})...")
        return WhisperModel(model_name, device=chosen_device, compute_type=compute)
    except Exception as e:
        if chosen_device == "cuda":
            print(f"⚠️  Konnte Modell nicht auf CUDA laden: {e}")
            print("   Fallback auf CPU...")
            try:
                return WhisperModel(model_name, device="cpu", compute_type="int8")
            except Exception as e2:
                print(f"❌ Auch CPU-Fallback fehlgeschlagen: {e2}")
                sys.exit(1)
        print(f"❌ Fehler beim Laden des Modells: {e}")
        sys.exit(1)


def _to_segments(seg_iter: Iterable[Any]) -> list[Segment]:
    """Wandelt faster-whisper Segmente in getypte Segmente um."""
    out: list[Segment] = []
    for raw in seg_iter:
        seg: Any = raw
        out.append(
            Segment(
                start=round(float(seg.start), 3),
                end=round(float(seg.end), 3),
                text=str(seg.text).strip(),
            )
        )
    return out


def transcribe_audio(
    audio_path: str,
    model_name: str = "large-v3-turbo",
    language: str | None = None,
    device: str | None = "cuda",
    vad: bool = False,
    beam_size: int = 5,
) -> tuple[list[Segment], TranscriptionInfo]:
    """Transkribiert eine Audiodatei mit faster-whisper.

    Args:
        audio_path: Pfad zur Audiodatei (idealerweise 16 kHz mono WAV).
        model_name: Whisper-Modell.
        language: Zwei-Buchstaben-Code oder None für automatische Erkennung.
        device: Bevorzugtes Gerät.
        vad: Voice Activity Detection aktivieren.
        beam_size: Beam size für die Decodierung.

    Returns:
        Tuple aus Liste von Segmenten und Transkriptions-Info.
    """
    model = load_model(model_name, device)

    lang = language if language and language != "auto" else None
    if lang:
        print(f"   Sprache: {lang}")

    start_time = time.time()
    seg_iter, raw_info = model.transcribe(
        audio_path,
        language=lang,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 500} if vad else None,
    )
    segments = _to_segments(seg_iter)
    elapsed = time.time() - start_time

    info = TranscriptionInfo(
        language=str(raw_info.language),
        language_probability=float(raw_info.language_probability),
        duration=float(raw_info.duration),
    )

    duration_min = info.duration / 60
    speedup = info.duration / elapsed if elapsed > 0 else 0.0
    print(f"   ✅ {len(segments)} Segmente, {elapsed:.1f}s")
    print(f"   Sprache: {info.language} ({info.language_probability:.1%})")
    print(f"   {duration_min:.1f} Min Audio, {speedup:.1f}x Echtzeit")

    return segments, info
