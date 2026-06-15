"""Exportformate: TXT, SRT, JSON für Transkriptions-Ergebnisse."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path

from lamnek.asr import TranscriptionInfo
from lamnek.lamnek import generate_header


def _format_timestamp(seconds: float) -> str:
    """SRT-konformer Zeitstempel HH:MM:SS,mmm."""
    td = timedelta(seconds=seconds)
    h, m = td.seconds // 3600, (td.seconds % 3600) // 60
    s, ms = td.seconds % 60, int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _out_path(base_name: str, suffix: str, ext: str) -> str:
    """Erzeugt einen Ausgabepfad."""
    return f"{base_name}_{suffix}.{ext}"


def export_txt(
    merged_segments: list[dict[str, object]],
    base_name: str,
    speaker_names: str | None = None,
    project_info: dict[str, object] | None = None,
) -> str:
    """Exportiert ein zeitgestempeltes TXT-Transkript."""
    path = _out_path(base_name, "transkript", "txt")
    header = generate_header(base_name, merged_segments, speaker_names, project_info)
    with Path(path).open("w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n")
        for seg in merged_segments:
            speaker = seg.get("speaker", "SPEAKER_00")
            f.write(f"[{_format_timestamp(float(seg['start']))}] {speaker}: {seg['text']}\n")  # type: ignore[arg-type]
    return path


def export_srt(merged_segments: list[dict[str, object]], base_name: str) -> str:
    """Exportiert SRT-Untertitel mit Sprecher-Präfixen."""
    path = _out_path(base_name, "transkript", "srt")
    srt_lines: list[str] = []
    for i, seg in enumerate(merged_segments, 1):
        srt_lines.append(str(i))
        srt_lines.append(
            f"{_format_timestamp(float(seg['start']))} --> {_format_timestamp(float(seg['end']))}"  # type: ignore[arg-type]
        )
        speaker = seg.get("speaker", "SPEAKER_00")
        srt_lines.append(f"[{speaker}] {seg['text']}")
        srt_lines.append("")
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    return path


def export_json(
    merged_segments: list[dict[str, object]],
    lamnek_text: str,
    base_name: str,
    info: TranscriptionInfo | None = None,
) -> str:
    """Exportiert das Ergebnis als JSON."""
    path = _out_path(base_name, "transkript", "json")
    data = {
        "segments": merged_segments,
        "lamnek": lamnek_text,
        "info": {
            "language": info.language if info else "unknown",
            "language_probability": info.language_probability if info else 1.0,
        },
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def export_lamnek(
    merged_segments: list[dict[str, object]],
    lamnek_text: str,
    base_name: str,
    speaker_names: str | None = None,
    project_info: dict[str, object] | None = None,
) -> str:
    """Exportiert das Lamnek/Krell-formatierte Transkript."""
    path = _out_path(base_name, "lamnek", "txt")
    header = generate_header(base_name, merged_segments, speaker_names, project_info)
    with Path(path).open("w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n")
        f.write(lamnek_text)
    return path


def generate_outputs(
    merged_segments: list[dict[str, object]],
    lamnek_text: str,
    base_name: str,
    formats: Iterable[str],
    info: TranscriptionInfo | None = None,
    project_info: dict[str, object] | None = None,
    speaker_names: str | None = None,
    is_lamnek: bool = False,
    explicit_path: str | None = None,
) -> list[tuple[str, str]]:
    """Generiert alle gewünschten Ausgabeformate.

    Args:
        merged_segments: Liste von kombinierten Segmenten.
        lamnek_text: Berechneter Lamnek-Text.
        base_name: Basisname für die Ausgabedateien.
        formats: Gewünschte Formate.
        info: ASR-Metadaten.
        project_info: Projektdaten für den Header.
        speaker_names: Komma-getrennte Sprechernamen.
        is_lamnek: Ob Lamnek-Modus aktiv ist.
        explicit_path: Optionaler expliziter Ausgabepfad.

    Returns:
        Liste von (pfad, format)-Tupeln.
    """
    outputs: list[tuple[str, str]] = []
    formats_set = set(formats)

    def _resolve_path(suffix: str, ext: str) -> str:
        if explicit_path and len(formats_set) == 1:
            return explicit_path
        return _out_path(base_name, suffix, ext)

    if "lamnek" in formats_set:
        path = _resolve_path("lamnek", "txt")
        export_lamnek(
            merged_segments, lamnek_text, base_name, speaker_names, project_info
        )
        outputs.append((path, "lamnek"))

    if "txt" in formats_set and not is_lamnek:
        path = _resolve_path("transkript", "txt")
        export_txt(merged_segments, base_name, speaker_names, project_info)
        outputs.append((path, "txt"))

    if "srt" in formats_set:
        path = _resolve_path("transkript", "srt")
        export_srt(merged_segments, base_name)
        outputs.append((path, "srt"))

    if "json" in formats_set:
        path = _resolve_path("transkript", "json")
        export_json(merged_segments, lamnek_text, base_name, info)
        outputs.append((path, "json"))

    return outputs
