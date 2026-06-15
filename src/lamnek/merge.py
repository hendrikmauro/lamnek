"""Kombiniert Diarization-Segmente mit ASR-Segmenten."""

from __future__ import annotations

from lamnek.asr import Segment
from lamnek.diarization import DiarizationSegment


def merge_results(
    diarization_segments: list[DiarizationSegment],
    transcription_segments: list[Segment],
    speaker_names: str | None = None,
) -> list[dict[str, object]]:
    """Ordnet jedem Transkriptions-Segment einen Sprecher zu.

    Args:
        diarization_segments: Liste von pyannote DiarizationSegmenten.
        transcription_segments: Liste von ASR-Segmenten.
        speaker_names: Komma-getrennte Sprechernamen oder None.

    Returns:
        Liste von Dictionaries mit start, end, text, speaker.
    """
    if not diarization_segments:
        merged: list[dict[str, object]] = [
            {"start": seg.start, "end": seg.end, "text": seg.text, "speaker": "SPEAKER_00"}
            for seg in transcription_segments
        ]
        return merged

    print("🔗 Kombiniere Sprecher + Text...")
    merged = []
    for seg in transcription_segments:
        best_speaker = "SPEAKER_UNKNOWN"
        max_overlap = 0.0
        for dseg in diarization_segments:
            overlap_start = max(seg.start, dseg.start)
            overlap_end = min(seg.end, dseg.end)
            overlap = overlap_end - overlap_start
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = dseg.speaker
        merged.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": best_speaker,
            }
        )

    if speaker_names:
        merged = _apply_speaker_names(merged, speaker_names)

    return merged


def _apply_speaker_names(
    merged: list[dict[str, object]],
    speaker_names: str,
) -> list[dict[str, object]]:
    """Weist Sprechernamen basierend auf Sprechdauer oder 1:1-Zuordnung zu."""
    unique_speakers = sorted({str(s["speaker"]) for s in merged})
    name_list = [n.strip() for n in speaker_names.split(",")]

    mapping: dict[str, str] = {}
    if len(name_list) == len(unique_speakers):
        mapping = dict(zip(unique_speakers, name_list, strict=True))
        print(f"   Sprecher benannt: {', '.join(name_list)}")
    elif len(name_list) < len(unique_speakers):
        speaker_durations: dict[str, float] = {}
        for spk in unique_speakers:
            spk_name = str(spk)
            duration = sum(
                (float(s["end"]) - float(s["start"]))  # type: ignore[arg-type,misc]
                for s in merged if str(s["speaker"]) == spk_name
            )
            speaker_durations[spk_name] = duration
        sorted_speakers = sorted(
            speaker_durations.items(), key=lambda x: x[1], reverse=True
        )
        for i, (spk_key, _) in enumerate(sorted_speakers):
            mapping[str(spk_key)] = name_list[i] if i < len(name_list) else "Unbekannt"

        assigned = [name for name in mapping.values() if name != "Unbekannt"]
        unassigned = len([s for s in mapping.values() if s == "Unbekannt"])
        print(f"   Sprecher benannt: {', '.join(assigned)}")
        if unassigned > 0:
            print(f"   {unassigned} weitere Sprecher → Unbekannt")
    else:
        print(
            f"   ⚠️  {len(name_list)} Namen für {len(unique_speakers)} Sprecher — ignoriert"
        )

    if mapping:
        for seg in merged:
            seg["speaker"] = mapping.get(str(seg["speaker"]), str(seg["speaker"]))

    return merged
