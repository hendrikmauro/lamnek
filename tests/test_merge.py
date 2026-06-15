"""Tests für Diarization + ASR Merge-Logik."""

from __future__ import annotations

from lamnek.asr import Segment
from lamnek.diarization import DiarizationSegment
from lamnek.merge import merge_results


def test_merge_without_diarization() -> None:
    transcription = [
        Segment(start=0.0, end=1.0, text="Hallo."),
        Segment(start=1.5, end=2.5, text="Welt."),
    ]
    merged = merge_results([], transcription)
    assert len(merged) == 2
    assert all(seg["speaker"] == "SPEAKER_00" for seg in merged)


def test_merge_with_diarization() -> None:
    diarization = [
        DiarizationSegment(start=0.0, end=1.2, speaker="SPEAKER_A"),
        DiarizationSegment(start=1.2, end=3.0, speaker="SPEAKER_B"),
    ]
    transcription = [
        Segment(start=0.0, end=1.0, text="Hallo."),
        Segment(start=1.5, end=2.5, text="Welt."),
    ]
    merged = merge_results(diarization, transcription)
    assert merged[0]["speaker"] == "SPEAKER_A"
    assert merged[1]["speaker"] == "SPEAKER_B"


def test_merge_with_speaker_names() -> None:
    diarization = [
        DiarizationSegment(start=0.0, end=1.2, speaker="SPEAKER_A"),
        DiarizationSegment(start=1.2, end=3.0, speaker="SPEAKER_B"),
    ]
    transcription = [
        Segment(start=0.0, end=1.0, text="Hallo."),
        Segment(start=1.5, end=2.5, text="Welt."),
    ]
    merged = merge_results(diarization, transcription, speaker_names="I,B")
    assert merged[0]["speaker"] == "I"
    assert merged[1]["speaker"] == "B"


def test_merge_with_too_few_names() -> None:
    diarization = [
        DiarizationSegment(start=0.0, end=1.2, speaker="SPEAKER_A"),
        DiarizationSegment(start=1.2, end=3.0, speaker="SPEAKER_B"),
    ]
    transcription = [
        Segment(start=0.0, end=1.0, text="Hallo."),
        Segment(start=1.5, end=2.5, text="Welt."),
    ]
    merged = merge_results(diarization, transcription, speaker_names="I")
    assert merged[0]["speaker"] == "I"
    assert merged[1]["speaker"] == "Unbekannt"
