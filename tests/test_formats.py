"""Tests für Exportformate."""

from __future__ import annotations

import json
from pathlib import Path

from lamnek.asr import TranscriptionInfo
from lamnek.formats import export_json, export_lamnek, export_srt, export_txt
from lamnek.lamnek import basic_lamnek


def test_export_txt(tmp_path) -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hallo.", "speaker": "A"},
    ]
    base = str(tmp_path / "out")
    path = export_txt(segments, base)
    assert Path(path).exists()
    content = Path(path).read_text(encoding="utf-8")
    assert "Hallo." in content
    assert "A:" in content


def test_export_srt(tmp_path) -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hallo.", "speaker": "A"},
    ]
    base = str(tmp_path / "out")
    path = export_srt(segments, base)
    content = Path(path).read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,000" in content
    assert "[A] Hallo." in content


def test_export_json(tmp_path) -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hallo.", "speaker": "A"},
    ]
    base = str(tmp_path / "out")
    info = TranscriptionInfo(language="de", language_probability=0.98, duration=1.0)
    path = export_json(segments, "", base, info)
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    assert content["info"]["language"] == "de"
    assert len(content["segments"]) == 1


def test_export_lamnek(tmp_path) -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hallo.", "speaker": "A"},
    ]
    lamnek = basic_lamnek(segments, line_numbers=False)
    base = str(tmp_path / "out")
    path = export_lamnek(segments, lamnek, base)
    content = Path(path).read_text(encoding="utf-8")
    assert "A: Hallo." in content
