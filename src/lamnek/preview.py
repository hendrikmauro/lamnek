"""Sprecher-Vorschau für die manuelle Zuordnung."""

from __future__ import annotations

from lamnek.formats import _format_timestamp


def show_speaker_preview(merged_segments: list[dict[str, object]]) -> None:
    """Zeigt repräsentative Text-Auszüge pro Sprecher."""
    print("\n🔍 Sprecher-Vorschau (für manuelle Zuordnung):")
    print("=" * 60)

    speaker_segments: dict[str, list[dict[str, object]]] = {}
    for seg in merged_segments:
        spk = str(seg.get("speaker", "SPEAKER_00"))
        speaker_segments.setdefault(spk, []).append(seg)

    for spk in sorted(speaker_segments.keys()):
        segments = speaker_segments[spk]
        total_time = sum(
            (float(s["end"]) - float(s["start"]))  # type: ignore[arg-type,misc]
            for s in segments
        )

        seen_texts: set[str] = set()
        preview_segments: list[dict[str, object]] = []
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                preview_segments.append(seg)
            if len(preview_segments) >= 3:
                break
        if len(preview_segments) < 3 and segments:
            preview_segments = segments[:3]

        print(f"\n📢 {spk} ({len(segments)} Segmente, {total_time:.1f}s gesamt):")
        print("-" * 60)
        for i, seg in enumerate(preview_segments, 1):
            text = str(seg.get("text", "")).strip()
            short = f"{text[:120]}..." if len(text) > 120 else text
            print(f"   {i}. [{_format_timestamp(float(seg['start']))}] {short}")  # type: ignore[arg-type]

    print("\n" + "=" * 60)
    print("💡 Tipp: Nutze --sprecher 'Name1,Name2,Name3'")
