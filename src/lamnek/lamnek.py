"""Lamnek/Krell-Formatierung für wissenschaftliche Transkripte."""

from __future__ import annotations

import datetime


def pause_marker(gap: float) -> str | None:
    """Wählt Lamnek-Pausenmarker basierend auf Lücke in Sekunden."""
    if gap >= 2.0:
        return f"({int(round(gap))}s)"
    elif gap >= 1.0:
        return "(---)"
    elif gap >= 0.5:
        return "(--)"
    return None


def wrap_text(text: str, width: int = 95) -> list[str]:
    """Bricht Text an Leerzeichen um, sodass keine Zeile länger als width ist.

    width <= 0 bedeutet: kein Umbruch.
    """
    if width <= 0:
        return [text]
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _collect_utterances(
    segments: list[dict[str, object]],
    anonymize: bool = False,
) -> list[tuple[str, str]]:
    """Fasst aufeinanderfolgende Segmente desselben Sprechers zu Äußerungen."""
    speaker_map: dict[str, str] = {}
    speaker_counter = 1

    utterances: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_texts: list[str] = []
    prev_end = 0.0

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        speaker = str(seg.get("speaker", "SPEAKER_00"))
        if anonymize:
            if speaker not in speaker_map:
                speaker_map[speaker] = f"<Person_{speaker_counter}>"
                speaker_counter += 1
            speaker = speaker_map[speaker]

        gap = float(seg.get("start", 0.0)) - prev_end  # type: ignore[arg-type]
        if prev_end > 0 and gap > 0.5:
            pause = pause_marker(gap)
            if pause and current_texts:
                current_texts[-1] = f"{current_texts[-1]} {pause}"

        if speaker != current_speaker:
            if current_texts:
                utterances.append((current_speaker or "SPEAKER_00", " ".join(current_texts)))
            current_speaker = speaker
            current_texts = [text]
        else:
            current_texts.append(text)

        prev_end = float(seg.get("end", 0.0))  # type: ignore[arg-type]

    if current_texts:
        utterances.append((current_speaker or "SPEAKER_00", " ".join(current_texts)))

    return utterances


def basic_lamnek(
    segments: list[dict[str, object]],
    anonymize: bool = False,
    line_numbers: bool = True,
    wrap_width: int = 0,
) -> str:
    """Konvertiert Segmente in wissenschaftliches Lamnek/Krell-Format.

    Regeln:
    - Sprecherlabel nur bei Wechsel, dann Text fortlaufend ohne Label
    - Pausen inline am Ende einer Äußerung
    - Zeilennummern fortlaufend
    - wrap_width=0: keine automatischen Zeilenumbrüche
    """
    utterances = _collect_utterances(segments, anonymize=anonymize)

    lines: list[str] = []
    line_num = 1
    prev_speaker: str | None = None

    for speaker, text in utterances:
        speaker_changed = speaker != prev_speaker
        if speaker_changed and prev_speaker is not None:
            lines.append("")

        wrapped_lines = wrap_text(text, width=wrap_width) if wrap_width > 0 else [text]
        for j, wline in enumerate(wrapped_lines):
            if j == 0 and speaker_changed:
                display = f"{speaker}: {wline}"
            elif j == 0:
                display = wline
            else:
                display = f"      {wline}"
            lines.append(f"{line_num:3d}  {display}" if line_numbers else display)
            line_num += 1

        prev_speaker = speaker

    return "\n".join(lines)


def generate_header(
    audio_path: str,
    segments: list[dict[str, object]],
    speaker_names: str | None = None,
    project_info: dict[str, object] | None = None,
) -> str:
    """Generiert wissenschaftliche Kopfzeile nach Lamnek/Krell."""
    duration_sec = float(segments[-1].get("end", 0.0)) if segments else 0.0  # type: ignore[arg-type]
    duration_min = max(1, int(round(duration_sec / 60))) if duration_sec > 0 else 0
    minute_word = "Minute" if duration_min == 1 else "Minuten"

    unique_speakers = sorted({str(s.get("speaker", "SPEAKER_00")) for s in segments})
    name_mapping: dict[str, str | None] = {}
    if speaker_names:
        name_list = [n.strip() for n in speaker_names.split(",")]
        for i, spk in enumerate(unique_speakers):
            if i < len(name_list):
                name_mapping[spk] = name_list[i]

    speaker_labels = [f"  {name_mapping.get(spk, spk)}" for spk in unique_speakers]

    now = datetime.datetime.now()
    date_str = now.strftime("%d.%m.%Y")
    time_str = now.strftime("%H:%M Uhr")

    pi = project_info or {}
    projekt = pi.get("projekt", "")
    pseudonym = pi.get("pseudonym", "")
    interviewer = pi.get("interviewer", "")
    ort = pi.get("ort", "")
    bemerkungen = pi.get("bemerkungen", "Keine besonderen Auffälligkeiten")

    return f"""Auszug aus einem ausführlichen Transkript

Projekt           {projekt}
ID
Interviewer       {interviewer}
Pseudonym         {pseudonym}
Datum             {date_str}
Zeit              {time_str}
Dauer             {duration_min} {minute_word}
Ort               {ort}
Bemerkungen       {bemerkungen}

Sprecher:
{chr(10).join(speaker_labels)}

"""
