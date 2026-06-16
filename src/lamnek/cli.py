"""Click-basierte Kommandozeile für lamnek und lamnek-simple."""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import click

import lamnek.cuda_setup as _cuda  # noqa: F401  # side-effects on import
from lamnek import __version__
from lamnek.asr import transcribe_audio
from lamnek.audio import preprocess_audio
from lamnek.diarization import diarize, ensure_hf_token
from lamnek.formats import generate_outputs
from lamnek.lamnek import basic_lamnek
from lamnek.merge import merge_results
from lamnek.preview import show_speaker_preview

FORMATS_SIMPLE = {"txt", "srt", "json", "alle"}
FORMATS_PRO = {"txt", "srt", "json", "lamnek", "alle"}


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="lamnek")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Lamnek — wissenschaftliche Interview-Transkription."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command(name="transcribe")
@click.argument("audio", type=click.Path(exists=True, dir_okay=False))
@click.option("--sprache", "-s", default="auto", help="de, en, es, pt, auto")
@click.option(
    "--gerät", "-g", default="cuda", type=click.Choice(["cuda", "cpu"]), help="cuda oder cpu"
)
@click.option("--modell", "-m", default="large-v3-turbo", help="Whisper-Modell")
@click.option(
    "--format", "-f", "fmt", default="txt", type=click.Choice(sorted(FORMATS_PRO)),
    help="Ausgabeformat"
)
@click.option("--output", "-o", default=None, help="Ausgabedatei (ohne Erweiterung)")
@click.option("--lamnek/--no-lamnek", default=False, help="Lamnek/Krell-Format aktivieren")
@click.option("--anonymisieren/--no-anonymisieren", default=False, help="Sprecher anonymisieren")
@click.option("--srt", "srt_flag", is_flag=True, help="SRT zusätzlich ausgeben")
@click.option("--json", "json_flag", is_flag=True, help="JSON zusätzlich ausgeben")
@click.option("--sprecher", default=None, help='Sprecher-Namen: "Name1,Name2,..."')
@click.option("--anzahl-sprecher", type=int, default=None, help="Erwartete Anzahl Sprecher")
@click.option("--no-diarize", is_flag=True, help="Diarisierung überspringen")
@click.option("--projekt", default="", help="Projektname für Kopfzeile")
@click.option("--pseudonym", default="", help="Pseudonym der Befragten")
@click.option("--interviewer", default="", help="Name des Interviewers")
@click.option("--ort", default="", help="Ort des Interviews")
@click.option(
    "--bemerkungen", default="Keine besonderen Auffälligkeiten", help="Bemerkungen"
)
@click.option("--no-zeilennummern", is_flag=True, help="Zeilennummern deaktivieren")
@click.option(
    "--zeilen-breite", type=int, default=0, help="Max. Zeilenlänge (0 = kein Umbruch)"
)
@click.option("--vorschau", is_flag=True, help="Sprecher-Vorschau anzeigen und beenden")
def transcribe(
    audio: str,
    sprache: str,
    gerät: str,
    modell: str,
    fmt: str,
    output: str | None,
    lamnek: bool,
    anonymisieren: bool,
    srt_flag: bool,
    json_flag: bool,
    sprecher: str | None,
    anzahl_sprecher: int | None,
    no_diarize: bool,
    projekt: str,
    pseudonym: str,
    interviewer: str,
    ort: str,
    bemerkungen: str,
    no_zeilennummern: bool,
    zeilen_breite: int,
    vorschau: bool,
) -> None:
    """Wissenschaftliche Transkription mit optionaler Diarization."""
    _run_transcribe(
        audio=audio,
        sprache=sprache,
        gerät=gerät,
        modell=modell,
        fmt=fmt,
        output=output,
        lamnek=lamnek,
        anonymisieren=anonymisieren,
        srt_flag=srt_flag,
        json_flag=json_flag,
        sprecher=sprecher,
        anzahl_sprecher=anzahl_sprecher,
        no_diarize=no_diarize,
        projekt=projekt,
        pseudonym=pseudonym,
        interviewer=interviewer,
        ort=ort,
        bemerkungen=bemerkungen,
        no_zeilennummern=no_zeilennummern,
        zeilen_breite=zeilen_breite,
        vorschau=vorschau,
        simple=False,
    )


@cli.command(name="transcribe-pro")
@click.argument("audio", type=click.Path(exists=True, dir_okay=False))
@click.option("--sprache", "-s", default="auto", help="de, en, es, pt, auto")
@click.option(
    "--gerät", "-g", default="cuda", type=click.Choice(["cuda", "cpu"]), help="cuda oder cpu"
)
@click.option("--modell", "-m", default="large-v3-turbo", help="Whisper-Modell")
@click.option(
    "--format", "-f", "fmt", default="lamnek", type=click.Choice(sorted(FORMATS_PRO)),
    help="Ausgabeformat"
)
@click.option("--output", "-o", default=None, help="Ausgabedatei (ohne Erweiterung)")
@click.option("--anonymisieren/--no-anonymisieren", default=False, help="Sprecher anonymisieren")
@click.option("--srt", "srt_flag", is_flag=True, help="SRT zusätzlich ausgeben")
@click.option("--json", "json_flag", is_flag=True, help="JSON zusätzlich ausgeben")
@click.option("--sprecher", default=None, help='Sprecher-Namen: "Name1,Name2,..."')
@click.option("--anzahl-sprecher", type=int, default=None, help="Erwartete Anzahl Sprecher")
@click.option("--no-diarize", is_flag=True, help="Diarisierung überspringen")
@click.option("--projekt", default="", help="Projektname für Kopfzeile")
@click.option("--pseudonym", default="", help="Pseudonym der Befragten")
@click.option("--interviewer", default="", help="Name des Interviewers")
@click.option("--ort", default="", help="Ort des Interviews")
@click.option(
    "--bemerkungen", default="Keine besonderen Auffälligkeiten", help="Bemerkungen"
)
@click.option("--no-zeilennummern", is_flag=True, help="Zeilennummern deaktivieren")
@click.option(
    "--zeilen-breite", type=int, default=0, help="Max. Zeilenlänge (0 = kein Umbruch)"
)
@click.option("--vorschau", is_flag=True, help="Sprecher-Vorschau anzeigen und beenden")
def transcribe_pro(
    audio: str,
    sprache: str,
    gerät: str,
    modell: str,
    fmt: str,
    output: str | None,
    anonymisieren: bool,
    srt_flag: bool,
    json_flag: bool,
    sprecher: str | None,
    anzahl_sprecher: int | None,
    no_diarize: bool,
    projekt: str,
    pseudonym: str,
    interviewer: str,
    ort: str,
    bemerkungen: str,
    no_zeilennummern: bool,
    zeilen_breite: int,
    vorschau: bool,
) -> None:
    """Alias für transcribe --lamnek (wissenschaftlicher Standard)."""
    _run_transcribe(
        audio=audio,
        sprache=sprache,
        gerät=gerät,
        modell=modell,
        fmt=fmt,
        output=output,
        lamnek=True,
        anonymisieren=anonymisieren,
        srt_flag=srt_flag,
        json_flag=json_flag,
        sprecher=sprecher,
        anzahl_sprecher=anzahl_sprecher,
        no_diarize=no_diarize,
        projekt=projekt,
        pseudonym=pseudonym,
        interviewer=interviewer,
        ort=ort,
        bemerkungen=bemerkungen,
        no_zeilennummern=no_zeilennummern,
        zeilen_breite=zeilen_breite,
        vorschau=vorschau,
        simple=False,
    )


def _run_transcribe(
    *,
    audio: str,
    sprache: str,
    gerät: str,
    modell: str,
    fmt: str,
    output: str | None,
    lamnek: bool,
    anonymisieren: bool,
    srt_flag: bool,
    json_flag: bool,
    sprecher: str | None,
    anzahl_sprecher: int | None,
    no_diarize: bool,
    projekt: str,
    pseudonym: str,
    interviewer: str,
    ort: str,
    bemerkungen: str,
    no_zeilennummern: bool,
    zeilen_breite: int,
    vorschau: bool,
    simple: bool,
) -> None:
    """Gemeinsame Pipeline für beide Hauptbefehle."""
    if not no_diarize:
        ensure_hf_token()

    audio_path = Path(audio)
    print(f"\n📁 {audio_path.name} ({audio_path.stat().st_size / (1024 * 1024):.1f} MB)")
    print("=" * 60)

    total_start = time.time()

    wav_path = preprocess_audio(audio)

    try:
        diarization_segments = []
        if not no_diarize:
            diarization_segments = diarize(wav_path, anzahl_sprecher)

        language = None if sprache == "auto" else sprache
        transcription_segments, info = transcribe_audio(
            wav_path, model_name=modell, language=language, device=gerät
        )

        merged = merge_results(diarization_segments, transcription_segments, sprecher)

        if vorschau:
            show_speaker_preview(merged)
            print(f"\n⏱️ Gesamtzeit: {time.time() - total_start:.1f}s")
            return

        project_info: dict[str, object] = {
            "projekt": projekt,
            "pseudonym": pseudonym,
            "interviewer": interviewer,
            "ort": ort,
            "bemerkungen": bemerkungen,
        }

        formats, lamnek_needed = _resolve_formats(
            fmt=fmt,
            lamnek=lamnek,
            srt_flag=srt_flag,
            json_flag=json_flag,
            simple=simple,
        )

        lamnek_text = ""
        if lamnek_needed:
            lamnek_text = basic_lamnek(
                merged,
                anonymize=anonymisieren,
                line_numbers=not no_zeilennummern,
                wrap_width=zeilen_breite,
            )

        explicit_path, base_name = _resolve_output_paths(audio, output)

        outputs = generate_outputs(
            merged,
            lamnek_text,
            base_name,
            formats,
            info,
            project_info,
            speaker_names=sprecher,
            is_lamnek=lamnek,
            explicit_path=explicit_path,
        )

        _print_summary(outputs, total_start, lamnek_text if lamnek_needed else None, merged)
    finally:
        _cleanup(wav_path)


def _resolve_formats(
    fmt: str,
    lamnek: bool,
    srt_flag: bool,
    json_flag: bool,
    simple: bool,
) -> tuple[set[str], bool]:
    """Berechnet das Set an Ausgabeformaten und ob Lamnek-Text nötig ist."""
    if fmt == "alle":
        if simple:
            formats: set[str] = {"txt", "srt", "json"}
        elif lamnek:
            formats = {"lamnek", "srt", "json"}
        else:
            formats = {"txt", "srt", "json"}
    elif fmt == "lamnek":
        formats = {"lamnek"}
    elif fmt == "txt":
        formats = {"lamnek" if lamnek else "txt"}
    else:
        formats = {fmt}

    if srt_flag:
        formats.add("srt")
    if json_flag:
        formats.add("json")

    lamnek_needed = "lamnek" in formats
    return formats, lamnek_needed


def _resolve_output_paths(audio: str, output: str | None) -> tuple[str | None, str]:
    """Bestimmt expliziten Pfad und Basisnamen."""
    if output:
        return output, str(Path(output).with_suffix(""))
    return None, str(Path(audio).with_suffix(""))


def _print_summary(
    outputs: list[tuple[str, str]],
    total_start: float,
    lamnek_text: str | None,
    merged: list[dict[str, object]],
) -> None:
    """Druckt Zusammenfassung und Vorschau."""
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("📄 Ausgabe:")
    for path, fmt in outputs:
        size = Path(path).stat().st_size / 1024
        print(f"   {path} ({size:.1f} KB) [{fmt}]")
    print(f"\n⏱️ Gesamtzeit: {total_time:.1f}s")

    if lamnek_text:
        preview_text = lamnek_text[:400]
    else:
        preview_lines = [
            f"[{_format_preview_time(float(seg['start']))}] "  # type: ignore[arg-type]
            f"{seg.get('speaker', 'SPEAKER_00')}: "
            f"{seg['text']}"
            for seg in merged[:5]
        ]
        preview_text = "\n".join(preview_lines)
    if preview_text:
        print(f"\n📝 Transkript-Vorschau:\n{preview_text}...")


def _format_preview_time(seconds: float) -> str:
    """Formatiert einen Sekunden-Wert für die Vorschau."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _cleanup(wav_path: str) -> None:
    """Entfernt temporäre WAV-Datei."""
    with contextlib.suppress(OSError):
        Path(wav_path).unlink()


@click.command()
@click.version_option(version=__version__, prog_name="lamnek-simple")
@click.argument("audio", type=click.Path(exists=True, dir_okay=False))
@click.option("--sprache", "-s", default="auto", help="de, en, es, pt, auto")
@click.option(
    "--gerät", "-g", default="cuda", type=click.Choice(["cuda", "cpu"]), help="cuda oder cpu"
)
@click.option("--modell", "-m", default="large-v3-turbo", help="Whisper-Modell")
@click.option(
    "--format", "-f", "fmt", default="txt", type=click.Choice(sorted(FORMATS_SIMPLE)),
    help="Ausgabeformat"
)
@click.option("--output", "-o", default=None, help="Ausgabedatei")
@click.option("--vad", is_flag=True, help="Voice Activity Detection aktivieren")
def simple_cli(
    audio: str,
    sprache: str,
    gerät: str,
    modell: str,
    fmt: str,
    output: str | None,
    vad: bool,
) -> None:
    """Einfache ASR-Transkription ohne Diarization.

    Audio Datei wird transkribiert und als TXT/SRT/JSON gespeichert.
    """
    if sprache not in {"de", "en", "es", "pt", "auto"}:
        click.echo(f"❌ Unbekannte Sprache: {sprache}", err=True)
        click.echo("   Verfügbar: de, en, es, pt, auto", err=True)
        sys.exit(1)

    audio_path = Path(audio)
    print(f"📁 {audio_path.name} ({audio_path.stat().st_size / (1024 * 1024):.1f} MB)")

    language = None if sprache == "auto" else sprache
    segments, info = transcribe_audio(
        audio, model_name=modell, language=language, device=gerät, vad=vad
    )

    merged = [
        {"start": s.start, "end": s.end, "text": s.text, "speaker": "SPEAKER_00"}
        for s in segments
    ]
    lamnek_text = "\n".join(s.text for s in segments)

    explicit_path, base_name = _resolve_output_paths(audio, output)
    formats = {"txt", "srt", "json"} if fmt == "alle" else {fmt}

    outputs = generate_outputs(
        merged,
        lamnek_text,
        base_name,
        formats,
        info,
        project_info={},
        speaker_names=None,
        is_lamnek=False,
        explicit_path=explicit_path,
    )

    print("\n📄 Ausgabe:")
    for path, fmt_name in outputs:
        size = Path(path).stat().st_size / 1024
        print(f"   {path} ({size:.1f} KB) [{fmt_name}]")

    print("\n⏱️ Gesamtzeit: wird separat gemessen")


# Backward-compat: `lamnek` als reiner transcribe-Befehl, wenn direkt aufgerufen.
if __name__ == "__main__":
    cli()
