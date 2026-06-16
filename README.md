# Lamnek

**Lokale, datenschutzfreundliche Transkription für wissenschaftliche Interviews**

Lamnek kombiniert **Speaker Diarization** (pyannote) mit **faster-whisper ASR**
(large-v3-turbo) und formatiert Ergebnisse nach dem wissenschaftlichen
**Lamnek/Krell-Standard**. Keine Audio-Daten verlassen den Rechner.

## Features

- 🔊 **Sprecher-Diarisierung** mit pyannote.audio
- 🎙️ **ASR** via faster-whisper large-v3-turbo
- 📜 **Lamnek/Krell-Format** mit Zeilennummern, Pausenmarkern und Header
- 🔒 **Anonymisierung** der Sprecher
- 📤 Export als **TXT**, **SRT**, **JSON**
- ⚡ **CUDA-Fallback** auf CPU
- 🇩🇪 **Deutsche Nutzerführung**

## Plattform

**Linux only.** Getestet auf Ubuntu/WSL mit NVIDIA CUDA.
Windows und macOS werden nicht unterstützt (CUDA-Pfade, `/tmp`-Verzeichnis,
Bash-Wrapper sind Linux-spezifisch).

## Installation

```bash
git clone https://github.com/hendrikmauro/lamnek.git
cd lamnek
pip install -r requirements.txt
pip install -e ".[dev]"
```

Voraussetzungen:
- **Linux** (getestet auf Ubuntu/WSL)
- Python ≥ 3.10
- ffmpeg im PATH
- Für Diarisierung: `HF_TOKEN` von HuggingFace
  1. Token erstellen: <https://huggingface.co/settings/tokens> (Type: Read)
  2. pyannote Modell-Lizenzen akzeptieren:
     - <https://huggingface.co/pyannote/speaker-diarization-3.1>
     - <https://huggingface.co/pyannote/segmentation-3.0>
  3. Token setzen: `export HF_TOKEN=dein_token`

## Umgebungsvariablen

| Variable | Bedeutung |
|---|---|
| `HF_TOKEN` | **Erforderlich** für pyannote Model-Download |
| `TRANSCRIBE_CUDA_PATH` | Optionaler CUDA-Bibliothekspfad |
| `TRANSCRIBE_TEMP_DIR` | Temporäres Verzeichnis (Default: `/tmp`) |

## CLI-Entry-Points

```bash
# Vollständige wissenschaftliche Transkription
lamnek transcribe interview.mp3 --lamnek --sprache de

# Alias mit Lamnek als Default
lamnek transcribe-pro interview.mp3 --sprecher "Interviewer,Befragte"

# Einfache ASR ohne Diarization
lamnek-simple podcast.m4a --sprache de --format srt
```

## Beispiele

```bash
export HF_TOKEN=hf_...

# Lamnek-Transkript mit Header und SRT/JSON
lamnek transcribe-pro interview.mp3 \
  --sprache de \
  --anzahl-sprecher 2 \
  --sprecher "Interviewer,Befragte" \
  --projekt "Studie 2026" \
  --pseudonym "P001" \
  --interviewer "Dr. Schmidt" \
  --ort "Berlin"

# Sprecher-Vorschau ohne Dateien zu schreiben
lamnek transcribe-pro interview.mp3 --vorschau

# Nur ASR, alle Formate
lamnek-simple meeting.ogg --format alle
```

## Output-Dateien

| Suffix | Format | Beschreibung |
|---|---|---|
| `*_lamnek.txt` | Lamnek | Wissenschaftliches Transkript |
| `*_transkript.txt` | TXT | Zeitgestempeltes Transkript |
| `*_transkript.srt` | SRT | Untertitel mit Sprecher-Präfix |
| `*_transkript.json` | JSON | Rohdaten + Lamnek-Text |

## Lizenz

MIT — siehe [LICENSE](LICENSE).
Copyright (c) 2026 hendrikmauro.

---

# Lamnek (English)

**Local, privacy-first transcription for scientific interviews.**

Lamnek combines **speaker diarization** (pyannote) with **faster-whisper ASR**
(large-v3-turbo) and formats output according to the scientific
**Lamnek/Krell standard**. No audio data leaves your machine.

## Features

- 🔊 Speaker diarization via pyannote.audio
- 🎙️ ASR via faster-whisper large-v3-turbo
- 📜 Lamnek/Krell formatting with line numbers, pause markers, and header
- 🔒 Speaker anonymization
- 📤 Export to TXT, SRT, JSON
- ⚡ CUDA with CPU fallback
- 🇩🇪 German user-facing messages

## Platform

**Linux only.** Tested on Ubuntu/WSL with NVIDIA CUDA.
Windows and macOS are not supported (CUDA paths, `/tmp` directory,
Bash wrappers are Linux-specific).

## Installation

```bash
git clone https://github.com/hendrikmauro/lamnek.git
cd lamnek
pip install -r requirements.txt
pip install -e ".[dev]"
```

Requirements:
- **Linux** (tested on Ubuntu/WSL)
- Python ≥ 3.10
- ffmpeg on PATH
- For diarization: `HF_TOKEN` from HuggingFace
  1. Create token: <https://huggingface.co/settings/tokens> (Type: Read)
  2. Accept pyannote model licenses:
     - <https://huggingface.co/pyannote/speaker-diarization-3.1>
     - <https://huggingface.co/pyannote/segmentation-3.0>
  3. Set token: `export HF_TOKEN=your_token`

## Environment Variables

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | **Required** for pyannote model download |
| `TRANSCRIBE_CUDA_PATH` | Optional CUDA library path |
| `TRANSCRIBE_TEMP_DIR` | Optional temp directory (default: `/tmp`) |

## CLI Entry Points

```bash
# Full scientific transcription
lamnek transcribe interview.mp3 --lamnek --sprache de

# Alias with Lamnek as default
lamnek transcribe-pro interview.mp3 --sprecher "Interviewer,Interviewee"

# Simple ASR without diarization
lamnek-simple podcast.m4a --sprache de --format srt
```

> **Note:** CLI flags are in German (e.g. `--sprache` for language,
> `--sprecher` for speakers, `--ort` for location, `--vorschau` for preview).
> Run `lamnek --help` for the full list.

## Examples

```bash
export HF_TOKEN=hf_...

# Lamnek transcript with header and SRT/JSON
lamnek transcribe-pro interview.mp3 \
  --sprache de \
  --anzahl-sprecher 2 \
  --sprecher "Interviewer,Interviewee" \
  --projekt "Study 2026" \
  --pseudonym "P001" \
  --interviewer "Dr. Schmidt" \
  --ort "Berlin"

# Speaker preview without writing files
lamnek transcribe-pro interview.mp3 --vorschau

# ASR only, all formats
lamnek-simple meeting.ogg --format alle
```

## Output Files

| Suffix | Format | Description |
|---|---|---|
| `*_lamnek.txt` | Lamnek | Scientific transcript |
| `*_transkript.txt` | TXT | Timestamped transcript |
| `*_transkript.srt` | SRT | Subtitles with speaker prefix |
| `*_transkript.json` | JSON | Raw data + Lamnek text |

## License

MIT — see [LICENSE](LICENSE).
Copyright (c) 2026 hendrikmauro.
