"""Audio-Preprocessing: jede Datei → 16 kHz Mono WAV via FFmpeg."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: str, timeout: int = 60, check: bool = True) -> tuple[str, str]:
    """Führt ein Shell-Kommando aus und gibt (stdout, stderr) zurück.

    Args:
        cmd: Auszuführendes Kommando.
        timeout: Timeout in Sekunden.
        check: Bei Exit-Code != 0 eine RuntimeError werfen.

    Returns:
        Tupel aus stdout und stderr.

    Raises:
        RuntimeError: Wenn check=True und der Befehl fehlschlägt.
    """
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Befehl fehlgeschlagen (Exit {result.returncode}): {cmd}\n"
            f"stderr: {result.stderr.strip()[:500]}"
        )
    return result.stdout, result.stderr


def preprocess_audio(input_path: str) -> str:
    """Konvertiert eine beliebige Audiodatei zu 16 kHz mono WAV.

    Args:
        input_path: Pfad zur Eingabedatei.

    Returns:
        Pfad zur erzeugten WAV-Datei.

    Raises:
        SystemExit: Wenn ffmpeg fehlt oder das Kommando fehlschlägt.
    """
    print(f"🔧 Preprocessing: {Path(input_path).name}")

    import os as _os
    temp_dir = Path(_os.environ.get("TRANSCRIBE_TEMP_DIR", "/tmp"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    base = Path(input_path).stem
    output_path = temp_dir / f"lamnek_{base}_{os.getpid()}_16k.wav"

    cmd = (
        f'ffmpeg -y -i "{input_path}" -ar 16000 -ac 1 -sample_fmt s16 '
        f'"{output_path}"'
    )
    try:
        run_cmd(cmd, timeout=300)
    except FileNotFoundError:
        print("❌ ffmpeg nicht gefunden. Bitte installiere ffmpeg.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ ffmpeg Fehler:\n{e}")
        sys.exit(1)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   → {output_path} ({size_mb:.1f} MB, 16 kHz mono)")
    return str(output_path)
