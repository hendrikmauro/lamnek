"""Lamnek — wissenschaftliche Interview-Transkription.

Module:
    audio        – FFmpeg-Audio-Preprocessing
    asr          – faster-whisper ASR-Wrapper
    diarization  – pyannote Speaker-Diarization
    merge        – Diarization + ASR kombinieren
    lamnek       – Lamnek/Krell-Formatierung
    formats      – TXT/SRT/JSON Export
    cli          – Click-basierte Kommandozeile
"""

__version__ = "2.0.0"
__author__ = "hendrikmauro"
__license__ = "MIT"

from lamnek.audio import preprocess_audio, run_cmd
from lamnek.lamnek import basic_lamnek, generate_header, pause_marker
from lamnek.merge import merge_results

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "basic_lamnek",
    "generate_header",
    "merge_results",
    "pause_marker",
    "preprocess_audio",
    "run_cmd",
]
