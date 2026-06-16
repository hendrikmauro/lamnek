"""CUDA-Setup: LD_LIBRARY_PATH + ctypes-Preload für WSL/Linux.

Löst das Problem, dass LD_LIBRARY_PATH zur Laufzeit gesetzt wird
(zu spät für ld.so), indem CUDA-Bibliotheken via ctypes mit
RTLD_GLOBAL preloaded werden — bevor faster-whisper/CTranslate2
geladen werden.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
from pathlib import Path

# Bibliotheken, die CTranslate2/faster-whisper braucht
_CUDA_LIBS = (
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudart.so.12",
    "libcudnn.so.8",
    "libcurand.so.10",
    "libcufft.so.11",
)

# Suchpfade für CUDA-Bibliotheken
_SEARCH_PATHS = [
    "/usr/local/lib/ollama/cuda_v12",
    "/usr/lib/wsl/lib",
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/lib",
    "/opt/cuda/lib64",
]


def _find_cuda_paths() -> list[str]:
    """Findet alle Verzeichnisse mit CUDA-Bibliotheken."""
    cuda_paths: list[str] = []

    # 1. TRANSCRIBE_CUDA_PATH (manuell überschreibbar)
    env_path = os.environ.get("TRANSCRIBE_CUDA_PATH")
    if env_path and Path(env_path).is_dir():
        cuda_paths.append(env_path)

    # 2. Bekannte Pfade
    for p in _SEARCH_PATHS:
        if Path(p).is_dir() and p not in cuda_paths:
            cuda_paths.append(p)

    # 3. Aus bestehender LD_LIBRARY_PATH
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for p in ld.split(":"):
        if p and Path(p).is_dir() and p not in cuda_paths:
            cuda_paths.append(p)

    # 4. aus ctypes.util.find_library
    for lib_name in _CUDA_LIBS:
        found = ctypes.util.find_library(Path(lib_name).stem)
        if found:
            parent = str(Path(found).parent)
            if parent not in cuda_paths:
                cuda_paths.append(parent)

    return cuda_paths


def _preload_libs(cuda_paths: list[str]) -> list[str]:
    """Preloaded CUDA-Bibliotheken mit RTLD_GLOBAL.

    CTranslate2 nutzt dlopen() zum Laden von CUDA-Libs.
    Durch RTLD_GLOBAL werden diese im globalen Symbol-Namespace
    verfügbar, sodass nachfolgende dlopen()-Aufrufe sie finden.
    """
    loaded: list[str] = []
    for lib_name in _CUDA_LIBS:
        for p in cuda_paths:
            lib_path = Path(p) / lib_name
            if lib_path.exists():
                try:
                    # RTLD_GLOBAL macht die Symbole für nachfolgende
                    # dlopen-Aufrufe in C++-Bibliotheken sichtbar
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                    loaded.append(lib_name)
                    break
                except OSError:
                    continue
    return loaded


def _set_env(cuda_paths: list[str]) -> None:
    """Setzt LD_LIBRARY_PATH und add_dll_directory für Kompatibilität."""
    if cuda_paths:
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = [p for p in cuda_paths if p not in ld]
        if new_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths + ([ld] if ld else []))

    for p in cuda_paths:
        with contextlib.suppress(AttributeError, OSError):
            os.add_dll_directory(p)  # type: ignore[attr-defined]


def setup_cuda() -> list[str]:
    """Setzt CUDA-Bibliotheken auf: findet Pfade, preloaded via ctypes,
    setzt LD_LIBRARY_PATH für Subprozesse.

    Returns:
        Liste der erkannten CUDA-Bibliotheksverzeichnisse.
    """
    cuda_paths = _find_cuda_paths()
    if not cuda_paths:
        return []

    _set_env(cuda_paths)
    _preload_libs(cuda_paths)

    return cuda_paths


# Beim Import ausführen — das ist der frühestmögliche Punkt
setup_cuda()
