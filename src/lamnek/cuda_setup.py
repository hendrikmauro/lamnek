"""CUDA-Setup: LD_LIBRARY_PATH + ctypes-Preload für WSL/Linux."""

from __future__ import annotations

import contextlib
import ctypes
import os
from pathlib import Path


def setup_cuda() -> list[str]:
    """Setzt LD_LIBRARY_PATH und preloaded kritische CUDA-Bibliotheken.

    Returns:
        Liste der erkannten CUDA-Bibliotheksverzeichnisse.
    """
    cuda_paths: list[str] = []
    env_path = os.environ.get("TRANSCRIBE_CUDA_PATH")
    if env_path and Path(env_path).is_dir():
        cuda_paths.append(env_path)

    common_paths = [
        "/usr/local/lib/ollama/cuda_v12",
        "/usr/lib/wsl/lib",
        "/usr/local/cuda/lib64",
    ]
    for p in common_paths:
        if Path(p).is_dir() and p not in cuda_paths:
            cuda_paths.append(p)

    if cuda_paths:
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = [p for p in cuda_paths if p not in ld]
        if new_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths + ([ld] if ld else []))

    for p in cuda_paths:
        with contextlib.suppress(AttributeError, OSError):
            os.add_dll_directory(p)  # type: ignore[attr-defined]

    for lib_name in ("libcublas.so.12", "libcudart.so.12", "libcublasLt.so.12"):
        for p in cuda_paths:
            lib_path = Path(p) / lib_name
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    pass

    return cuda_paths
