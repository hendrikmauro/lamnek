#!/bin/bash
# Shell-Wrapper für lamnek (Haupt-Tool) in WSL.
# Setzt LD_LIBRARY_PATH für CUDA-Bibliotheken vor dem Python-Start.

export LD_LIBRARY_PATH="${TRANSCRIBE_CUDA_PATH:-/usr/local/lib/ollama/cuda_v12}:/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
exec lamnek "$@"
