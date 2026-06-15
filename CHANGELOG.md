# Changelog

## 2.0.0 — 2026-06-15

- Initial release of `lamnek` package.
- Speaker diarization via pyannote.audio.
- ASR via faster-whisper large-v3-turbo.
- Lamnek/Krell scientific transcript formatting.
- TXT, SRT, JSON export.
- German user-facing CLI messages.
- CUDA setup via `TRANSCRIBE_CUDA_PATH` + ctypes preload + shell wrappers.
- Two entry points: `lamnek` and `lamnek-simple`.
- Type-safe code with mypy --strict and ruff linting.
