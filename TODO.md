# TODO / Known Issues

## pyannote returns more speakers than --anzahl-sprecher specifies

**Priority:** Low (graceful workaround exists)

When running `lamnek transcribe-pro` with `--anzahl-sprecher 2`, pyannote may
still return 3+ speaker labels. The excess speakers are currently marked as
`Unbekannt` in the merge step.

### Reproduction

```bash
lamnek transcribe-pro interview.mp4 --sprache de --anzahl-sprecher 2 --sprecher "Interviewer,Befragte"
```

Output:
```
👥 Diarisierung (Sprecher-Erkennung)...
   ✅ 743 Segmente, 86.3s
   Sprecher: SPEAKER_00, SPEAKER_01, SPEAKER_02
🔗 Kombiniere Sprecher + Text...
   Sprecher benannt: Interviewer, Befragte
   1 weitere Sprecher → Unbekannt
```

### Investigation needed

- Check if `pyannote/speaker-diarization-3.1` respects `num_speakers` strictly
  or just as a hint
- Consider using `min_speakers` / `max_speakers` params instead
- The merge code (`merge.py`) already handles this gracefully by sorting
  speakers by duration and assigning names to the top N

### Workaround

Current behavior is acceptable — excess speakers get `Unbekannt` label.
No data loss, transcript remains usable.