# Repository Guidelines

## Project Structure & Module Organization
- `scripts/` contains the runnable tools:
  - `scripts/vad_segment.py` for VAD-based segmentation and audio slicing.
  - `scripts/transcribe_google.py` for Google Speech-to-Text batch transcription.
- `requirements.txt` pins Python dependencies.
- `README.md` documents setup and usage examples.
- Generated outputs typically live under `./output/` and `./transcripts/`.

## Build, Test, and Development Commands
- `python -m venv .venv` and `source .venv/bin/activate` set up the local environment.
- `pip install -r requirements.txt` installs dependencies.
- `python scripts/vad_segment.py /path/to/audio --recursive --output-dir ./output` runs VAD segmentation.
- `python scripts/transcribe_google.py ./output/2026-01-04 --recursive --output-dir ./transcripts` transcribes segments.

There is no build system or test runner configured; scripts are executed directly.

## Coding Style & Naming Conventions
- Use 4-space indentation and standard Python conventions (PEP 8 style).
- Keep functions small and single-purpose; prefer `Path` from `pathlib` for filesystem work.
- CLI flags use `--kebab-case` (see `vad_segment.py` and `transcribe_google.py`).
- Add new dependencies to `requirements.txt` with pinned versions.

## Testing Guidelines
- Tests are run with `pytest` and live under `tests/`.
- Name test files `test_*.py` and keep them focused on pure functions when possible.
- Run `make test` for quick checks and `make coverage` for coverage reporting.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative summaries (example from history: “Add VAD segmentation and Google STT scripts”).
- PRs should describe the change, list key commands run, and include sample output paths (for example, `output/2026-01-04/...`).

## Security & Configuration Tips
- `ffmpeg` must be available on PATH for `vad_segment.py`.
- Set `GOOGLE_APPLICATION_CREDENTIALS` when using Google STT.
- Avoid committing audio data or transcripts; use local paths only.

## Agent-Specific Instructions
- When asked, save the current chat to `chat_<date>.md` in the repository root (use `YYYY-MM-DD` for `<date>`).
