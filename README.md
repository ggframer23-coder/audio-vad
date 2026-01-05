# Voice Ext - VAD Segmentation

CPU-only voice activity detection (VAD) to extract speech segments from long recordings.

## Features

- WebRTC VAD default (fast, lightweight)
- Optional Silero VAD if you provide a local model file
- Outputs per-day folders based on filename date
- Writes timestamps (CSV + JSON) and cuts audio segments

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires `ffmpeg` in PATH.

## Usage

```bash
python scripts/vad_segment.py /path/to/audio/dir --recursive --output-dir ./output
```

Output layout:

```
output/
  YYYY-MM-DD/
    recording_2026-01-04_143022_segments.json
    recording_2026-01-04_143022_segments.csv
    recording_2026-01-04_143022_seg_0001.flac
    recording_2026-01-04_143022_seg_0002.flac
```

## Tuning (car noise)

Start with these defaults (already set):

- `--aggressiveness 3`
- `--min-speech-ms 600`
- `--merge-gap-ms 200`
- `--padding-ms 100`

If noise still leaks in, increase `--min-speech-ms` to 800 and `--merge-gap-ms` to 300.
If quiet speech is missing, reduce `--aggressiveness` to 2 and `--min-speech-ms` to 400.

## Silero VAD (optional)

If you have a local Silero VAD torchscript model:

```bash
python scripts/vad_segment.py /path/to/audio/dir \
  --recursive \
  --vad silero \
  --silero-model /path/to/silero_vad.jit \
  --output-dir ./output
```

This requires `torch` installed in your environment.

## Notes

- Supported input formats: WAV, MP3, M4A, FLAC, OGG, WMA, AAC
- Output format defaults to `flac` for smaller size. Use `--output-format wav` for LINEAR16.

## Google Speech-to-Text

Set credentials and run:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
python scripts/transcribe_google.py ./output/2026-01-04 --recursive --output-dir ./transcripts
```

Notes:
- Expected input: 16kHz mono LINEAR16 WAV (use `--output-format wav` in the VAD script).
- Requires Google Cloud Speech-to-Text API enabled for your project.
