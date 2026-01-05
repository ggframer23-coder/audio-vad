#!/usr/bin/env python3
"""Batch transcribe segmented audio with Google Speech-to-Text."""

import argparse
import os
from pathlib import Path
from typing import Iterable

from google.cloud import speech

SUPPORTED_EXTS = {".wav", ".flac"}


def iter_audio_files(root: Path, recursive: bool) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() in SUPPORTED_EXTS:
        yield root
        return
    pattern = "**/*" if recursive else "*"
    for path in root.glob(pattern):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def transcribe_file(client: speech.SpeechClient, path: Path, language: str) -> str:
    audio = speech.RecognitionAudio(content=path.read_bytes())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language,
        enable_automatic_punctuation=True,
    )
    response = client.recognize(config=config, audio=audio)
    transcripts = [r.alternatives[0].transcript for r in response.results]
    return " ".join(transcripts).strip()


def write_transcript(path: Path, text: str) -> None:
    path.write_text(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio with Google STT")
    parser.add_argument("input", help="Audio file or directory")
    parser.add_argument("--recursive", action="store_true", help="Recurse directories")
    parser.add_argument("--language", default="en-US", help="Language code")
    parser.add_argument("--output-dir", default="./transcripts", help="Output directory")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise SystemExit("Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON file")

    client = speech.SpeechClient()
    input_path = Path(args.input).expanduser()
    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    for audio_path in iter_audio_files(input_path, args.recursive):
        text = transcribe_file(client, audio_path, args.language)
        rel = audio_path.name
        out_path = output_root / f"{Path(rel).stem}.txt"
        write_transcript(out_path, text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
