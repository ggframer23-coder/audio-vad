#!/usr/bin/env python3
"""
CPU-only VAD segmentation for long recordings.
- Produces per-day folders with timestamp CSV/JSON and cut audio segments.
- Default VAD: WebRTC (fast). Optional Silero VAD if a local model path is provided.
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac"}


@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def run_cmd(cmd: Sequence[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr.strip()}")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")


def parse_date_from_filename(name: str) -> Optional[dt.date]:
    patterns = [
        r"(\d{4})[-_](\d{2})[-_](\d{2})",
        r"(\d{4})(\d{2})(\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            y, m, d = match.groups()
            try:
                return dt.date(int(y), int(m), int(d))
            except ValueError:
                return None
    return None


def parse_time_from_filename(name: str) -> Optional[dt.time]:
    patterns = [
        r"(\d{2})[:_\-]?(\d{2})[:_\-]?(\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            h, m, s = match.groups()
            try:
                return dt.time(int(h), int(m), int(s))
            except ValueError:
                return None
    return None


def day_folder_for_file(path: Path) -> str:
    date = parse_date_from_filename(path.name)
    if date is None:
        ts = dt.datetime.fromtimestamp(path.stat().st_mtime)
        date = ts.date()
    return date.isoformat()


def convert_to_wav(input_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_path),
    ]
    run_cmd(cmd)


def read_wave(path: Path) -> Tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        if channels != 1:
            raise ValueError("Audio must be mono")
        if sample_width != 2:
            raise ValueError("Audio must be 16-bit PCM")
        if sample_rate != 16000:
            raise ValueError("Audio must be 16kHz")
        pcm = wf.readframes(wf.getnframes())
        return pcm, sample_rate


def frame_generator(pcm: bytes, sample_rate: int, frame_ms: int) -> List[Tuple[bytes, float, float]]:
    frame_size = int(sample_rate * frame_ms / 1000) * 2
    frames = []
    offset = 0
    timestamp = 0.0
    duration = frame_ms / 1000.0
    while offset + frame_size <= len(pcm):
        frame = pcm[offset : offset + frame_size]
        frames.append((frame, timestamp, duration))
        timestamp += duration
        offset += frame_size
    return frames


def segments_from_frames(
    speech_flags: Sequence[bool],
    frame_duration: float,
    total_duration: float,
) -> List[Segment]:
    segments: List[Segment] = []
    in_speech = False
    start = 0.0
    for idx, is_speech in enumerate(speech_flags):
        t = idx * frame_duration
        if is_speech and not in_speech:
            in_speech = True
            start = t
        elif not is_speech and in_speech:
            end = t
            segments.append(Segment(start, end))
            in_speech = False
    if in_speech:
        segments.append(Segment(start, total_duration))
    return segments


def merge_segments(segments: List[Segment], merge_gap: float) -> List[Segment]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start)
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.start - last.end <= merge_gap:
            merged[-1] = Segment(last.start, max(last.end, seg.end))
        else:
            merged.append(seg)
    return merged


def apply_padding(segments: List[Segment], padding: float, total_duration: float) -> List[Segment]:
    padded = []
    for seg in segments:
        start = max(0.0, seg.start - padding)
        end = min(total_duration, seg.end + padding)
        padded.append(Segment(start, end))
    return padded


def filter_short(segments: List[Segment], min_duration: float) -> List[Segment]:
    return [seg for seg in segments if seg.duration >= min_duration]


def webrtc_vad_segments(
    wav_path: Path,
    aggressiveness: int,
    frame_ms: int,
) -> Tuple[List[Segment], float]:
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError("Missing dependency: webrtcvad") from exc

    pcm, sample_rate = read_wave(wav_path)
    frames = frame_generator(pcm, sample_rate, frame_ms)
    vad = webrtcvad.Vad(aggressiveness)
    speech_flags = [vad.is_speech(frame, sample_rate) for frame, _, _ in frames]
    frame_duration = frame_ms / 1000.0
    total_duration = len(pcm) / (sample_rate * 2.0)
    return segments_from_frames(speech_flags, frame_duration, total_duration), total_duration


def silero_vad_segments(
    wav_path: Path,
    model_path: Path,
    threshold: float,
    min_silence_ms: int,
) -> Tuple[List[Segment], float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing dependency: torch") from exc

    model = torch.jit.load(str(model_path))
    model.eval()

    pcm, _ = read_wave(wav_path)
    audio = torch.frombuffer(pcm, dtype=torch.int16).float() / 32768.0
    audio = audio.view(1, -1)

    with torch.no_grad():
        probs = model(audio)[0].squeeze().cpu().numpy().tolist()

    frame_duration = 0.032
    total_duration = audio.numel() / 16000.0

    speech_flags = [p >= threshold for p in probs]
    segments = segments_from_frames(speech_flags, frame_duration, total_duration)

    min_silence = min_silence_ms / 1000.0
    segments = merge_segments(segments, min_silence)
    return segments, total_duration


def write_segments_csv(path: Path, segments: List[Segment]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "start_sec", "end_sec", "duration_sec"])
        for idx, seg in enumerate(segments, 1):
            writer.writerow([idx, f"{seg.start:.3f}", f"{seg.end:.3f}", f"{seg.duration:.3f}"])


def write_segments_json(path: Path, segments: List[Segment]) -> None:
    payload = [
        {
            "index": idx,
            "start_sec": round(seg.start, 3),
            "end_sec": round(seg.end, 3),
            "duration_sec": round(seg.duration, 3),
        }
        for idx, seg in enumerate(segments, 1)
    ]
    path.write_text(json.dumps(payload, indent=2))


def cut_segment(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
    output_format: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
    ]
    if output_format == "wav":
        cmd += ["-c:a", "pcm_s16le"]
    elif output_format == "flac":
        cmd += ["-c:a", "flac", "-compression_level", "5"]
    else:
        raise ValueError("Unsupported output format")
    cmd.append(str(output_path))
    run_cmd(cmd)


def resolve_inputs(paths: Sequence[str], recursive: bool) -> List[Path]:
    files = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            if recursive:
                candidates = path.rglob("*")
            else:
                candidates = path.glob("*")
            for cand in candidates:
                if cand.is_file() and cand.suffix.lower() in SUPPORTED_EXTS:
                    files.append(cand)
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
    return sorted(set(files))


def build_output_prefix(path: Path) -> str:
    return path.stem


def process_file(
    input_path: Path,
    output_root: Path,
    vad_mode: str,
    output_format: str,
    aggressiveness: int,
    frame_ms: int,
    min_speech_ms: int,
    merge_gap_ms: int,
    padding_ms: int,
    silero_model: Optional[Path],
    silero_threshold: float,
    silero_min_silence_ms: int,
    dry_run: bool,
) -> None:
    day_folder = day_folder_for_file(input_path)
    output_dir = output_root / day_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = build_output_prefix(input_path)
    json_path = output_dir / f"{prefix}_segments.json"
    csv_path = output_dir / f"{prefix}_segments.csv"

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "audio_16k.wav"
        convert_to_wav(input_path, wav_path)

        if vad_mode == "webrtc":
            segments, total_duration = webrtc_vad_segments(wav_path, aggressiveness, frame_ms)
        elif vad_mode == "silero":
            if silero_model is None:
                raise RuntimeError("--silero-model is required for silero mode")
            segments, total_duration = silero_vad_segments(
                wav_path,
                silero_model,
                silero_threshold,
                silero_min_silence_ms,
            )
        else:
            raise ValueError("Unknown VAD mode")

    segments = filter_short(segments, min_speech_ms / 1000.0)
    segments = merge_segments(segments, merge_gap_ms / 1000.0)
    segments = apply_padding(segments, padding_ms / 1000.0, total_duration)
    segments = merge_segments(segments, merge_gap_ms / 1000.0)

    write_segments_json(json_path, segments)
    write_segments_csv(csv_path, segments)

    if dry_run:
        return

    ext = "wav" if output_format == "wav" else "flac"
    for idx, seg in enumerate(segments, 1):
        seg_path = output_dir / f"{prefix}_seg_{idx:04d}.{ext}"
        cut_segment(input_path, seg_path, seg.start, seg.end, output_format)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract voice segments and cut audio files (CPU-only)."
    )
    parser.add_argument("inputs", nargs="+", help="Input files or directories")
    parser.add_argument("--output-dir", default="./output", help="Output root directory")
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories")
    parser.add_argument("--vad", default="webrtc", choices=["webrtc", "silero"], help="VAD mode")
    parser.add_argument("--output-format", default="flac", choices=["flac", "wav"], help="Segment format")

    parser.add_argument("--aggressiveness", type=int, default=3, help="WebRTC VAD aggressiveness (0-3)")
    parser.add_argument("--frame-ms", type=int, default=30, help="WebRTC frame size in ms")
    parser.add_argument("--min-speech-ms", type=int, default=600, help="Minimum segment duration")
    parser.add_argument("--merge-gap-ms", type=int, default=200, help="Merge gaps shorter than this")
    parser.add_argument("--padding-ms", type=int, default=100, help="Pad segments on both sides")

    parser.add_argument("--silero-model", type=str, help="Path to Silero VAD torchscript model")
    parser.add_argument("--silero-threshold", type=float, default=0.5, help="Silero speech probability threshold")
    parser.add_argument("--silero-min-silence-ms", type=int, default=300, help="Silero min silence for merging")

    parser.add_argument("--dry-run", action="store_true", help="Only write timestamp files")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    ensure_ffmpeg()

    inputs = resolve_inputs(args.inputs, args.recursive)
    if not inputs:
        print("No input audio files found.")
        return 1

    output_root = Path(args.output_dir).expanduser()
    for input_path in inputs:
        process_file(
            input_path=input_path,
            output_root=output_root,
            vad_mode=args.vad,
            output_format=args.output_format,
            aggressiveness=args.aggressiveness,
            frame_ms=args.frame_ms,
            min_speech_ms=args.min_speech_ms,
            merge_gap_ms=args.merge_gap_ms,
            padding_ms=args.padding_ms,
            silero_model=Path(args.silero_model) if args.silero_model else None,
            silero_threshold=args.silero_threshold,
            silero_min_silence_ms=args.silero_min_silence_ms,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
