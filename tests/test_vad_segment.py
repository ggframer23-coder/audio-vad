import datetime as dt
import os
import sys
import types
from pathlib import Path

import scripts.vad_segment as vad


def test_parse_date_from_filename():
    assert vad.parse_date_from_filename("recording_2026-01-04_120000.wav") == dt.date(2026, 1, 4)
    assert vad.parse_date_from_filename("20260104_recording.wav") == dt.date(2026, 1, 4)
    assert vad.parse_date_from_filename("no_date_here.wav") is None


def test_parse_time_from_filename():
    assert vad.parse_time_from_filename("recording_12-34-56.wav") == dt.time(12, 34, 56)
    assert vad.parse_time_from_filename("recording_123456.wav") == dt.time(12, 34, 56)
    assert vad.parse_time_from_filename("no_time_here.wav") is None


def test_day_folder_for_file_uses_filename_date(tmp_path: Path):
    path = tmp_path / "recording_2026-01-04_120000.wav"
    path.write_text("stub")
    assert vad.day_folder_for_file(path) == "2026-01-04"


def test_day_folder_for_file_falls_back_to_mtime(tmp_path: Path):
    path = tmp_path / "recording.wav"
    path.write_text("stub")
    target = dt.datetime(2026, 1, 4, 12, 0, 0)
    ts = target.timestamp()
    os.utime(path, (ts, ts))
    assert vad.day_folder_for_file(path) == "2026-01-04"


def test_segments_from_frames_and_merge_padding():
    speech_flags = [False, True, True, False, False, True, True]
    segments = vad.segments_from_frames(speech_flags, frame_duration=0.03, total_duration=0.21)
    assert segments == [vad.Segment(0.03, 0.09), vad.Segment(0.15, 0.21)]

    merged = vad.merge_segments(segments, merge_gap=0.07)
    assert merged == [vad.Segment(0.03, 0.21)]

    padded = vad.apply_padding(merged, padding=0.05, total_duration=0.21)
    assert padded == [vad.Segment(0.0, 0.21)]


def test_filter_short():
    segments = [vad.Segment(0.0, 0.2), vad.Segment(0.5, 1.2)]
    assert vad.filter_short(segments, min_duration=0.3) == [vad.Segment(0.5, 1.2)]


def test_resolve_output_format():
    assert vad.resolve_output_format(Path("audio.mp3"), "auto") == "mp3"
    assert vad.resolve_output_format(Path("audio.wav"), "auto") == "wav"
    assert vad.resolve_output_format(Path("audio.unknown"), "auto") == "flac"
    assert vad.resolve_output_format(Path("audio.mp3"), "flac") == "flac"


def test_run_cmd_success(monkeypatch):
    def fake_run(cmd, stdout, stderr, text):
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(vad.subprocess, "run", fake_run)
    vad.run_cmd(["echo", "ok"])


def test_run_cmd_failure(monkeypatch):
    def fake_run(cmd, stdout, stderr, text):
        return types.SimpleNamespace(returncode=1, stderr="boom")

    monkeypatch.setattr(vad.subprocess, "run", fake_run)
    try:
        vad.run_cmd(["false"])
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_ensure_ffmpeg(monkeypatch):
    monkeypatch.setattr(vad.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    vad.ensure_ffmpeg()


def test_convert_to_wav_builds_command(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)

    monkeypatch.setattr(vad, "run_cmd", fake_run)
    input_path = tmp_path / "in.mp3"
    wav_path = tmp_path / "out.wav"
    vad.convert_to_wav(input_path, wav_path)
    assert calls and calls[0][0] == "ffmpeg"
    assert str(input_path) in calls[0]
    assert str(wav_path) in calls[0]


def test_read_wave_and_frame_generator(tmp_path: Path):
    wav_path = tmp_path / "sample.wav"
    with vad.wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)

    pcm, sample_rate = vad.read_wave(wav_path)
    assert sample_rate == 16000
    frames = vad.frame_generator(pcm, sample_rate, frame_ms=20)
    assert frames


def test_resolve_inputs(tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    other = tmp_path / "notes.txt"
    other.write_text("nope")
    nested = tmp_path / "nested"
    nested.mkdir()
    nested_audio = nested / "clip.flac"
    nested_audio.write_bytes(b"fLaC\x00\x00\x00\x00")

    files = vad.resolve_inputs([str(tmp_path)], recursive=False)
    assert files == [audio]
    files = vad.resolve_inputs([str(tmp_path)], recursive=True)
    assert sorted(files) == sorted([audio, nested_audio])


def test_write_segments_csv_json(tmp_path: Path):
    segments = [vad.Segment(0.0, 1.0), vad.Segment(2.0, 3.5)]
    csv_path = tmp_path / "segments.csv"
    json_path = tmp_path / "segments.json"
    vad.write_segments_csv(csv_path, segments)
    vad.write_segments_json(json_path, segments)
    assert csv_path.read_text().splitlines()[0] == "index,start_sec,end_sec,duration_sec"
    assert "\"duration_sec\"" in json_path.read_text()


def test_cut_segment_builds_command(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)

    monkeypatch.setattr(vad, "run_cmd", fake_run)
    vad.cut_segment(tmp_path / "in.wav", tmp_path / "out.flac", 0.0, 1.0, "flac")
    assert calls and "-c:a" in calls[0]


def test_cut_segment_copy_format(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)

    monkeypatch.setattr(vad, "run_cmd", fake_run)
    vad.cut_segment(tmp_path / "in.mp3", tmp_path / "out.mp3", 0.0, 1.0, "mp3")
    assert calls and "copy" in calls[0]


def test_webrtc_vad_segments_with_stub(monkeypatch, tmp_path: Path):
    wav_path = tmp_path / "sample.wav"
    with vad.wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 3200)

    class FakeVad:
        def __init__(self, aggressiveness):
            self.calls = 0

        def is_speech(self, frame, sample_rate):
            self.calls += 1
            return self.calls % 2 == 0

    fake_module = types.SimpleNamespace(Vad=FakeVad)
    monkeypatch.setitem(sys.modules, "webrtcvad", fake_module)

    segments, total = vad.webrtc_vad_segments(wav_path, aggressiveness=2, frame_ms=30)
    assert total > 0
    assert segments


def test_silero_vad_segments_with_stub(monkeypatch, tmp_path: Path):
    wav_path = tmp_path / "sample.wav"
    with vad.wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)

    class FakeProbTensor:
        def __init__(self, probs):
            self._probs = probs

        def squeeze(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self

        def tolist(self):
            return self._probs

    class FakeAudioTensor:
        def __init__(self, data):
            self._data = data

        def float(self):
            return self

        def __truediv__(self, other):
            return self

        def view(self, *args):
            return self

        def numel(self):
            return len(self._data)

    class FakeModel:
        def eval(self):
            return None

        def __call__(self, audio):
            return [FakeProbTensor([0.1, 0.9, 0.2, 0.8])]

    class FakeJit:
        def load(self, path):
            return FakeModel()

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return None

    fake_torch = types.SimpleNamespace(
        jit=FakeJit(),
        int16=object(),
        frombuffer=lambda data, dtype: FakeAudioTensor(data),
        no_grad=lambda: FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    segments, total = vad.silero_vad_segments(
        wav_path,
        model_path=tmp_path / "fake.jit",
        threshold=0.5,
        min_silence_ms=100,
    )
    assert total > 0
    assert segments


def test_process_file_dry_run(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "recording_2026-01-04_120000.wav"
    input_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    output_root = tmp_path / "out"

    def fake_convert(in_path, wav_path):
        with vad.wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)

    monkeypatch.setattr(vad, "convert_to_wav", fake_convert)
    monkeypatch.setattr(vad, "webrtc_vad_segments", lambda *args, **kwargs: ([vad.Segment(0.0, 1.0)], 1.0))

    vad.process_file(
        input_path=input_path,
        output_root=output_root,
        vad_mode="webrtc",
        output_format="flac",
        aggressiveness=3,
        frame_ms=30,
        min_speech_ms=100,
        merge_gap_ms=200,
        padding_ms=0,
        silero_model=None,
        silero_threshold=0.5,
        silero_min_silence_ms=300,
        dry_run=True,
    )

    day_dir = output_root / "2026-01-04"
    assert (day_dir / "recording_2026-01-04_120000_segments.json").exists()
    assert (day_dir / "recording_2026-01-04_120000_segments.csv").exists()


def test_process_file_calls_cut(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "recording_2026-01-04_120000.wav"
    input_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    output_root = tmp_path / "out"
    calls = []

    def fake_convert(in_path, wav_path):
        with vad.wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)

    def fake_cut(*args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(vad, "convert_to_wav", fake_convert)
    monkeypatch.setattr(vad, "webrtc_vad_segments", lambda *args, **kwargs: ([vad.Segment(0.0, 1.0)], 1.0))
    monkeypatch.setattr(vad, "cut_segment", fake_cut)

    vad.process_file(
        input_path=input_path,
        output_root=output_root,
        vad_mode="webrtc",
        output_format="flac",
        aggressiveness=3,
        frame_ms=30,
        min_speech_ms=100,
        merge_gap_ms=200,
        padding_ms=0,
        silero_model=None,
        silero_threshold=0.5,
        silero_min_silence_ms=300,
        dry_run=False,
    )

    assert calls
