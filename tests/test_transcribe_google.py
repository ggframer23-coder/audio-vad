import sys
from pathlib import Path

import scripts.transcribe_google as tg


def test_iter_audio_files_single_file(tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    assert list(tg.iter_audio_files(audio, recursive=False)) == [audio]


def test_iter_audio_files_recursive(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    audio = tmp_path / "nested" / "sample.flac"
    audio.write_bytes(b"fLaC\x00\x00\x00\x00")
    assert list(tg.iter_audio_files(tmp_path, recursive=True)) == [audio]


def test_write_transcript(tmp_path: Path):
    out_path = tmp_path / "out.txt"
    tg.write_transcript(out_path, "hello world")
    assert out_path.read_text() == "hello world\n"


def test_transcribe_file_builds_response(tmp_path: Path):
    class FakeAlternative:
        def __init__(self, transcript: str) -> None:
            self.transcript = transcript

    class FakeResult:
        def __init__(self, transcript: str) -> None:
            self.alternatives = [FakeAlternative(transcript)]

    class FakeResponse:
        def __init__(self, transcripts):
            self.results = [FakeResult(t) for t in transcripts]

    class FakeClient:
        def recognize(self, config, audio):
            assert config.language_code == "en-US"
            assert audio.content
            return FakeResponse(["hello", "world"])

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    text = tg.transcribe_file(FakeClient(), audio_path, language="en-US")
    assert text == "hello world"


def test_build_parser_defaults():
    parser = tg.build_parser()
    args = parser.parse_args(["input.wav"])
    assert args.language == "en-US"
    assert args.output_dir == "./transcripts"


def test_main_requires_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "input.wav"])
    try:
        tg.main()
    except SystemExit as exc:
        assert "GOOGLE_APPLICATION_CREDENTIALS" in str(exc)
    else:
        raise AssertionError("Expected SystemExit")


def test_main_writes_transcripts(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    audio_path = input_dir / "sample.wav"
    audio_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    output_dir = tmp_path / "out"

    class FakeClient:
        pass

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "creds.json"))
    monkeypatch.setattr(tg.speech, "SpeechClient", lambda: FakeClient())
    monkeypatch.setattr(tg, "transcribe_file", lambda client, path, language: "hello world")
    monkeypatch.setattr(sys, "argv", ["prog", str(input_dir), "--output-dir", str(output_dir)])

    assert tg.main() == 0
    out_path = output_dir / "sample.txt"
    assert out_path.read_text() == "hello world\n"
