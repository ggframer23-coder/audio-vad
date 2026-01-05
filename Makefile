SHELL := /bin/bash

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python3
UV ?= uv

INPUT ?= ./audio
OUTPUT ?= ./output
TRANSCRIPTS ?= ./transcripts
LANGUAGE ?= en-US
RECURSIVE ?= --recursive

VAD ?= webrtc
OUTPUT_FORMAT ?= flac

AGGRESSIVENESS ?= 3
FRAME_MS ?= 30
MIN_SPEECH_MS ?= 600
MERGE_GAP_MS ?= 200
PADDING_MS ?= 100

SILERO_MODEL ?=
SILERO_THRESHOLD ?= 0.5
SILERO_MIN_SILENCE_MS ?= 300

.PHONY: help venv install install-dev check-ffmpeg vad transcribe test coverage all clean

help:
	@echo "Repository commands:"
	@echo "  make venv            Create virtual environment"
	@echo "  make install         Install Python dependencies"
	@echo "  make install-dev     Install dev dependencies (pytest, coverage)"
	@echo "  make vad             Run VAD segmentation"
	@echo "  make transcribe      Run Google STT on segments"
	@echo "  make test            Run pytest test suite"
	@echo "  make coverage        Run tests with coverage report"
	@echo "  make all             Run vad then transcribe"
	@echo "  make clean           Remove local outputs"
	@echo ""
	@echo "Common overrides:"
	@echo "  INPUT=./path/to/audio OUTPUT=./output TRANSCRIPTS=./transcripts"
	@echo "  VAD=webrtc|silero OUTPUT_FORMAT=flac|wav"
	@echo "  SILERO_MODEL=/path/to/silero_vad.jit"

venv:
	$(UV) venv $(VENV)

install: venv
	$(UV) pip install -r requirements.txt

install-dev: install
	$(UV) pip install -r requirements-dev.txt

check-ffmpeg:
	@command -v ffmpeg >/dev/null || (echo "ffmpeg not found in PATH" && exit 1)

vad: check-ffmpeg
	$(PYTHON) scripts/vad_segment.py $(INPUT) $(RECURSIVE) \
		--output-dir $(OUTPUT) \
		--vad $(VAD) \
		--output-format $(OUTPUT_FORMAT) \
		--aggressiveness $(AGGRESSIVENESS) \
		--frame-ms $(FRAME_MS) \
		--min-speech-ms $(MIN_SPEECH_MS) \
		--merge-gap-ms $(MERGE_GAP_MS) \
		--padding-ms $(PADDING_MS) \
		$(if $(SILERO_MODEL),--silero-model $(SILERO_MODEL),) \
		--silero-threshold $(SILERO_THRESHOLD) \
		--silero-min-silence-ms $(SILERO_MIN_SILENCE_MS)

transcribe:
	@if [ -z "$$GOOGLE_APPLICATION_CREDENTIALS" ]; then \
		echo "Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON file"; \
		exit 1; \
	fi
	$(PYTHON) scripts/transcribe_google.py $(OUTPUT) $(RECURSIVE) \
		--language $(LANGUAGE) \
		--output-dir $(TRANSCRIPTS)

all: vad transcribe

test:
	@$(PYTHON) -c "import pytest" >/dev/null 2>&1 || (echo "pytest not installed. Run: make install-dev" && exit 1)
	$(PYTHON) -m pytest

coverage:
	@$(PYTHON) -c "import pytest" >/dev/null 2>&1 || (echo "pytest not installed. Run: make install-dev" && exit 1)
	$(PYTHON) -m pytest --cov --cov-report=term-missing

clean:
	rm -rf $(OUTPUT) $(TRANSCRIPTS)
