import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lib.transcribe import transcribe


@pytest.fixture
def stub_mp3(tmp_path: Path) -> Path:
    """Stand-in audio file (not a real MP3 — bytes only, since transcribe hashes contents)."""
    p = tmp_path / "stub.mp3"
    p.write_bytes(b"FAKE_AUDIO_BYTES_FOR_HASHING_ONLY")
    return p


def test_transcribe_uses_cache_when_present(tmp_path: Path, stub_mp3: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    h = hashlib.sha256(stub_mp3.read_bytes()).hexdigest()[:16]
    cache_file = cache_dir / f"whisperx_{h}.json"
    cache_file.write_text(json.dumps({
        "words": [{"w": "hello", "start": 0.5, "end": 0.8}],
        "text": "hello",
    }))

    with patch("lib.transcribe._run_whisperx") as mock_run:
        result = transcribe(stub_mp3, cache_dir)
        mock_run.assert_not_called()

    assert result["text"] == "hello"
    assert result["words"][0]["w"] == "hello"


def test_transcribe_invokes_whisperx_on_cache_miss(tmp_path: Path, stub_mp3: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    fake_response = {"words": [{"w": "test", "start": 0.0, "end": 0.5}], "text": "test"}
    with patch("lib.transcribe._run_whisperx", return_value=fake_response) as mock_run:
        result = transcribe(stub_mp3, cache_dir)
        mock_run.assert_called_once()

    assert result == fake_response
    cache_files = list(cache_dir.glob("whisperx_*.json"))
    assert len(cache_files) == 1
