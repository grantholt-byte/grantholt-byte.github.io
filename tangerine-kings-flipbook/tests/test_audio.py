"""Tests for lib.audio.

These tests require librosa + soundfile, which build from source on macOS
(needs LLVM). They run cleanly on Linux (the friend's rig) where wheels
are available. Locally on macOS without LLVM, these will be skipped.
"""
from pathlib import Path

import pytest

pytest.importorskip("librosa", reason="librosa not installed (macOS without LLVM)")

from lib.audio import compute_intensity, detect_screams, segment_song


def _ensure_fixture(fixtures_dir: Path) -> Path:
    """Generate the tiny.mp3 test fixture if missing.

    5-second mono test audio: 220 Hz sine + drum-like transients at 1.0, 2.5, 4.0 sec.
    """
    path = fixtures_dir / "tiny.mp3"
    if path.exists():
        return path
    soundfile = pytest.importorskip("soundfile")
    import numpy as np

    sr = 22050
    t = np.linspace(0, 5, sr * 5)
    signal = 0.3 * np.sin(2 * np.pi * 220 * t)
    rng = np.random.default_rng(42)
    for hit_t in (1.0, 2.5, 4.0):
        idx = int(hit_t * sr)
        signal[idx:idx + 200] += rng.standard_normal(200) * 0.5
    soundfile.write(str(path), signal, sr)
    return path


@pytest.fixture
def tiny_mp3(fixtures_dir: Path) -> Path:
    return _ensure_fixture(fixtures_dir)


def test_compute_intensity_returns_per_second_envelope(tiny_mp3: Path):
    result = compute_intensity(tiny_mp3, fps_keyframes=8)

    assert "duration_sec" in result
    assert 4.5 < result["duration_sec"] < 5.5
    assert "intensity" in result
    assert isinstance(result["intensity"], list)
    assert 38 <= len(result["intensity"]) <= 42

    sample = result["intensity"][0]
    assert set(sample.keys()) == {"t", "rms", "onsets", "flatness"}
    assert 0.0 <= sample["rms"] <= 1.0
    assert sample["onsets"] >= 0
    assert 0.0 <= sample["flatness"] <= 1.0


def test_compute_intensity_normalizes_rms_to_song_max(tiny_mp3: Path):
    result = compute_intensity(tiny_mp3, fps_keyframes=8)
    rms_values = [s["rms"] for s in result["intensity"]]
    assert max(rms_values) == pytest.approx(1.0, abs=0.01)


def test_segment_song_returns_labeled_sections(tiny_mp3: Path):
    intensity = compute_intensity(tiny_mp3, fps_keyframes=8)
    rms = [s["rms"] for s in intensity["intensity"]]

    sections = segment_song(rms, fps_keyframes=8, expected_sections=3)

    assert len(sections) >= 1
    for sec in sections:
        assert {"start", "end", "label"}.issubset(sec.keys())
        assert sec["start"] < sec["end"]
        assert sec["label"] in {"intro", "verse", "chorus", "bridge", "breakdown", "outro"}

    for a, b in zip(sections, sections[1:]):
        assert a["end"] == pytest.approx(b["start"], abs=0.2)


def test_detect_screams_finds_high_flatness_high_rms_sustained_regions():
    intensity = [
        {"t": 0.0, "rms": 0.10, "onsets": 0, "flatness": 0.10},
        {"t": 0.125, "rms": 0.20, "onsets": 0, "flatness": 0.10},
        {"t": 1.0, "rms": 0.85, "onsets": 1, "flatness": 0.40},
        {"t": 1.125, "rms": 0.90, "onsets": 0, "flatness": 0.42},
        {"t": 1.25, "rms": 0.88, "onsets": 0, "flatness": 0.41},
        {"t": 1.375, "rms": 0.92, "onsets": 0, "flatness": 0.43},
        {"t": 1.5, "rms": 0.86, "onsets": 0, "flatness": 0.40},
        {"t": 1.625, "rms": 0.20, "onsets": 0, "flatness": 0.15},
    ]
    cfg = {"rms_min": 0.7, "flatness_min": 0.3, "min_duration_sec": 0.4}

    screams = detect_screams(intensity, cfg)

    assert len(screams) == 1
    s = screams[0]
    assert s["start"] == pytest.approx(1.0, abs=0.05)
    assert s["end"] == pytest.approx(1.5, abs=0.15)
    assert s["intensity"] > 0.85


def test_detect_screams_ignores_short_spikes():
    intensity = [
        {"t": 0.0, "rms": 0.10, "onsets": 0, "flatness": 0.10},
        {"t": 0.125, "rms": 0.85, "onsets": 0, "flatness": 0.40},
        {"t": 0.25, "rms": 0.10, "onsets": 0, "flatness": 0.10},
    ]
    cfg = {"rms_min": 0.7, "flatness_min": 0.3, "min_duration_sec": 0.4}

    assert detect_screams(intensity, cfg) == []
