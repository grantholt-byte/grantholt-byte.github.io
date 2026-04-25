import json
import random
from pathlib import Path

import pytest

from lib.schedule import (
    apply_scream_override,
    bin_intensity,
    build_schedule,
    pick_shot,
    ramp_at_boundaries,
)


@pytest.fixture
def sample_timing_map(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "timing_map_sample.json").read_text())


@pytest.fixture
def rotations() -> dict:
    return {
        "intro":    ["singer_back", "crowd_pov"],
        "verse_1":  ["singer_front", "singer_3q"],
        "chorus_1": ["wide_stage", "singer_close"],
        "outro":    ["singer_close", "wide_stage"],
    }


def test_pick_shot_returns_shot_from_section_rotation(sample_timing_map, rotations):
    rng = random.Random(42)
    shot = pick_shot(2.0, sample_timing_map["sections"], rotations, 2.0, 0.0, rng)
    assert shot in {"singer_back", "crowd_pov"}


def test_pick_shot_advances_within_section_after_hold_period(sample_timing_map, rotations):
    rng = random.Random(0)
    shot_at_0 = pick_shot(0.0, sample_timing_map["sections"], rotations, 2.0, 0.0, rng)
    shot_at_3 = pick_shot(3.0, sample_timing_map["sections"], rotations, 2.0, 0.0, rng)
    assert shot_at_0 != shot_at_3


def test_pick_shot_is_deterministic_with_same_seed(sample_timing_map, rotations):
    rng_a = random.Random(99)
    rng_b = random.Random(99)
    a = [
        pick_shot(t, sample_timing_map["sections"], rotations, 2.0, 0.5, rng_a)
        for t in (1.0, 5.0, 9.0, 13.0)
    ]
    b = [
        pick_shot(t, sample_timing_map["sections"], rotations, 2.0, 0.5, rng_b)
        for t in (1.0, 5.0, 9.0, 13.0)
    ]
    assert a == b


def test_bin_intensity_maps_rms_to_label():
    bins = {"calm": [0.0, 0.30], "medium": [0.30, 0.65], "heavy": [0.65, 1.0]}
    assert bin_intensity(0.10, bins, prev=None, hysteresis_rms=0.05) == "calm"
    assert bin_intensity(0.50, bins, prev=None, hysteresis_rms=0.05) == "medium"
    assert bin_intensity(0.90, bins, prev=None, hysteresis_rms=0.05) == "heavy"


def test_bin_intensity_applies_hysteresis():
    bins = {"calm": [0.0, 0.30], "medium": [0.30, 0.65], "heavy": [0.65, 1.0]}
    # rms = 0.32, prev = "calm" → hysteresis pulls back to calm
    assert bin_intensity(0.32, bins, prev="calm", hysteresis_rms=0.05) == "calm"
    # rms = 0.36, prev = "calm" → past the hysteresis margin, advance to medium
    assert bin_intensity(0.36, bins, prev="calm", hysteresis_rms=0.05) == "medium"


def test_apply_scream_override_promotes_shot_during_scream():
    screams = [{"start": 5.0, "end": 7.0, "intensity": 0.9}]
    fallbacks = ["singer_close", "singer_kneeling", "crowd_close"]
    shot, intensity = apply_scream_override(
        t=6.0, base_shot="singer_front", base_intensity="medium",
        screams=screams, fallbacks=fallbacks,
    )
    assert shot in fallbacks
    assert intensity == "heavy"


def test_apply_scream_override_passthrough_when_not_screaming():
    screams = [{"start": 5.0, "end": 7.0, "intensity": 0.9}]
    fallbacks = ["singer_close"]
    shot, intensity = apply_scream_override(
        t=10.0, base_shot="wide_stage", base_intensity="calm",
        screams=screams, fallbacks=fallbacks,
    )
    assert shot == "wide_stage"
    assert intensity == "calm"


def test_apply_scream_override_round_robins_fallbacks_within_long_scream():
    screams = [{"start": 5.0, "end": 9.0, "intensity": 0.9}]
    fallbacks = ["singer_close", "singer_kneeling", "crowd_close"]
    chosen = []
    for t in (5.5, 6.5, 7.5, 8.5):
        shot, _ = apply_scream_override(
            t=t, base_shot="singer_front", base_intensity="medium",
            screams=screams, fallbacks=fallbacks,
        )
        chosen.append(shot)
    assert len(set(chosen)) >= 2


def test_ramp_at_boundaries_smooths_intensity_jumps():
    entries = [
        {"index": i, "t": i * 0.125, "intensity_label": "calm", "intensity_rms": 0.20}
        for i in range(8)
    ] + [
        {"index": 8 + i, "t": (8 + i) * 0.125, "intensity_label": "heavy", "intensity_rms": 0.85}
        for i in range(8)
    ]
    out = ramp_at_boundaries(entries, ramp_keyframes=3)
    labels = [e["intensity_label"] for e in out]
    # Frames near the boundary should ramp from calm → medium → heavy
    assert labels[5] == "medium"
    assert labels[6] == "medium"


def test_build_schedule_produces_per_keyframe_entries(sample_timing_map):
    rotations = {
        "intro":    ["singer_back"],
        "verse_1":  ["singer_front"],
        "chorus_1": ["wide_stage"],
        "outro":    ["singer_close"],
    }
    bins = {"calm": [0.0, 0.30], "medium": [0.30, 0.65], "heavy": [0.65, 1.0]}
    sample_timing_map["intensity"] = [
        {"t": i * 0.125, "rms": 0.10 + (i / 200), "onsets": 0, "flatness": 0.1}
        for i in range(128)
    ]

    schedule = build_schedule(
        timing_map=sample_timing_map,
        rotations=rotations,
        bins=bins,
        scream_fallbacks=["singer_close"],
        hold_sec=2.0,
        jitter=0.0,
        hysteresis_rms=0.05,
        ramp_keyframes=3,
        seed=42,
    )

    assert len(schedule) == 128
    e = schedule[0]
    assert {"index", "t", "section", "shot", "intensity_label", "intensity_rms", "seed"}.issubset(e.keys())
