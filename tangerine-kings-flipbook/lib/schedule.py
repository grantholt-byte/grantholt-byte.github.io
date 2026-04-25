"""Per-keyframe schedule construction: shot rotation, intensity binning,
scream override, boundary ramping.
"""
from __future__ import annotations

import random


INTENSITY_ORDER = {"calm": 0, "medium": 1, "heavy": 2}
ORDER_TO_LABEL = {v: k for k, v in INTENSITY_ORDER.items()}


def pick_shot(
    t: float,
    sections: list[dict],
    rotations: dict[str, list[str]],
    hold_sec: float,
    jitter: float,
    rng: random.Random,
) -> str:
    """Pick a shot ID for a keyframe at time t given section rotations."""
    section = _section_at(t, sections)
    rotation = rotations.get(section["label"])
    if not rotation:
        rotation = next(iter(rotations.values()))

    section_start = section["start"]
    elapsed_in_section = t - section_start

    section_rng = random.Random(f"{section['label']}-{section_start}-{rng.random()}")
    accum = 0.0
    rotation_idx = 0
    while accum + hold_sec <= elapsed_in_section:
        rotation_idx = (rotation_idx + 1) % len(rotation)
        local_jitter = section_rng.uniform(-jitter, jitter)
        accum += max(0.5, hold_sec + local_jitter)

    return rotation[rotation_idx]


def _section_at(t: float, sections: list[dict]) -> dict:
    for sec in sections:
        if sec["start"] <= t < sec["end"]:
            return sec
    return sections[-1]


def bin_intensity(
    rms: float,
    bins: dict[str, list[float]],
    prev: str | None,
    hysteresis_rms: float,
) -> str:
    """Bin RMS into intensity label with hysteresis to prevent flicker."""
    label = _bin_raw(rms, bins)
    if prev is None or label == prev:
        return label

    prev_range = bins[prev]
    if rms < prev_range[1] + hysteresis_rms and rms >= prev_range[0] - hysteresis_rms:
        return prev
    return label


def _bin_raw(rms: float, bins: dict[str, list[float]]) -> str:
    for label, (lo, hi) in bins.items():
        if lo <= rms <= hi:
            return label
    return "medium"


def apply_scream_override(
    t: float,
    base_shot: str,
    base_intensity: str,
    screams: list[dict],
    fallbacks: list[str],
) -> tuple[str, str]:
    """Override shot+intensity if t falls within a scream window.

    Round-robins through `fallbacks` based on offset within the scream.
    """
    for s in screams:
        if s["start"] <= t < s["end"]:
            offset = t - s["start"]
            idx = int(offset / 0.5) % len(fallbacks)
            return fallbacks[idx], "heavy"
    return base_shot, base_intensity


def ramp_at_boundaries(entries: list[dict], ramp_keyframes: int) -> list[dict]:
    """Smooth large (>=2-level) intensity jumps over `ramp_keyframes` preceding frames.

    Inserts the midpoint level between prev and cur for the `ramp_keyframes` frames
    preceding the jump, so RIFE has a less-extreme pose delta to morph through.
    """
    out = [dict(e) for e in entries]
    for i in range(1, len(out)):
        prev_level = INTENSITY_ORDER[out[i - 1]["intensity_label"]]
        cur_level = INTENSITY_ORDER[out[i]["intensity_label"]]
        if abs(cur_level - prev_level) >= 2:
            mid_level = (prev_level + cur_level) // 2
            for j in range(1, ramp_keyframes + 1):
                target = i - j
                if target < 0:
                    break
                if INTENSITY_ORDER[out[target]["intensity_label"]] == prev_level:
                    out[target]["intensity_label"] = ORDER_TO_LABEL[mid_level]
    return out


def build_schedule(
    *,
    timing_map: dict,
    rotations: dict[str, list[str]],
    bins: dict[str, list[float]],
    scream_fallbacks: list[str],
    hold_sec: float,
    jitter: float,
    hysteresis_rms: float,
    ramp_keyframes: int,
    seed: int,
) -> list[dict]:
    """Build a per-keyframe schedule list."""
    rng = random.Random(seed)
    sections = timing_map["sections"]
    intensity = timing_map["intensity"]
    screams = timing_map.get("screams", [])

    entries: list[dict] = []
    prev_label = None
    for i, frame in enumerate(intensity):
        t = frame["t"]
        section = _section_at(t, sections)["label"]
        shot = pick_shot(t, sections, rotations, hold_sec, jitter, rng)

        label = bin_intensity(frame["rms"], bins, prev_label, hysteresis_rms)
        prev_label = label

        shot, label = apply_scream_override(
            t, shot, label, screams, scream_fallbacks,
        )

        entries.append({
            "index": i,
            "t": t,
            "section": section,
            "shot": shot,
            "intensity_label": label,
            "intensity_rms": frame["rms"],
            "seed": seed + (i * 17),
        })

    return ramp_at_boundaries(entries, ramp_keyframes)
