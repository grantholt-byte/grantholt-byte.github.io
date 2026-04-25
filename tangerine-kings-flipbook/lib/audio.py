"""Audio feature extraction: per-keyframe intensity, song segmentation, scream detection.

All functions are pure-numeric and deterministic given a fixed audio file.
librosa is the only third-party dependency.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def compute_intensity(audio_path: Path, fps_keyframes: int) -> dict:
    """Extract per-keyframe intensity envelope from an audio file.

    Returns:
        {
          "duration_sec": float,
          "sample_rate": int,
          "intensity": [{"t": float, "rms": 0..1, "onsets": int, "flatness": 0..1}, ...]
        }
    """
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(y) / sr

    hop = int(sr / fps_keyframes)
    frame_length = hop * 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    rms_norm = rms / (rms.max() + 1e-9)

    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=frame_length, hop_length=hop
    )[0]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames"
    )

    n = min(len(rms_norm), len(flatness))
    intensity = []
    for i in range(n):
        t = i * hop / sr
        onsets_in_window = int(np.sum((onset_frames >= i) & (onset_frames < i + 1)))
        intensity.append({
            "t": round(t, 3),
            "rms": float(round(rms_norm[i], 4)),
            "onsets": onsets_in_window,
            "flatness": float(round(flatness[i], 4)),
        })

    return {
        "duration_sec": round(duration, 3),
        "sample_rate": int(sr),
        "intensity": intensity,
    }


def segment_song(
    rms_envelope: list[float],
    fps_keyframes: int,
    expected_sections: int = 8,
) -> list[dict]:
    """Segment a song into labeled sections using RMS-based agglomerative clustering."""
    arr = np.asarray(rms_envelope).reshape(1, -1)
    if arr.shape[1] < expected_sections + 2:
        return [{
            "start": 0.0,
            "end": round(arr.shape[1] / fps_keyframes, 3),
            "label": "intro",
        }]

    boundary_frames = librosa.segment.agglomerative(arr, k=expected_sections)
    boundaries_sec = sorted(set(
        [0.0]
        + [round(int(f) / fps_keyframes, 3) for f in boundary_frames]
        + [round(arr.shape[1] / fps_keyframes, 3)]
    ))

    labels = _heuristic_section_labels(rms_envelope, boundaries_sec, fps_keyframes)
    sections = []
    for (start, end), label in zip(zip(boundaries_sec, boundaries_sec[1:]), labels):
        sections.append({"start": start, "end": end, "label": label})
    return sections


def _heuristic_section_labels(
    rms: list[float], boundaries_sec: list[float], fps: int
) -> list[str]:
    """Heuristic: first = intro, last = outro, highest-mean = chorus, low-mean middle = bridge."""
    n = len(boundaries_sec) - 1
    means = []
    for start, end in zip(boundaries_sec, boundaries_sec[1:]):
        a, b = int(start * fps), int(end * fps)
        means.append(float(np.mean(rms[a:b])) if b > a else 0.0)

    labels = ["verse"] * n
    if n >= 1:
        labels[0] = "intro"
    if n >= 2:
        labels[-1] = "outro"
    if n >= 3:
        peak_idx = means[1:-1].index(max(means[1:-1])) + 1
        labels[peak_idx] = "chorus"
    if n >= 5:
        middle = list(enumerate(means[1:-1], start=1))
        middle.sort(key=lambda kv: kv[1])
        for idx, _ in middle:
            if labels[idx] == "verse":
                labels[idx] = "bridge"
                break
    if n >= 7:
        candidates = [(i, m) for i, m in enumerate(means) if labels[i] == "verse"]
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        if candidates:
            labels[candidates[0][0]] = "breakdown"
    return labels


def detect_screams(intensity: list[dict], cfg: dict) -> list[dict]:
    """Find sustained regions of high RMS + high spectral flatness.

    cfg: {"rms_min": float, "flatness_min": float, "min_duration_sec": float}
    Returns: [{"start": float, "end": float, "intensity": float}]
    """
    if not intensity:
        return []

    rms_min = cfg["rms_min"]
    flat_min = cfg["flatness_min"]
    min_dur = cfg["min_duration_sec"]

    runs = []
    cur_start = None
    cur_peak = 0.0

    for i, frame in enumerate(intensity):
        is_scream = frame["rms"] >= rms_min and frame["flatness"] >= flat_min
        if is_scream:
            if cur_start is None:
                cur_start = frame["t"]
                cur_peak = frame["rms"]
            else:
                cur_peak = max(cur_peak, frame["rms"])
        else:
            if cur_start is not None:
                cur_end = intensity[i - 1]["t"]
                if cur_end - cur_start >= min_dur:
                    runs.append({
                        "start": cur_start,
                        "end": cur_end,
                        "intensity": round(cur_peak, 3),
                    })
                cur_start = None
                cur_peak = 0.0

    if cur_start is not None:
        cur_end = intensity[-1]["t"]
        if cur_end - cur_start >= min_dur:
            runs.append({
                "start": cur_start,
                "end": cur_end,
                "intensity": round(cur_peak, 3),
            })

    return runs
