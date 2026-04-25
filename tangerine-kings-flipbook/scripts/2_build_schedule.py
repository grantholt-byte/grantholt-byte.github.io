"""Build the per-keyframe render schedule from timing_map.json + configs.

Usage:
    python scripts/2_build_schedule.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.schedule import build_schedule


def main() -> None:
    repo = Path(__file__).parent.parent
    output_dir = repo / "output"
    timing_map = json.loads((output_dir / "timing_map.json").read_text())

    audio_cfg = yaml.safe_load((repo / "config" / "analyze_audio.yaml").read_text())
    sched_cfg = yaml.safe_load((repo / "config" / "schedule_rules.yaml").read_text())

    schedule = build_schedule(
        timing_map=timing_map,
        rotations=sched_cfg["section_rotations"],
        bins=audio_cfg["intensity_bins"],
        scream_fallbacks=sched_cfg["scream_fallbacks"],
        hold_sec=sched_cfg["hold_sec"],
        jitter=sched_cfg["jitter"],
        hysteresis_rms=sched_cfg["hysteresis_rms"],
        ramp_keyframes=sched_cfg["ramp_keyframes"],
        seed=sched_cfg["seed"],
    )

    out_path = output_dir / "keyframe_schedule.jsonl"
    with out_path.open("w") as f:
        for entry in schedule:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {out_path} ({len(schedule)} keyframes)")
    counts: dict[str, int] = {}
    for e in schedule:
        key = f"{e['shot']}.{e['intensity_label']}"
        counts[key] = counts.get(key, 0) + 1
    print("\nShot/intensity distribution:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32} {v:>5}")


if __name__ == "__main__":
    main()
