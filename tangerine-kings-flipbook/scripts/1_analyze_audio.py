"""Run audio analysis: librosa intensity + sections + screams + WhisperX lyrics.

Usage:
    python scripts/1_analyze_audio.py audio/TheFailing.mp3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.audio import compute_intensity, detect_screams, segment_song
from lib.transcribe import transcribe


def main(audio_path: Path) -> None:
    repo = Path(__file__).parent.parent
    config = yaml.safe_load((repo / "config" / "analyze_audio.yaml").read_text())
    output_dir = repo / "output"
    cache_dir = repo / "cache"
    output_dir.mkdir(exist_ok=True)

    fps = config["fps_keyframes"]

    print(f"[1/3] Computing intensity envelope @ {fps} fps...")
    intensity_data = compute_intensity(audio_path, fps_keyframes=fps)

    print("[2/3] Segmenting song into sections...")
    rms = [s["rms"] for s in intensity_data["intensity"]]
    sections = segment_song(
        rms,
        fps_keyframes=fps,
        expected_sections=config["section_segmentation"]["expected_sections"],
    )

    print("[3/3] Detecting screams + transcribing lyrics (WhisperX)...")
    screams = detect_screams(intensity_data["intensity"], config["scream_detection"])
    lyrics = transcribe(audio_path, cache_dir)

    timing_map = {
        "song": audio_path.stem,
        "duration_sec": intensity_data["duration_sec"],
        "fps_keyframes": fps,
        "intensity": intensity_data["intensity"],
        "sections": sections,
        "words": lyrics["words"],
        "screams": screams,
    }

    out_path = output_dir / "timing_map.json"
    out_path.write_text(json.dumps(timing_map, indent=2))

    print(f"\nWrote {out_path}")
    print(f"  Duration: {timing_map['duration_sec']} sec")
    print(f"  Keyframes: {len(timing_map['intensity'])}")
    print(f"  Sections ({len(sections)}):")
    for sec in sections:
        print(f"    {sec['start']:>7.2f} → {sec['end']:>7.2f}  {sec['label']}")
    print(f"  Screams: {len(screams)}")
    print(f"  Words transcribed: {len(timing_map['words'])}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/1_analyze_audio.py <audio_path>")
        sys.exit(1)
    main(Path(sys.argv[1]).resolve())
