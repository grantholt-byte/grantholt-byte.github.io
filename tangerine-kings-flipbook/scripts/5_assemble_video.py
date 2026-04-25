"""Apply post-processing to interpolated frames and mux with audio.

Usage:
    python scripts/5_assemble_video.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.post import apply_halftone, apply_paper_overlay, apply_wobble_jitter

PLAYBACK_FPS = 24


def main() -> None:
    repo = Path(__file__).parent.parent
    frames_dir = repo / "output" / "frames"
    processed_dir = repo / "output" / "frames_processed"
    audio_path = repo / "audio" / "TheFailing.mp3"
    out_path = repo / "output" / "the_failing_flipbook.mp4"

    processed_dir.mkdir(parents=True, exist_ok=True)
    paper_path = repo / "audio" / "paper_texture.png"  # optional
    paper = cv2.imread(str(paper_path)) if paper_path.exists() else None

    print("Applying post filters to frames...")
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        sys.exit(f"No frames in {frames_dir}; run earlier scripts first.")

    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if paper is not None:
            paper_resized = cv2.resize(paper, (img.shape[1], img.shape[0]))
            img = apply_paper_overlay(img, paper_resized, opacity=0.25)
        img = apply_wobble_jitter(img, max_shift_px=1, seed=i)
        img = apply_halftone(img, levels=8)
        cv2.imwrite(str(processed_dir / frame_path.name), img)
        if i % 200 == 0:
            print(f"  {i}/{len(frames)}")

    print(f"Muxing {len(frames)} frames @ {PLAYBACK_FPS} fps with audio...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(PLAYBACK_FPS),
        "-i", str(processed_dir / "%06d.png"),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
