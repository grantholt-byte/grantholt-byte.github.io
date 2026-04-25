"""RIFE 4.x subprocess wrapper + frame-count math."""
from __future__ import annotations

import subprocess
from pathlib import Path


def interpolate(
    input_dir: Path,
    output_dir: Path,
    multiplier: int,
    rife_root: Path,
) -> int:
    """Interpolate frames in input_dir using RIFE, writing to output_dir.

    Returns the number of output frames written.

    rife_root must contain Practical-RIFE's `inference_img.py` script.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists() or not any(input_dir.glob("*.png")):
        raise FileNotFoundError(f"No PNGs in {input_dir}")

    cmd = [
        "python",
        str(rife_root / "inference_img.py"),
        "--img_dir", str(input_dir),
        "--out_dir", str(output_dir),
        "--exp", str(multiplier - 1),
    ]
    subprocess.run(cmd, check=True)
    return len(list(output_dir.glob("*.png")))


def expected_output_count(input_count: int, multiplier: int) -> int:
    """RIFE 3x: between each pair of N keyframes, insert (multiplier - 1)
    intermediate frames → total = (N-1)*multiplier + 1.
    For multiplier=3 and N=K: 3K - 2.
    """
    if input_count <= 0:
        return 0
    return (input_count - 1) * multiplier + 1
