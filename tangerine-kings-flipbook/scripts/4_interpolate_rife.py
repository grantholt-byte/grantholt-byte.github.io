"""Run RIFE 4.x on output/keyframes/ to produce output/frames/ at 3x.

Usage:
    python scripts/4_interpolate_rife.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.rife import expected_output_count, interpolate

MULTIPLIER = 3


def main() -> None:
    repo = Path(__file__).parent.parent
    input_dir = repo / "output" / "keyframes"
    output_dir = repo / "output" / "frames"
    rife_root = repo / "vendor" / "Practical-RIFE"

    if not rife_root.exists():
        raise SystemExit(
            f"RIFE not found at {rife_root}. Run setup.sh to install it."
        )

    n_input = len(list(input_dir.glob("*.png")))
    expected = expected_output_count(n_input, MULTIPLIER)
    print(f"Interpolating {n_input} keyframes → {expected} frames at {MULTIPLIER}x")

    n_written = interpolate(
        input_dir, output_dir, multiplier=MULTIPLIER, rife_root=rife_root
    )
    print(f"RIFE wrote {n_written} frames to {output_dir}")


if __name__ == "__main__":
    main()
