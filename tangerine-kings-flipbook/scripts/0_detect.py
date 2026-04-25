"""Preflight + hardware detection. Writes output/hardware.json.

Run as the first step of run.sh — fails fast with one-line fixes if the
environment is missing required tools, then writes auto-tuned recommendations
that downstream scripts consume.

Usage:
    python scripts/0_detect.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.hardware import write_hardware_json


REQUIRED_DISK_GB = 80


def _color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def red(s: str) -> str: return _color(s, "31")
def green(s: str) -> str: return _color(s, "32")
def yellow(s: str) -> str: return _color(s, "33")


def main() -> int:
    print("═══ Preflight checks ═══\n")
    failures: list[str] = []

    # Python version
    py = sys.version_info
    if py < (3, 11):
        failures.append(f"Python {py.major}.{py.minor} detected, need 3.11+. Install via pyenv or upgrade.")
    else:
        print(f"  {green('OK')}    Python {py.major}.{py.minor}.{py.micro}")

    # ffmpeg
    if shutil.which("ffmpeg") is None:
        failures.append("ffmpeg not on PATH. Install: `apt install ffmpeg` or `brew install ffmpeg`.")
    else:
        print(f"  {green('OK')}    ffmpeg")

    # git
    if shutil.which("git") is None:
        failures.append("git not on PATH. Install: `apt install git`.")
    else:
        print(f"  {green('OK')}    git")

    # Disk
    repo = Path(__file__).parent.parent
    free_gb = shutil.disk_usage(repo).free / (1024 ** 3)
    if free_gb < REQUIRED_DISK_GB:
        failures.append(f"{free_gb:.1f} GB free at {repo}, need {REQUIRED_DISK_GB} GB. Free up space and retry.")
    else:
        print(f"  {green('OK')}    {free_gb:.0f} GB free disk")

    # NVIDIA driver
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        print(f"  {green('OK')}    NVIDIA driver responsive")
    except (FileNotFoundError, subprocess.CalledProcessError):
        failures.append("nvidia-smi not responsive. Install/repair the NVIDIA driver.")

    if failures:
        print()
        for f in failures:
            print(f"  {red('FAIL')}  {f}")
        print()
        return 1

    print()
    print("═══ Hardware detection ═══\n")
    rec = write_hardware_json(repo / "output" / "hardware.json")

    if rec.get("unsupported"):
        print(f"  {red('UNSUPPORTED')}  {rec['reason']}")
        return 1

    print(f"  GPUs detected: {len(rec['gpus'])}")
    for g in rec["gpus"]:
        print(f"    • {g['name']}  {g['vram_gb']} GB")
    print()
    print(f"  Smallest card has {rec['min_vram_gb']} GB → using:")
    print(f"    ComfyUI mode:    {rec['comfy_vram_mode']}")
    print(f"    Flux precision:  {rec['flux_precision']}")
    print(f"    Batch size:      {rec['batch_size']}")
    print(f"    Parallelism:     {rec['parallelism']}× (one ComfyUI per GPU)")
    print()

    keyframes = 1582
    sec_per_frame = {"fp16": 3.0, "fp8": 5.0}[rec["flux_precision"]]
    est_min = (keyframes * sec_per_frame) / rec["parallelism"] / 60
    print(f"  Estimated keyframe render time: ~{est_min:.0f} min")
    print()
    print(f"  {green('Ready.')}  Wrote output/hardware.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
