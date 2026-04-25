"""GPU detection + tiered Flux/ComfyUI setting recommendations.

Stdlib-only so this module can be imported by scripts/0_detect.py before
the Python venv has third-party deps installed.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def detect_gpus() -> list[dict]:
    """Returns [{"name": str, "vram_gb": float}] for each NVIDIA GPU, or []."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            gpus.append({"name": parts[0], "vram_gb": round(float(parts[1]) / 1024, 1)})
    return gpus


def recommend_settings(gpus: list[dict]) -> dict:
    """Recommend Flux/ComfyUI/RIFE settings based on detected GPUs.

    Tiers (decided by smallest card's VRAM):
      ≥40 GB → highvram + fp16 + batch 4
      ≥22 GB → highvram + fp16 + batch 2
      ≥16 GB → normalvram + fp8 + batch 1
      ≥12 GB → normalvram + fp8 + batch 1
      <12 GB → unsupported
    """
    if not gpus:
        return {
            "unsupported": True,
            "reason": "No NVIDIA GPU detected. Pipeline requires at least one card with 12 GB+ VRAM.",
        }

    min_vram = min(g["vram_gb"] for g in gpus)

    if min_vram < 12:
        return {
            "unsupported": True,
            "reason": f"Minimum card has {min_vram} GB VRAM; pipeline requires 12 GB+ per card.",
        }

    if min_vram >= 40:
        comfy_vram_mode, precision, batch = "--highvram", "fp16", 4
    elif min_vram >= 22:
        comfy_vram_mode, precision, batch = "--highvram", "fp16", 2
    elif min_vram >= 16:
        comfy_vram_mode, precision, batch = "--normalvram", "fp8", 1
    else:
        comfy_vram_mode, precision, batch = "--normalvram", "fp8", 1

    return {
        "unsupported": False,
        "gpus": gpus,
        "min_vram_gb": min_vram,
        "comfy_vram_mode": comfy_vram_mode,
        "flux_precision": precision,
        "batch_size": batch,
        "parallelism": len(gpus),
    }


def write_hardware_json(path: Path) -> dict:
    """Detect + recommend + write to disk. Returns the recommendation dict."""
    gpus = detect_gpus()
    rec = recommend_settings(gpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2))
    return rec


def load_hardware_json(path: Path) -> dict:
    """Read hardware.json. Raises FileNotFoundError if not present."""
    return json.loads(path.read_text())
