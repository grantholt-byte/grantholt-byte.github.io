"""Frame-level post-processing filters: paper overlay, wobble jitter, halftone."""
from __future__ import annotations

import numpy as np


def apply_paper_overlay(
    frame: np.ndarray, paper: np.ndarray, opacity: float
) -> np.ndarray:
    """Multiply-blend a paper texture over the frame at the given opacity."""
    if paper.shape != frame.shape:
        raise ValueError(f"paper shape {paper.shape} != frame shape {frame.shape}")
    blended = frame.astype(np.float32) * (paper.astype(np.float32) / 255.0)
    return (
        frame.astype(np.float32) * (1 - opacity) + blended * opacity
    ).astype(np.uint8)


def apply_wobble_jitter(
    frame: np.ndarray, max_shift_px: int, seed: int
) -> np.ndarray:
    """Random small per-frame XY shift, simulating hand-drawn registration jitter."""
    rng = np.random.default_rng(seed)
    dx = int(rng.integers(-max_shift_px, max_shift_px + 1))
    dy = int(rng.integers(-max_shift_px, max_shift_px + 1))
    return np.roll(frame, (dy, dx), axis=(0, 1))


def apply_halftone(frame: np.ndarray, levels: int) -> np.ndarray:
    """Quantize each channel to `levels` discrete brightness steps."""
    if levels < 2:
        raise ValueError("levels must be >= 2")
    step = 255 / (levels - 1)
    out = np.round(frame.astype(np.float32) / step) * step
    return np.clip(out, 0, 255).astype(np.uint8)
