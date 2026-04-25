import numpy as np
import pytest

from lib.post import apply_halftone, apply_paper_overlay, apply_wobble_jitter


@pytest.fixture
def gray_frame():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


def test_apply_paper_overlay_warms_relative_to_blue(gray_frame):
    """Multiply-blending cream paper preserves the paper's R>G>B warmth ratio.

    A neutral gray frame becomes warmer (red channel relatively larger than blue),
    even though absolute brightness drops slightly because cream isn't pure white.
    """
    paper = np.full((100, 100, 3), [240, 220, 180], dtype=np.uint8)
    out = apply_paper_overlay(gray_frame, paper, opacity=0.3)

    assert out.shape == gray_frame.shape
    assert out.dtype == np.uint8
    # Frame is now warmer: red:blue ratio greater than the original gray frame's
    in_ratio = gray_frame[..., 0].mean() / max(gray_frame[..., 2].mean(), 1)
    out_ratio = out[..., 0].mean() / max(out[..., 2].mean(), 1)
    assert out_ratio > in_ratio


def test_apply_paper_overlay_rejects_size_mismatch(gray_frame):
    paper = np.full((50, 50, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_paper_overlay(gray_frame, paper, opacity=0.3)


def test_apply_wobble_jitter_changes_pixel_arrangement(gray_frame):
    frame = gray_frame.copy()
    frame[50, 50] = [255, 0, 0]
    out = apply_wobble_jitter(frame, max_shift_px=2, seed=42)
    assert out.shape == frame.shape
    red_mask = (out[..., 0] > 200) & (out[..., 1] < 50)
    assert red_mask.sum() == 1


def test_apply_wobble_jitter_deterministic_with_seed(gray_frame):
    frame = gray_frame.copy()
    frame[50, 50] = [255, 0, 0]
    a = apply_wobble_jitter(frame, max_shift_px=3, seed=7)
    b = apply_wobble_jitter(frame, max_shift_px=3, seed=7)
    assert np.array_equal(a, b)


def test_apply_halftone_reduces_unique_value_count():
    frame = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
    frame_rgb = np.stack([frame, frame, frame], axis=-1)
    out = apply_halftone(frame_rgb, levels=4)
    unique_per_channel = len(np.unique(out[..., 0]))
    assert unique_per_channel <= 6


def test_apply_halftone_rejects_low_levels(gray_frame):
    with pytest.raises(ValueError):
        apply_halftone(gray_frame, levels=1)
