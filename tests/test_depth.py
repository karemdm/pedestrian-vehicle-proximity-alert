"""Tests for depth normalisation, median patch sampling and colorisation (no model)."""

import numpy as np
import pytest

from proximity_alert.depth import colorize, normalize_inverted, robust_depth


def test_normalize_inverts_and_scales_to_unit_range():
    raw = np.array([[0, 128, 255]], dtype=np.float32)  # 255 = nearest
    z = normalize_inverted(raw)
    assert z.dtype == np.float32
    assert z[0, 2] == 0.0 and z[0, 0] == 1.0  # nearest -> 0, farthest -> 1
    assert z[0, 1] == pytest.approx(1 - 128 / 255, abs=1e-6)


def test_normalize_is_independent_of_input_scale_and_offset():
    raw = np.random.default_rng(0).random((8, 8)).astype(np.float32)
    assert np.allclose(normalize_inverted(raw), normalize_inverted(raw * 40 + 7), atol=1e-5)


@pytest.mark.parametrize("value", [0, 77, 255])
def test_constant_frame_yields_all_ones(value):
    z = normalize_inverted(np.full((5, 6), value, dtype=np.uint8))
    assert z.shape == (5, 6) and np.all(z == 1.0)


def test_nan_map_does_not_crash_and_is_degenerate():
    z = normalize_inverted(np.full((3, 3), np.nan, dtype=np.float32))
    assert np.all(z == 1.0)


def test_normalize_accepts_integer_input_without_wraparound():
    z = normalize_inverted(np.array([[10, 20]], dtype=np.uint8))
    assert z.tolist() == [[1.0, 0.0]]


def test_robust_depth_median_rejects_single_pixel_noise():
    d = np.full((20, 20), 0.5, dtype=np.float32)
    d[10, 10] = 0.0
    assert robust_depth(d, 10, 10, 7) == pytest.approx(0.5)
    assert robust_depth(d, 10, 10, 1) == 0.0


def test_robust_depth_clips_patch_at_borders():
    d = np.arange(100, dtype=np.float32).reshape(10, 10)
    # Top-left corner, 7x7 patch -> rows/cols 0..3 only.
    assert robust_depth(d, 0, 0, 7) == pytest.approx(float(np.median(d[0:4, 0:4])))
    # Bottom-right corner (last pixel).
    assert robust_depth(d, 9, 9, 7) == pytest.approx(float(np.median(d[6:10, 6:10])))
    # x == width / y == height (clipped boxes) still use the border patch.
    assert robust_depth(d, 10, 10, 7) == pytest.approx(float(np.median(d[7:10, 7:10])))


def test_robust_depth_far_outside_falls_back_to_nearest_pixel():
    d = np.arange(100, dtype=np.float32).reshape(10, 10)
    assert robust_depth(d, -50, 5, 7) == d[5, 0]
    assert robust_depth(d, 5, 500, 7) == d[9, 5]
    assert robust_depth(d, 500, -3, 3) == d[0, 9]


def test_robust_depth_even_window_is_rounded_up_to_odd():
    d = np.arange(100, dtype=np.float32).reshape(10, 10)
    assert robust_depth(d, 5, 5, 6) == robust_depth(d, 5, 5, 7)


def test_robust_depth_on_tiny_maps():
    assert robust_depth(np.array([[0.3]], dtype=np.float32), 0, 0, 7) == pytest.approx(0.3)


def test_colorize_near_is_dark_far_is_bright():
    z = np.array([[0.0, 1.0]], dtype=np.float32)
    img = colorize(z)
    assert img.shape == (1, 2, 3) and img.dtype == np.uint8
    assert img[0, 0].sum() < img[0, 1].sum()


def test_colorize_handles_exact_one_without_wraparound():
    img = colorize(np.ones((2, 2), dtype=np.float32))
    assert np.all(img == img[0, 0])


def test_has_depth_evidence_is_false_only_for_constant_maps():
    from proximity_alert.depth import has_depth_evidence

    assert not has_depth_evidence(normalize_inverted(np.full((4, 4), 90, dtype=np.uint8)))
    assert not has_depth_evidence(np.ones((4, 4), np.float32))
    assert has_depth_evidence(normalize_inverted(np.array([[0, 1]], dtype=np.uint8)))
