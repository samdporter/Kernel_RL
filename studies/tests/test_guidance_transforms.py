"""Tests for Task 4: guidance mismatch transforms."""
import numpy as np
import pytest

from krl_studies.datasets.transforms import apply_guidance_condition


def test_exact_guidance_is_unchanged():
    image = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    result = apply_guidance_condition(image, "exact", (2.0, 2.0, 2.0))
    assert np.array_equal(result, image)


def test_positive_shift_moves_an_impulse_in_physical_axes():
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, 4, 4] = 1.0
    result = apply_guidance_condition(image, "shift_p2", (1.0, 1.0, 1.0), order=0)
    assert np.unravel_index(np.argmax(result), result.shape) == (6, 6, 6)


def test_unknown_guidance_condition_raises():
    with pytest.raises(ValueError):
        apply_guidance_condition(np.zeros((3, 3, 3)), "shift_p9", (1, 1, 1))


def test_t2_condition_raises_in_transforms():
    with pytest.raises(ValueError, match="t2 guidance is loaded from the dataset"):
        apply_guidance_condition(np.zeros((3, 3, 3)), "t2", (1, 1, 1))


def test_shift_m2_moves_opposite_direction():
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, 4, 4] = 1.0
    result = apply_guidance_condition(image, "shift_m2", (1.0, 1.0, 1.0), order=0)
    assert np.unravel_index(np.argmax(result), result.shape) == (2, 2, 2)


def test_shift_p5_larger_magnitude():
    image = np.zeros((15, 15, 15), dtype=np.float32)
    image[7, 7, 7] = 1.0
    result = apply_guidance_condition(image, "shift_p5", (1.0, 1.0, 1.0), order=0)
    assert np.unravel_index(np.argmax(result), result.shape) == (12, 12, 12)


def test_voxel_size_scaling():
    # 2mm voxels, shift 2mm = 1 voxel
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, 4, 4] = 1.0
    result = apply_guidance_condition(image, "shift_p2", (2.0, 2.0, 2.0), order=0)
    assert np.unravel_index(np.argmax(result), result.shape) == (5, 5, 5)
