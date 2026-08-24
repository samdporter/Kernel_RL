import numpy as np
import pytest

from krl_studies.simulation.geometry import resample_array_zyx


def test_constant_volume_is_preserved_when_resampled():
    source = np.full((10, 20, 30), 3.5, dtype=np.float32)
    result, target_shape = resample_array_zyx(
        source, source_voxel_mm=(2.0, 1.0, 1.0), target_voxel_mm=(1.0, 2.0, 2.0)
    )
    assert result.shape == target_shape == (20, 10, 15)
    assert np.allclose(result, 3.5, atol=1e-5)


def test_resampling_preserves_physical_extent():
    source = np.zeros((10, 20, 30), dtype=np.float32)
    source[5, 10, 15] = 1.0
    result, target_shape = resample_array_zyx(
        source,
        source_voxel_mm=(2.0, 1.0, 1.0),
        target_voxel_mm=(1.0, 2.0, 2.0),
    )
    assert result.shape == target_shape
    peak = np.array(np.unravel_index(np.argmax(result), result.shape))
    source_position_mm = np.array((5, 10, 15)) * np.array((2.0, 1.0, 1.0))
    target_position_mm = peak * np.array((1.0, 2.0, 2.0))
    assert np.all(np.abs(target_position_mm - source_position_mm) <= (1.0, 2.0, 2.0))


def test_invalid_voxel_sizes_raise():
    with pytest.raises(ValueError):
        resample_array_zyx(np.ones((2, 2, 2)), (0, 1, 1), (1, 1, 1))


def test_resample_to_fov_scales_physically_and_embeds_centred():
    from krl_studies.simulation.geometry import resample_to_fov_zyx

    # 40 mm-wide block in a 60 mm volume, FOV grid at 2 mm voxels.
    source = np.zeros((60, 60, 60), dtype=np.float32)
    source[10:50, 10:50, 10:50] = 1.0
    fov, shape = resample_to_fov_zyx(source, (1.0, 1.0, 1.0), (30, 45, 45), (2.0, 2.0, 2.0))
    assert shape == (30, 45, 45)
    # Centre value preserved; corners padded with zero.
    assert fov[15, 22, 22] == 1.0
    assert fov[0, 0, 0] == 0.0
    # Block width shrinks from 40 samples @1 mm to 20 samples @2 mm.
    row = fov[15, 22, :]
    assert np.count_nonzero(row >= 0.5) == 20


def test_resample_from_fov_crops_centre_window_and_restores_grid():
    from krl_studies.simulation.geometry import resample_from_fov_zyx

    fov = np.zeros((30, 45, 45), dtype=np.float32)
    fov[15, 22, 22] = 10.0
    back = resample_from_fov_zyx(fov, (2.0, 2.0, 2.0), (44, 44, 44), (1.0, 1.0, 1.0))
    assert back.shape == (44, 44, 44)
    peak = np.array(np.unravel_index(np.argmax(back), back.shape))
    # Centre is preserved to within one target voxel (even-sized grids leave
    # an unavoidable half-voxel centring ambiguity).
    assert np.all(np.abs(peak - 21.5) <= 1.0)

    block = np.zeros((30, 45, 45), dtype=np.float32)
    block[15, 16:28, 16:28] = 1.0  # 12 samples @ 2 mm
    restored = resample_from_fov_zyx(block, (2.0, 2.0, 2.0), (44, 44, 44), (1.0, 1.0, 1.0))
    centre = int(round(21.5))
    width = np.count_nonzero(restored[centre, centre, :] >= 0.5)
    assert abs(width - 24.0) <= 2.0
