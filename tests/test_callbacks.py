"""Tests for the KRL callbacks driven by a real CIL Algorithm.run()."""

import numpy as np
import pytest
from cil.framework import ImageGeometry
from cil.optimisation.algorithms import Algorithm

from krl.callbacks import NRMSECallback, SaveIterationCallback
from krl.utils import load_image


class SignedAlgorithm(Algorithm):
    """Minimal CIL algorithm with a signed solution (x decreases by one per update)."""

    def __init__(self, x0):
        super().__init__()
        self.x = x0.clone()
        self.configured = True

    def update(self):
        self.x = self.x - 1.0

    def update_objective(self):
        self.loss.append(float(self.x.as_array().sum()))


class AliasOperator:
    """Operator whose direct() returns the input container itself."""

    def direct(self, image):
        return image


@pytest.fixture
def geometry():
    return ImageGeometry(voxel_num_x=8, voxel_num_y=6, voxel_num_z=4)


@pytest.fixture
def ground_truth(geometry):
    """Positive ground truth (maximum = 192)."""
    img = geometry.allocate(0.0)
    img.fill(np.arange(1, 193, dtype=np.float32).reshape(4, 6, 8))
    return img


@pytest.fixture
def signed_start(geometry):
    """Signed integer-valued start (range -96..95), so clamping is detectable."""
    img = geometry.allocate(0.0)
    img.fill(np.arange(-96, 96, dtype=np.float32).reshape(4, 6, 8))
    return img


def test_nrmse_values_and_csv_rows(ground_truth, signed_start, tmp_path):
    """NRMSE matches RMSE/max(ground_truth) at every iteration and CSV row."""
    algorithm = SignedAlgorithm(signed_start)
    csv_path = tmp_path / "nrmse.csv"
    callback = NRMSECallback(ground_truth, csv_path, interval=1, verbose=False)
    algorithm.run(iterations=3, callbacks=[callback], verbose=0)

    gt_array = ground_truth.as_array()
    start = signed_start.as_array()

    assert [iteration for iteration, _ in callback.nrmse_values] == [0, 1, 2, 3]
    for iteration, value in callback.nrmse_values:
        expected = np.sqrt(np.mean((start - iteration - gt_array) ** 2)) / gt_array.max()
        assert value == pytest.approx(expected, rel=1e-5)

    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "iteration,nrmse"
    rows = [line.split(",") for line in lines[1:]]
    assert [int(row[0]) for row in rows] == [0, 1, 2, 3]
    for (_, value), row in zip(callback.nrmse_values, rows):
        assert float(row[1]) == pytest.approx(value, rel=1e-7)


def test_nrmse_interval(ground_truth, signed_start, tmp_path):
    """The CSV only records iterations that are multiples of the interval."""
    algorithm = SignedAlgorithm(signed_start)
    callback = NRMSECallback(ground_truth, tmp_path / "nrmse.csv", interval=2, verbose=False)
    algorithm.run(iterations=5, callbacks=[callback], verbose=0)

    assert [iteration for iteration, _ in callback.nrmse_values] == [0, 2, 4]


def test_save_schedule_files_and_arrays(signed_start, tmp_path):
    """Saves iterations 0, 1 (first N) and 2, 4 (interval), clamped to non-negative."""
    algorithm = SignedAlgorithm(signed_start)
    output_dir = tmp_path / "iterations"
    callback = SaveIterationCallback(output_dir, interval=2, prefix="iter", save_first_n=1)
    algorithm.run(iterations=5, callbacks=[callback], verbose=0)

    files = sorted(path.name for path in output_dir.glob("iter_*.nii.gz"))
    assert files == [
        "iter_0000.nii.gz",
        "iter_0001.nii.gz",
        "iter_0002.nii.gz",
        "iter_0004.nii.gz",
    ]

    start = signed_start.as_array()
    for name, iteration in [("iter_0000", 0), ("iter_0001", 1), ("iter_0002", 2), ("iter_0004", 4)]:
        saved = load_image(output_dir / f"{name}.nii.gz").as_array()
        np.testing.assert_allclose(saved, np.maximum(start - iteration, 0), rtol=1e-5, atol=1e-6)
        assert saved.min() >= 0


def test_save_leaves_signed_solution_unchanged(signed_start, tmp_path):
    """Saving must not clamp the algorithm's own solution, only the saved copy."""
    reference = SignedAlgorithm(signed_start)
    reference.run(iterations=4, callbacks=[], verbose=0)

    watched = SignedAlgorithm(signed_start)
    callback = SaveIterationCallback(tmp_path / "iterations", interval=1, save_first_n=2)
    watched.run(iterations=4, callbacks=[callback], verbose=0)

    solution = watched.solution.as_array()
    assert solution.min() < 0
    np.testing.assert_array_equal(solution, reference.solution.as_array())

    saved = load_image(tmp_path / "iterations" / "iter_0000.nii.gz").as_array()
    assert saved.min() >= 0
    np.testing.assert_array_equal(saved, np.maximum(signed_start.as_array(), 0))


def test_save_with_aliasing_kernel_operator_is_observational(signed_start, tmp_path):
    """Even if direct() returns the solution itself, saving must not clamp it."""
    reference = SignedAlgorithm(signed_start)
    reference.run(iterations=3, callbacks=[], verbose=0)

    watched = SignedAlgorithm(signed_start)
    callback = SaveIterationCallback(
        tmp_path / "iterations", interval=1, save_first_n=0, kernel_operator=AliasOperator()
    )
    watched.run(iterations=3, callbacks=[callback], verbose=0)

    assert watched.solution.as_array().min() < 0
    np.testing.assert_array_equal(watched.solution.as_array(), reference.solution.as_array())

    saved = load_image(tmp_path / "iterations" / "iter_0000.nii.gz").as_array()
    assert saved.min() >= 0
    np.testing.assert_array_equal(saved, np.maximum(signed_start.as_array(), 0))


@pytest.mark.parametrize("bad_interval", [0, -1, 2.5, "3", True, False])
def test_save_callback_rejects_invalid_interval(bad_interval, tmp_path):
    with pytest.raises(ValueError, match="interval must be a positive integer"):
        SaveIterationCallback(tmp_path / "out", interval=bad_interval)


@pytest.mark.parametrize("bad_interval", [0, -3, 1.5, True, False])
def test_nrmse_callback_rejects_invalid_interval(bad_interval, ground_truth, tmp_path):
    with pytest.raises(ValueError, match="interval must be a positive integer"):
        NRMSECallback(ground_truth, tmp_path / "nrmse.csv", interval=bad_interval)


@pytest.mark.parametrize("bad_count", [-1, 1.5, "2", True, False])
def test_save_callback_rejects_invalid_save_first_n(bad_count, tmp_path):
    with pytest.raises(ValueError, match="save_first_n must be a non-negative integer"):
        SaveIterationCallback(tmp_path / "out", save_first_n=bad_count)


def test_save_callback_accepts_zero_save_first_n(tmp_path):
    SaveIterationCallback(tmp_path / "out", save_first_n=0)


@pytest.mark.parametrize(
    "bad_ground_truth",
    [
        np.zeros((4, 6, 8), dtype=np.float32),
        -np.ones((4, 6, 8), dtype=np.float32),
        np.full((4, 6, 8), np.nan, dtype=np.float32),
        np.full((4, 6, 8), np.inf, dtype=np.float32),
    ],
    ids=["zero", "negative", "nan", "inf"],
)
def test_nrmse_rejects_invalid_normalisation(bad_ground_truth, geometry, tmp_path):
    ground_truth = geometry.allocate(0.0)
    ground_truth.fill(bad_ground_truth)

    with pytest.raises(ValueError, match="positive finite ground truth maximum"):
        NRMSECallback(ground_truth, tmp_path / "nrmse.csv")
