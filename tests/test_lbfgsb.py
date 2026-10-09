"""Tests for the L-BFGS-B optimiser and its ImageData/array adapter."""

import numpy as np
import pytest
from cil.framework import ImageGeometry
from cil.optimisation.functions import L2NormSquared

from krl.algorithms.lbfgsb import LBFGSBOptimizer, LBFGSBOptions, _ImageArrayAdapter


def make_image(geometry, data):
    image = geometry.allocate(0.0, dtype=np.float64)
    image.fill(np.asarray(data, dtype=np.float64))
    return image


def make_mismatched_template():
    """A float64 image whose geometry still defaults to float32 allocation.

    CIL normally syncs ``geometry.dtype`` to the data dtype, so the divergence
    is forced here to pin the adapter to the template dtype rather than the
    geometry's default allocation dtype.
    """
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2, dtype=np.float32)
    template = geometry.allocate(0.0)
    template.array = template.array.astype(np.float64)
    return template


def test_lbfgsb_reduces_simple_objective():
    geometry = ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(0).uniform(1.0, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))

    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
        options=LBFGSBOptions(ftol=1e-12, gtol=1e-12),
    )
    optimizer.run(iterations=20, verbose=0)

    assert float(np.all(np.isfinite(optimizer.solution.as_array())))
    assert optimizer.objective[-1] < optimizer.objective[0]
    assert optimizer.objective[-1] == pytest.approx(0.0, abs=1e-6)


class PrecisionProbe:
    """Quadratic functional whose value and gradient are sensitive to float64."""

    def __init__(self, target):
        self.target = target

    def __call__(self, x):
        diff = x.as_array() - self.target.as_array()
        return float(np.sum(diff * diff))

    def gradient(self, x):
        return x - self.target


def test_adapter_preserves_template_dtype():
    template = make_mismatched_template()

    assert template.dtype == np.float64
    assert template.geometry.allocate(value=0).dtype == np.float32

    adapter = _ImageArrayAdapter(template)
    assert adapter._working_image.dtype == np.float64


def test_adapter_round_trip_preserves_float64():
    template = make_mismatched_template()
    adapter = _ImageArrayAdapter(template)

    # A value that is not exactly representable as float32.
    precise = 1.23456789012345e-12
    flat = np.full(adapter.size, precise, dtype=np.float64)
    image = adapter.array_to_image(flat)

    assert image.dtype == np.float64
    assert np.array_equal(image.as_array().ravel(order="C"), flat)


def test_adapter_evaluation_and_gradient_preserve_float64():
    template = make_mismatched_template()
    precise = 1.23456789012345e-12
    target = template.clone()
    target.fill(0.0)

    probe = PrecisionProbe(target)
    optimizer = LBFGSBOptimizer(initial_estimate=template.clone(), data_fidelity=probe)
    # Cloning a template syncs its geometry dtype, so install the mismatched
    # adapter explicitly to exercise the float64 evaluation/gradient path.
    optimizer._adapter = _ImageArrayAdapter(template)

    flat = np.full(optimizer._adapter.size, precise, dtype=np.float64)
    image_array = flat.reshape(template.shape)

    value = optimizer._objective_from_array(flat)
    expected_value = float(np.sum((image_array - target.as_array()) ** 2))
    assert value == expected_value

    gradient = optimizer._gradient_from_array(flat)
    expected_gradient = (image_array - target.as_array()).ravel(order="C")
    assert np.array_equal(gradient, expected_gradient)
