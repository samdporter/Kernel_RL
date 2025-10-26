"""Tests for hybrid kernel freezing behavior in HKRL.

These tests verify that the hybrid kernel operator correctly freezes the emission
reference when freeze_emission_kernel is set to True, ensuring the objective
function converges properly.
"""

import numpy as np
import pytest

from krl.operators.kernel_operator import (
    NUMBA_AVAIL,
    get_kernel_operator,
)

if not NUMBA_AVAIL:
    pytest.skip(
        "Numba backend required for hybrid freezing tests.",
        allow_module_level=True,
    )


class DummyGeometry:
    """Minimal geometry mock for testing."""
    def __init__(self, shape):
        self.shape = shape

    def allocate(self, value=0.0):
        data = np.full(self.shape, value, dtype=np.float64)
        return DummyImage(data)


class DummyImage:
    """Minimal image mock for testing."""
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64)

    @property
    def shape(self):
        return self._data.shape

    def as_array(self):
        return self._data

    def clone(self):
        return DummyImage(self._data.copy())

    def fill(self, values):
        self._data[...] = np.asarray(values, dtype=np.float64)


@pytest.fixture
def geometry():
    """Small 3D geometry for testing."""
    return DummyGeometry((10, 10, 10))


@pytest.fixture
def anatomical_image(geometry):
    """Anatomical image with spatial variation."""
    rng = np.random.default_rng(42)
    arr = rng.normal(loc=100, scale=20, size=geometry.shape)
    return DummyImage(arr)


@pytest.fixture
def emission_v1(geometry):
    """First emission image."""
    rng = np.random.default_rng(123)
    arr = rng.uniform(low=50, high=150, size=geometry.shape)
    return DummyImage(arr)


@pytest.fixture
def emission_v2(geometry):
    """Second emission image (different from v1)."""
    rng = np.random.default_rng(456)
    arr = rng.uniform(low=50, high=150, size=geometry.shape)
    return DummyImage(arr)


def test_hybrid_kernel_updates_without_freeze(geometry, anatomical_image, emission_v1, emission_v2):
    """Test that hybrid kernel updates emission reference when NOT frozen."""
    operator = get_kernel_operator(
        geometry,
        backend='numba',
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=True,
        use_mask=False,
    )
    operator.set_anatomical_image(anatomical_image)

    # Ensure not frozen
    assert operator.freeze_emission_kernel == False
    assert operator.frozen_emission_kernel is None

    # First forward pass with emission_v1
    result1 = operator.direct(emission_v1)
    frozen_ref1 = operator.frozen_emission_kernel.copy()

    # Verify frozen reference was updated
    assert operator.frozen_emission_kernel is not None
    np.testing.assert_array_equal(frozen_ref1, emission_v1.as_array())

    # Second forward pass with emission_v2 (different emission)
    result2 = operator.direct(emission_v2)
    frozen_ref2 = operator.frozen_emission_kernel.copy()

    # Verify frozen reference was UPDATED to emission_v2
    assert operator.frozen_emission_kernel is not None
    np.testing.assert_array_equal(frozen_ref2, emission_v2.as_array())

    # Verify the references are different
    assert not np.allclose(frozen_ref1, frozen_ref2, rtol=1e-10)


def test_hybrid_kernel_freezes_when_flag_set(geometry, anatomical_image, emission_v1, emission_v2):
    """Test that hybrid kernel DOES NOT update emission reference when frozen."""
    operator = get_kernel_operator(
        geometry,
        backend='numba',
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=True,
        use_mask=False,
    )
    operator.set_anatomical_image(anatomical_image)

    # First forward pass with emission_v1 (while not frozen)
    result1 = operator.direct(emission_v1)
    frozen_ref1 = operator.frozen_emission_kernel.copy()

    # NOW FREEZE the kernel
    operator.freeze_emission_kernel = True

    # Second forward pass with emission_v2 (different emission)
    result2 = operator.direct(emission_v2)
    frozen_ref2 = operator.frozen_emission_kernel.copy()

    # Verify frozen reference was NOT updated - should still be emission_v1
    np.testing.assert_array_equal(frozen_ref2, frozen_ref1)
    np.testing.assert_array_equal(frozen_ref2, emission_v1.as_array())

    # Verify it's NOT emission_v2
    assert not np.allclose(frozen_ref2, emission_v2.as_array(), rtol=1e-10)

    # Third forward pass with yet another emission
    emission_v3 = DummyImage(np.random.default_rng(789).uniform(50, 150, geometry.shape))
    result3 = operator.direct(emission_v3)
    frozen_ref3 = operator.frozen_emission_kernel.copy()

    # Verify STILL using emission_v1 reference
    np.testing.assert_array_equal(frozen_ref3, frozen_ref1)
    np.testing.assert_array_equal(frozen_ref3, emission_v1.as_array())


def test_hybrid_kernel_adjoint_uses_frozen_reference(geometry, anatomical_image, emission_v1, emission_v2):
    """Test that adjoint also uses the frozen emission reference."""
    operator = get_kernel_operator(
        geometry,
        backend='numba',
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=True,
        use_mask=False,
    )
    operator.set_anatomical_image(anatomical_image)

    # Forward pass to initialize normalization map and frozen reference
    _ = operator.direct(emission_v1)
    frozen_ref_before_freeze = operator.frozen_emission_kernel.copy()

    # Freeze the kernel
    operator.freeze_emission_kernel = True

    # Update emission to v2
    _ = operator.direct(emission_v2)

    # Adjoint pass - should use frozen reference (emission_v1), not current emission
    adjoint_input = geometry.allocate(1.0)
    adjoint_result = operator.adjoint(adjoint_input)

    # Verify frozen reference is still emission_v1
    np.testing.assert_array_equal(
        operator.frozen_emission_kernel,
        frozen_ref_before_freeze
    )
    np.testing.assert_array_equal(
        operator.frozen_emission_kernel,
        emission_v1.as_array()
    )


def test_hybrid_kernel_freeze_with_mask(geometry, anatomical_image, emission_v1, emission_v2):
    """Test that freezing works correctly when using sparse masking."""
    operator = get_kernel_operator(
        geometry,
        backend='numba',
        num_neighbours=5,
        sigma_anat=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=True,
        use_mask=True,
        mask_k=20,
    )
    operator.set_anatomical_image(anatomical_image)

    # First forward pass
    _ = operator.direct(emission_v1)
    frozen_ref1 = operator.frozen_emission_kernel.copy()

    # Freeze
    operator.freeze_emission_kernel = True

    # Second forward pass with different emission
    _ = operator.direct(emission_v2)
    frozen_ref2 = operator.frozen_emission_kernel.copy()

    # Verify frozen reference unchanged
    np.testing.assert_array_equal(frozen_ref2, frozen_ref1)
    np.testing.assert_array_equal(frozen_ref2, emission_v1.as_array())


def test_non_hybrid_kernel_ignores_freeze_flag(geometry, anatomical_image, emission_v1, emission_v2):
    """Test that non-hybrid kernels ignore the freeze_emission_kernel flag."""
    operator = get_kernel_operator(
        geometry,
        backend='numba',
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=False,  # NOT hybrid
        use_mask=False,
    )
    operator.set_anatomical_image(anatomical_image)

    # Set freeze flag (should be ignored since hybrid=False)
    operator.freeze_emission_kernel = True

    # Forward passes - should work normally without using frozen reference
    result1 = operator.direct(emission_v1)
    result2 = operator.direct(emission_v2)

    # Results should be different (no freezing happening)
    assert not np.allclose(result1.as_array(), result2.as_array(), rtol=1e-10)


def test_freeze_before_first_call_initializes_on_first_use(geometry, anatomical_image, emission_v1, emission_v2):
    """Test that setting freeze flag before any forward pass works correctly."""
    operator = get_kernel_operator(
        geometry,
        backend='numba',
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=True,
        use_mask=False,
    )
    operator.set_anatomical_image(anatomical_image)

    # Freeze BEFORE any forward pass
    operator.freeze_emission_kernel = True
    assert operator.frozen_emission_kernel is None

    # First forward pass - should freeze emission_v1
    _ = operator.direct(emission_v1)
    frozen_ref1 = operator.frozen_emission_kernel.copy()
    np.testing.assert_array_equal(frozen_ref1, emission_v1.as_array())

    # Second forward pass with different emission
    _ = operator.direct(emission_v2)
    frozen_ref2 = operator.frozen_emission_kernel.copy()

    # Should still be using emission_v1
    np.testing.assert_array_equal(frozen_ref2, frozen_ref1)
    np.testing.assert_array_equal(frozen_ref2, emission_v1.as_array())


def test_unfreezing_after_freeze_updates_reference(geometry, anatomical_image, emission_v1, emission_v2):
    """Test that unfreezing allows updates to resume."""
    operator = get_kernel_operator(
        geometry,
        backend='numba',
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=True,
        use_mask=False,
    )
    operator.set_anatomical_image(anatomical_image)

    # Forward pass and freeze
    _ = operator.direct(emission_v1)
    operator.freeze_emission_kernel = True
    frozen_ref1 = operator.frozen_emission_kernel.copy()

    # Forward pass while frozen
    _ = operator.direct(emission_v2)
    assert np.array_equal(operator.frozen_emission_kernel, frozen_ref1)

    # UNFREEZE
    operator.freeze_emission_kernel = False

    # Forward pass after unfreezing - should update to emission_v2
    _ = operator.direct(emission_v2)
    frozen_ref2 = operator.frozen_emission_kernel.copy()

    # Should now be emission_v2
    np.testing.assert_array_equal(frozen_ref2, emission_v2.as_array())
    assert not np.allclose(frozen_ref2, frozen_ref1, rtol=1e-10)


def test_frozen_adjoint_uses_same_reference_as_forward(geometry, anatomical_image, emission_v1, emission_v2):
    """Test that forward and adjoint use the same frozen emission reference.

    This is CRITICAL for convergence: if forward and adjoint use different emission
    references, the composed operator is no longer self-adjoint and RL will diverge.
    """
    operator = get_kernel_operator(
        geometry,
        backend='numba',
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=True,
        use_mask=False,
    )
    operator.set_anatomical_image(anatomical_image)

    # Initial forward pass with emission_v1 (establishes normalization map)
    fwd1 = operator.direct(emission_v1)
    frozen_ref_before = operator.frozen_emission_kernel.copy()

    # Freeze the kernel
    operator.freeze_emission_kernel = True

    # Forward pass with emission_v2 (different emission)
    fwd2 = operator.direct(emission_v2)
    frozen_ref_after_fwd = operator.frozen_emission_kernel.copy()

    # Adjoint pass
    test_input = geometry.allocate(1.0)
    adj2 = operator.adjoint(test_input)
    frozen_ref_after_adj = operator.frozen_emission_kernel.copy()

    # ALL three should be using emission_v1 as the frozen reference
    np.testing.assert_array_equal(frozen_ref_before, emission_v1.as_array(),
                                   err_msg="Initial frozen ref should be emission_v1")
    np.testing.assert_array_equal(frozen_ref_after_fwd, emission_v1.as_array(),
                                   err_msg="Frozen ref after forward should still be emission_v1")
    np.testing.assert_array_equal(frozen_ref_after_adj, emission_v1.as_array(),
                                   err_msg="Frozen ref after adjoint should still be emission_v1")

    # Verify it's NOT emission_v2
    assert not np.allclose(frozen_ref_after_fwd, emission_v2.as_array(), rtol=1e-10)
    assert not np.allclose(frozen_ref_after_adj, emission_v2.as_array(), rtol=1e-10)


def test_richardson_lucy_style_iteration_with_freezing(geometry, anatomical_image):
    """Simulate a few RL iterations with freezing to ensure stability.

    This mimics what happens in the actual HKRL algorithm.
    """
    operator = get_kernel_operator(
        geometry,
        backend='numba',
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=True,
        use_mask=False,
    )
    operator.set_anatomical_image(anatomical_image)

    # Initial emission estimate
    rng = np.random.default_rng(42)
    x_current = DummyImage(rng.uniform(50, 150, geometry.shape))

    frozen_refs = []

    for iteration in range(5):
        # Simulate RL: forward then adjoint
        forward_result = operator.direct(x_current)

        # Store frozen reference state
        frozen_refs.append(operator.frozen_emission_kernel.copy())

        # Freeze AFTER iteration 1 completes (so iteration 2+ use iteration 1's state)
        if iteration == 1:
            operator.freeze_emission_kernel = True

        # Adjoint (simulating correction step)
        test_input = geometry.allocate(1.0)
        adjoint_result = operator.adjoint(test_input)

        # Update x for next iteration (simplified - just add small noise)
        x_arr = x_current.as_array()
        x_arr += rng.normal(0, 0.1, geometry.shape)
        x_current.fill(x_arr)

    # After freezing at iteration 1, all subsequent frozen refs should be identical
    for i in range(2, 5):
        np.testing.assert_array_equal(
            frozen_refs[i], frozen_refs[1],
            err_msg=f"Frozen ref at iteration {i} should match frozen ref at iteration 1"
        )

    # Iteration 0 should be different from frozen state
    assert not np.allclose(frozen_refs[0], frozen_refs[1], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
