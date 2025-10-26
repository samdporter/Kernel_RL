"""Tests for MAPRL preconditioner functionality."""
import importlib
import numpy as np
import pytest


@pytest.fixture
def maprl_module():
    module = importlib.import_module("src.krl.algorithms.maprl")
    return importlib.reload(module)


class DummyImage:
    """Minimal ImageData-like class for testing."""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)

    def clone(self):
        return DummyImage(self.data.copy())

    def as_array(self):
        return self.data

    def fill(self, value):
        if np.isscalar(value):
            self.data.fill(value)
        else:
            self.data = np.array(value, dtype=np.float64)

    def maximum(self, value, out=None):
        result = np.maximum(self.data, value)
        if out is None:
            return DummyImage(result)
        out.data = result
        return out

    def __add__(self, other):
        return DummyImage(self.data + (other.data if hasattr(other, 'data') else other))

    def __sub__(self, other):
        return DummyImage(self.data - (other.data if hasattr(other, 'data') else other))

    def __mul__(self, other):
        return DummyImage(self.data * (other.data if hasattr(other, 'data') else other))

    def __truediv__(self, other):
        return DummyImage(self.data / (other.data if hasattr(other, 'data') else other))


class DummyFunctional:
    """Minimal Function-like class for testing."""
    def __init__(self, grad_value=0.1):
        self.grad_value = grad_value

    def gradient(self, x):
        return DummyImage(np.ones_like(x.as_array()) * self.grad_value)

    def __call__(self, x):
        return float(np.sum(x.as_array()**2))


def test_maprl_preconditioner_initialization(maprl_module):
    """Test that MAPRL can be initialized with preconditioner parameters."""
    initial = DummyImage(np.ones((5, 5)))
    data_fid = DummyFunctional()
    prior = DummyFunctional()

    def test_preconditioner(x):
        return DummyImage(np.ones_like(x.as_array()) * 0.5)

    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fid,
        prior=prior,
        step_size=1.0,
        initial_line_search=False,
        armijo_iterations=0,
        preconditioner=test_preconditioner,
        preconditioner_update_initial=5,
        preconditioner_update_interval=10,
    )

    assert maprl.preconditioner is not None
    assert maprl.preconditioner_update_initial == 5
    assert maprl.preconditioner_update_interval == 10
    assert callable(maprl.preconditioner)
    assert maprl._preconditioner_image is None  # Not computed yet


def test_maprl_preconditioner_update_initial_iterations(maprl_module):
    """Test that preconditioner is updated every iteration for first N iterations."""
    initial = DummyImage(np.ones((5, 5)))
    data_fid = DummyFunctional(grad_value=0.1)
    prior = DummyFunctional(grad_value=0.05)

    update_count = []

    def test_preconditioner(x):
        update_count.append(True)
        return DummyImage(np.ones_like(x.as_array()) * 0.5)

    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fid,
        prior=prior,
        step_size=1.0,
        initial_line_search=False,
        armijo_iterations=0,
        preconditioner=test_preconditioner,
        preconditioner_update_initial=3,
        preconditioner_update_interval=10,
    )

    # Run first 5 iterations
    for i in range(1, 6):
        maprl.iteration = i
        maprl.update()

    # Should update on iterations 1, 2, 3 (initial period)
    # Then not on 4, 5
    assert len(update_count) == 3


def test_maprl_preconditioner_update_periodic(maprl_module):
    """Test that preconditioner updates periodically after initial iterations."""
    initial = DummyImage(np.ones((5, 5)))
    data_fid = DummyFunctional(grad_value=0.1)
    prior = DummyFunctional(grad_value=0.05)

    update_iterations = []

    def test_preconditioner(x):
        update_iterations.append(maprl.iteration)
        return DummyImage(np.ones_like(x.as_array()) * 0.5)

    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fid,
        prior=prior,
        step_size=1.0,
        initial_line_search=False,
        armijo_iterations=0,
        preconditioner=test_preconditioner,
        preconditioner_update_initial=2,
        preconditioner_update_interval=5,
    )

    # Run 15 iterations
    for i in range(1, 16):
        maprl.iteration = i
        maprl.update()

    # Should update on: 1, 2 (initial), then 5, 10, 15 (periodic)
    expected = [1, 2, 5, 10, 15]
    assert update_iterations == expected


def test_maprl_preconditioner_application(maprl_module):
    """Test that preconditioner is applied correctly in the update step."""
    initial = DummyImage(np.ones((3, 3)))
    data_fid = DummyFunctional(grad_value=1.0)
    prior = DummyFunctional(grad_value=0.0)

    # Preconditioner value (should multiply gradient)
    def test_preconditioner(x):
        return DummyImage(np.ones_like(x.as_array()) * 2.0)

    eps = 1e-8
    relaxation_eta = 0.01
    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fid,
        prior=prior,
        step_size=0.1,
        eps=eps,
        relaxation_eta=relaxation_eta,
        initial_line_search=False,
        armijo_iterations=0,
        preconditioner=test_preconditioner,
        preconditioner_update_initial=1,
    )

    # Perform one update
    maprl.iteration = 1
    initial_values = maprl.x.as_array().copy()
    maprl.update()
    updated_values = maprl.x.as_array()

    # New formula: x = x - grad * precond * step
    # With preconditioner = 2.0, gradient = 1.0
    # step_size at iteration 1 = 0.1 / (1 + 0.01 * 1)
    step_actual = 0.1 / (1 + relaxation_eta * 1)
    expected = initial_values - 1.0 * 2.0 * step_actual
    expected = np.maximum(expected, 0.0)  # Non-negativity constraint

    assert np.allclose(updated_values, expected, rtol=1e-5)


def test_maprl_without_preconditioner(maprl_module):
    """Test that MAPRL works correctly without preconditioner."""
    initial = DummyImage(np.ones((3, 3)))
    data_fid = DummyFunctional(grad_value=0.5)
    prior = DummyFunctional(grad_value=0.0)

    eps = 1e-8
    relaxation_eta = 0.01
    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fid,
        prior=prior,
        step_size=0.1,
        eps=eps,
        relaxation_eta=relaxation_eta,
        initial_line_search=False,
        armijo_iterations=0,
        preconditioner=None,
    )

    # Perform one update
    maprl.iteration = 1
    initial_values = maprl.x.as_array().copy()
    maprl.update()
    updated_values = maprl.x.as_array()

    # Without preconditioner: x = x - (x + eps) * grad * step
    # step_size at iteration 1 = 0.1 / (1 + 0.01 * 1)
    step_actual = 0.1 / (1 + relaxation_eta * 1)
    expected = initial_values - (initial_values + eps) * 0.5 * step_actual
    expected = np.maximum(expected, 0.0)

    assert np.allclose(updated_values, expected, rtol=1e-5)


def test_maprl_preconditioner_static_image(maprl_module):
    """Test that MAPRL works with a static ImageData preconditioner."""
    initial = DummyImage(np.ones((3, 3)))
    data_fid = DummyFunctional(grad_value=1.0)
    prior = DummyFunctional(grad_value=0.0)

    # Static preconditioner image
    static_precond = DummyImage(np.ones((3, 3)) * 2.0)

    eps = 1e-8
    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fid,
        prior=prior,
        step_size=0.1,
        eps=eps,
        initial_line_search=False,
        armijo_iterations=0,
        preconditioner=static_precond,
    )

    # Perform one update
    maprl.iteration = 1
    maprl.update()

    # Check that preconditioner was used
    assert maprl._preconditioner_image is not None
    assert np.allclose(maprl._preconditioner_image.as_array(), static_precond.as_array())


def test_parallel_sum_preconditioner(maprl_module):
    """Test that the parallel-sum preconditioner works correctly."""
    initial = DummyImage(np.array([[1.0, 2.0], [3.0, 4.0]]))
    data_fid = DummyFunctional(grad_value=0.5)
    prior = DummyFunctional(grad_value=0.0)

    # Create a preconditioner that represents the parallel sum of D=x and R=1/H
    # For testing: D = x (data preconditioner), R = 0.5 (prior preconditioner)
    # Parallel sum: P = (D * R) / (D + R)
    def parallel_preconditioner(x):
        D = x.as_array()  # Data preconditioner
        R = 0.5  # Prior preconditioner (constant for simplicity)
        P = (D * R) / (D + R)
        return DummyImage(P)

    eps = 1e-8
    relaxation_eta = 0.0  # No relaxation for simpler testing
    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fid,
        prior=prior,
        step_size=0.1,
        eps=eps,
        relaxation_eta=relaxation_eta,
        initial_line_search=False,
        armijo_iterations=0,
        preconditioner=parallel_preconditioner,
        preconditioner_update_initial=1,
    )

    # Perform one update
    maprl.iteration = 1
    initial_values = maprl.x.as_array().copy()
    maprl.update()
    updated_values = maprl.x.as_array()

    # Expected: x = x - grad * P * step
    # P = max((D*R)/(D+R+eps), eps) where D=x, R=0.5
    D = initial_values
    R = 0.5
    P = (D * R) / (D + R + 1e-6)
    P = np.maximum(P, 1e-6)
    expected = initial_values - 0.5 * P * 0.1
    expected = np.maximum(expected, 0.0)

    assert np.allclose(updated_values, expected, rtol=1e-5)


def _require_cil():
    """Helper to skip tests if CIL is not available."""
    try:
        from cil.framework import ImageGeometry
        from cil.optimisation.operators import GradientOperator
        import cil.optimisation.functions as fn
        import cil.optimisation.operators as op
        return ImageGeometry, GradientOperator, fn, op
    except ImportError as e:
        pytest.skip(f"CIL not available: {e}")


def test_diagonal_hessian_monkey_patch():
    """Test diagonal Hessian monkey-patching with actual CIL functions."""
    ImageGeometry, GradientOperator, fn, op = _require_cil()
    from src.krl.operators.directional import DirectionalOperator

    # Create simple geometry and test image
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=1)
    test_image = geometry.allocate('random', seed=42)

    # Create directional operator
    grad = GradientOperator(geometry, method='forward', bnd_cond='Neumann')
    grad_ref = grad.direct(test_image)
    d_op = DirectionalOperator(grad_ref)

    # Create prior structure
    alpha = 0.1
    epsilon = test_image.max() * 1e-3
    composition_op = op.CompositionOperator(d_op, grad)
    prior = alpha * fn.OperatorCompositionFunction(
        fn.SmoothMixedL21Norm(epsilon=epsilon), composition_op
    )

    # Monkey-patch as in run_deconv.py
    smooth_l21_func = prior.function.function
    composition_func = prior.function
    epsilon_smooth = smooth_l21_func.epsilon

    def smooth_l21_diag_hess(y):
        norm_sq_arr = sum(c.as_array()**2 for c in y.containers)
        denom = (norm_sq_arr + epsilon_smooth**2)**(1.5)
        diag_arr = epsilon_smooth**2 / (denom + 1e-12)
        result = y.containers[0].clone()
        result.fill(diag_arr)
        return result

    def composition_diag_hess(x):
        y = composition_func.operator.direct(x)
        base_diag = smooth_l21_func.diagonal_hessian_approx(y)
        return base_diag

    def scaled_diag_hess(x):
        return prior.scalar * prior.function.diagonal_hessian_approx(x)

    smooth_l21_func.diagonal_hessian_approx = smooth_l21_diag_hess
    composition_func.diagonal_hessian_approx = composition_diag_hess
    prior.diagonal_hessian_approx = scaled_diag_hess

    # Test calling the diagonal Hessian
    diag_hess = prior.diagonal_hessian_approx(test_image)

    # Check basic properties
    assert diag_hess.shape == test_image.shape
    assert np.all(np.isfinite(diag_hess.as_array()))
    assert np.all(diag_hess.as_array() > 0)  # Should be positive

    # Check scaling by alpha
    assert diag_hess.max() > 0
    # The diagonal Hessian should be scaled by alpha
    unscaled_diag = prior.function.diagonal_hessian_approx(test_image)
    assert np.allclose(diag_hess.as_array(), alpha * unscaled_diag.as_array())


def test_maprl_armijo_update_periodic(maprl_module):
    """Test that Armijo line search updates periodically after initial iterations."""
    initial = DummyImage(np.ones((5, 5)))
    data_fid = DummyFunctional(grad_value=0.1)
    prior = DummyFunctional(grad_value=0.05)

    armijo_iterations = []

    # Store original _armijo_step to track when it's called
    original_armijo_step = maprl_module.MAPRL._armijo_step

    def tracked_armijo_step(self, suggested_step):
        armijo_iterations.append(self.iteration)
        return original_armijo_step(self, suggested_step)

    maprl_module.MAPRL._armijo_step = tracked_armijo_step

    try:
        maprl = maprl_module.MAPRL(
            initial_estimate=initial,
            data_fidelity=data_fid,
            prior=prior,
            step_size=1.0,
            initial_line_search=False,
            armijo_iterations=25,
            armijo_update_initial=3,
            armijo_update_interval=5,
        )

        # Run 20 iterations
        for i in range(1, 21):
            maprl.iteration = i
            maprl.update()

        # Should update on: 1, 2, 3 (initial), then 5, 10, 15, 20 (periodic)
        expected = [1, 2, 3, 5, 10, 15, 20]
        assert armijo_iterations == expected, f"Expected {expected}, got {armijo_iterations}"

    finally:
        # Restore original method
        maprl_module.MAPRL._armijo_step = original_armijo_step


def test_parallel_sum_with_cil_functions():
    """Test the full parallel-sum preconditioner with CIL functions."""
    ImageGeometry, GradientOperator, fn, op = _require_cil()
    from src.krl.operators.directional import DirectionalOperator

    # Create simple geometry and test image
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=1)
    test_image = geometry.allocate('random', seed=42)
    # Make sure it's positive
    test_image.fill(np.abs(test_image.as_array()) + 0.1)

    # Create directional operator
    grad = GradientOperator(geometry, method='forward', bnd_cond='Neumann')
    grad_ref = grad.direct(test_image)
    d_op = DirectionalOperator(grad_ref)

    # Create prior structure
    alpha = 0.1
    epsilon = test_image.max() * 1e-3
    composition_op = op.CompositionOperator(d_op, grad)
    prior = alpha * fn.OperatorCompositionFunction(
        fn.SmoothMixedL21Norm(epsilon=epsilon), composition_op
    )

    # Monkey-patch as in run_deconv.py
    smooth_l21_func = prior.function.function
    composition_func = prior.function
    epsilon_smooth = smooth_l21_func.epsilon

    def smooth_l21_diag_hess(y):
        norm_sq_arr = sum(c.as_array()**2 for c in y.containers)
        denom = (norm_sq_arr + epsilon_smooth**2)**(1.5)
        diag_arr = epsilon_smooth**2 / (denom + 1e-12)
        result = y.containers[0].clone()
        result.fill(diag_arr)
        return result

    def composition_diag_hess(x):
        y = composition_func.operator.direct(x)
        base_diag = smooth_l21_func.diagonal_hessian_approx(y)
        return base_diag

    def scaled_diag_hess(x):
        return prior.scalar * prior.function.diagonal_hessian_approx(x)

    smooth_l21_func.diagonal_hessian_approx = smooth_l21_diag_hess
    composition_func.diagonal_hessian_approx = composition_diag_hess
    prior.diagonal_hessian_approx = scaled_diag_hess

    # Create parallel-sum preconditioner as in run_deconv.py
    def compute_preconditioner(x):
        eps_safe = 1e-6
        data_precond = x
        prior_hess_diag = prior.diagonal_hessian_approx(x)
        prior_precond = 1.0 / (prior_hess_diag + eps_safe)
        precond = (data_precond * prior_precond) / (data_precond + prior_precond + eps_safe)
        return precond

    # Compute preconditioner
    precond = compute_preconditioner(test_image)

    # Verify properties
    assert precond.shape == test_image.shape
    assert np.all(np.isfinite(precond.as_array()))
    assert np.all(precond.as_array() > 0)

    # Verify parallel-sum property: P <= min(D, R)
    D = test_image.as_array()
    prior_hess = prior.diagonal_hessian_approx(test_image).as_array()
    R = 1.0 / (prior_hess + 1e-6)
    P = precond.as_array()

    # Parallel sum should be less than or equal to both components
    assert np.all(P <= D + 1e-5)
    assert np.all(P <= R + 1e-5)
    assert np.all(P >= 1e-6 - 1e-12)

    # Verify formula with positivity floor: P = max((D*R)/(D+R), eps_safe)
    expected_P = (D * R) / (D + R + 1e-6)
    expected_P = np.maximum(expected_P, 1e-6)
    assert np.allclose(P, expected_P, rtol=1e-4)
