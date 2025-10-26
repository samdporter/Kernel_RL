import ctypes
import importlib
import importlib.util
import numpy as np
import pytest
from pathlib import Path
import sys


class DummyImage:
    __array_priority__ = 1000

    def __init__(self, data, geometry=None):
        self.array = np.array(data, dtype=np.float64)
        self.geometry = geometry

    def as_array(self):
        return self.array

    @property
    def ndim(self):
        return self.array.ndim

    def clone(self):
        return DummyImage(self.array.copy(), geometry=self.geometry)

    def fill(self, values):
        if np.isscalar(values):
            self.array.fill(values)
        else:
            self.array = np.array(values, dtype=np.float64)

    def power(self, exponent):
        return DummyImage(np.power(self.array, exponent))

    def sqrt(self):
        return DummyImage(np.sqrt(self.array))

    def maximum(self, value, out=None):
        value_arr = value.array if isinstance(value, DummyImage) else value
        result = np.maximum(self.array, value_arr)
        if out is None:
            return DummyImage(result)
        out.array = result
        return out

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)

    def _coerce(self, other):
        return other.array if isinstance(other, DummyImage) else other

    def __add__(self, other):
        return DummyImage(self.array + self._coerce(other))

    def __radd__(self, other):
        return DummyImage(self._coerce(other) + self.array)

    def __sub__(self, other):
        return DummyImage(self.array - self._coerce(other))

    def __rsub__(self, other):
        return DummyImage(self._coerce(other) - self.array)

    def __mul__(self, other):
        return DummyImage(self.array * self._coerce(other))

    def __rmul__(self, other):
        return DummyImage(self._coerce(other) * self.array)

    def __truediv__(self, other):
        return DummyImage(self.array / self._coerce(other))

    def __rtruediv__(self, other):
        return DummyImage(self._coerce(other) / self.array)

    def __neg__(self):
        return DummyImage(-self.array)

    def __iadd__(self, other):
        self.array += self._coerce(other)
        return self


class DummyBlock:
    def __init__(self, *containers):
        self.containers = tuple(containers)

    def clone(self):
        return DummyBlock(*[container.clone() for container in self.containers])

    def pnorm(self):
        squares = sum(container.as_array() ** 2 for container in self.containers)
        return DummyImage(np.sqrt(squares))

    def _apply(self, other, op):
        if isinstance(other, DummyBlock):
            return DummyBlock(
                *[op(a, b) for a, b in zip(self.containers, other.containers)]
            )
        return DummyBlock(*[op(a, other) for a in self.containers])

    def __add__(self, other):
        return self._apply(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply(other, lambda a, b: a - b)

    def __rsub__(self, other):
        if isinstance(other, DummyBlock):
            return other.__sub__(self)
        return DummyBlock(*[other - a for a in self.containers])

    def __mul__(self, other):
        return self._apply(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply(other, lambda a, b: a / b)

    def __iter__(self):
        return iter(self.containers)


class DummyGeometry:
    def __init__(self, shape, voxel_sizes=(1.0, 1.0, 1.0)):
        self.shape = shape
        self.voxel_size_z, self.voxel_size_y, self.voxel_size_x = voxel_sizes


def _require_cil_gradient():
    def _attempt():
        from cil.framework import ImageGeometry  # type: ignore
        from cil.optimisation.operators import GradientOperator  # type: ignore
        return GradientOperator, ImageGeometry

    try:
        return _attempt()
    except Exception as first_error:
        root = None
        try:
            spec = importlib.util.find_spec("cil")
            if spec and spec.origin:
                root = Path(spec.origin).resolve()
        except ValueError:
            module = sys.modules.get("cil")
            module_file = getattr(module, "__file__", None) if module else None
            module_paths = list(getattr(module, "__path__", [])) if module else []
            if module_file:
                root = Path(module_file).resolve()
            elif module_paths:
                root = Path(module_paths[0]).resolve()

        if root is None:
            for entry in map(Path, sys.path):
                candidate = entry / "cil" / "__init__.py"
                if candidate.exists():
                    root = candidate.resolve()
                    break

        if root is not None:
            candidates = []
            for parent in root.parents:
                candidates.append(parent / "lib" / "libcilacc.so")
                candidates.append(parent / "cil" / "lib" / "libcilacc.so")
            for candidate in candidates:
                if candidate.exists():
                    try:
                        ctypes.cdll.LoadLibrary(str(candidate))
                        sys.modules.pop("cil.framework", None)
                        sys.modules.pop("cil.optimisation.operators", None)
                        sys.modules.pop("cil.optimisation", None)
                        if (module := sys.modules.get("cil")) is not None and getattr(module, "__spec__", None) is None:
                            sys.modules.pop("cil", None)
                        return _attempt()
                    except OSError:
                        continue

        pytest.skip(f"CIL GradientOperator unavailable: {first_error}")


def _forward_neumann_diff(data, axis):
    diff = np.zeros_like(data, dtype=data.dtype)
    axis_len = data.shape[axis]
    if axis_len > 1:
        slicer_curr = [slice(None)] * data.ndim
        slicer_next = [slice(None)] * data.ndim
        slicer_curr[axis] = slice(0, axis_len - 1)
        slicer_next[axis] = slice(1, axis_len)
        diff[tuple(slicer_curr)] = (
            data[tuple(slicer_next)] - data[tuple(slicer_curr)]
        )
    return diff


def _backward_neumann_diff(data, axis):
    diff = np.zeros_like(data, dtype=data.dtype)
    axis_len = data.shape[axis]
    slicer_first = [slice(None)] * data.ndim
    slicer_first[axis] = 0
    diff[tuple(slicer_first)] = data[tuple(slicer_first)]
    if axis_len > 1:
        slicer_curr = [slice(None)] * data.ndim
        slicer_prev = [slice(None)] * data.ndim
        slicer_curr[axis] = slice(1, axis_len)
        slicer_prev[axis] = slice(0, axis_len - 1)
        diff[tuple(slicer_curr)] = (
            data[tuple(slicer_curr)] - data[tuple(slicer_prev)]
        )
        slicer_last = [slice(None)] * data.ndim
        slicer_last[axis] = axis_len - 1
        slicer_penultimate = [slice(None)] * data.ndim
        slicer_penultimate[axis] = axis_len - 2
        diff[tuple(slicer_last)] = -data[tuple(slicer_penultimate)]
    return diff


@pytest.fixture
def directional_module():
    module = importlib.import_module("src.krl.operators.directional")
    return importlib.reload(module)


@pytest.fixture
def gaussian_module():
    module = importlib.import_module("src.krl.operators.blurring")
    return importlib.reload(module)


@pytest.fixture
def maprl_module():
    module = importlib.import_module("src.krl.algorithms.maprl")
    return importlib.reload(module)


def test_gradient_forward_neumann_direct():
    GradientOperator, ImageGeometry = _require_cil_gradient()
    geometry = ImageGeometry(voxel_num_y=2, voxel_num_x=2)
    image = geometry.allocate(None)
    data = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    image.fill(data)

    grad_op = GradientOperator(geometry, method="forward", bnd_cond="Neumann", backend="numpy")
    result = grad_op.direct(image)

    components = [
        result.get_item(i).as_array().astype(np.float32)
        for i in range(len(result.containers))
    ]
    result_arr = np.stack(components, axis=-1)

    expected_axis0 = _forward_neumann_diff(data, axis=0)
    expected_axis1 = _forward_neumann_diff(data, axis=1)
    expected = np.stack([expected_axis0, expected_axis1], axis=-1)
    assert result_arr.shape == expected.shape
    assert np.allclose(result_arr, expected)


def test_gradient_adjoint_matches_manual_divergence():
    GradientOperator, ImageGeometry = _require_cil_gradient()
    geometry = ImageGeometry(voxel_num_y=2, voxel_num_x=2)
    image = geometry.allocate(None)
    data = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    image.fill(data)

    grad_op = GradientOperator(geometry, method="forward", bnd_cond="Neumann", backend="numpy")
    grad_field = grad_op.direct(image)
    result = grad_op.adjoint(grad_field)

    gradient_tensor = np.stack(
        [grad_field.get_item(i).as_array().astype(np.float32) for i in range(len(grad_field.containers))],
        axis=-1,
    )
    components = []
    for axis in range(gradient_tensor.shape[-1]):
        components.append(_backward_neumann_diff(gradient_tensor[..., axis], axis))
    expected = -np.sum(components, axis=0)
    assert np.allclose(result.as_array(), expected)


def test_directional_operator_direct_matches_formula(directional_module):
    anat = DummyBlock(
        DummyImage([[1.0, 2.0], [3.0, 4.0]]),
        DummyImage([[0.5, 1.5], [2.0, 2.5]]),
    )
    operator = directional_module.DirectionalOperator(anat, gamma=0.3, eta=0.1)

    xi = operator.xi
    test_block = DummyBlock(
        DummyImage([[2.0, 1.0], [0.0, -1.0]]),
        DummyImage([[1.0, -0.5], [0.5, 0.0]]),
    )
    dot_val = operator.dot(xi, test_block)
    expected = test_block - operator.gamma * xi * dot_val
    result = operator.direct(test_block)

    for res_component, exp_component in zip(result.containers, expected.containers):
        assert np.allclose(res_component.as_array(), exp_component.as_array())


def test_directional_operator_adjoint_is_self_adjoint(directional_module):
    anat = DummyBlock(
        DummyImage([[1.0, 1.0], [1.0, 1.0]]),
        DummyImage([[0.5, 0.5], [0.5, 0.5]]),
    )
    operator = directional_module.DirectionalOperator(anat, gamma=0.2, eta=0.05)
    block = DummyBlock(
        DummyImage([[1.0, 0.0], [0.0, 1.0]]),
        DummyImage([[0.5, -0.5], [0.0, 0.0]]),
    )

    direct_res = operator.direct(block)
    adjoint_res = operator.adjoint(block)

    for direct_comp, adjoint_comp in zip(direct_res.containers, adjoint_res.containers):
        assert np.allclose(
            direct_comp.as_array(), adjoint_comp.as_array()
        ), "Directional operator should be self adjoint."


def test_gaussian_blur_numba_backend_matches_reference(gaussian_module):
    geometry = DummyGeometry((3, 3, 3))
    op = gaussian_module.GaussianBlurringOperator(
        sigma=(1.0, 1.0, 1.0), domain_geometry=geometry, backend="numba"
    )
    spike = np.zeros((3, 3, 3), dtype=np.float64)
    spike[1, 1, 1] = 1.0
    image = DummyImage(spike)

    blurred = op.direct(image)
    expected = gaussian_module._numba_convolve_3d(spike, op.psf)
    assert np.allclose(blurred.as_array(), expected)


def test_gaussian_blur_adjoint_equals_direct_for_symmetric_psf(gaussian_module):
    geometry = DummyGeometry((3, 3, 3))
    op = gaussian_module.GaussianBlurringOperator(
        sigma=(0.8, 0.8, 0.8), domain_geometry=geometry, backend="numba"
    )
    data = np.random.default_rng(0).normal(size=(3, 3, 3))
    image = DummyImage(data)

    out_direct = op.direct(image).as_array()
    adjoint = DummyImage(np.zeros_like(data))
    out_adjoint = op.adjoint(image, out=adjoint).as_array()

    assert np.allclose(out_direct, out_adjoint)


def test_maprl_step_size_schedule(maprl_module):
    img = DummyImage(np.ones((2, 2)))

    class ZeroFunctional:
        def gradient(self, x):
            return DummyImage(np.zeros_like(x.as_array()))

        def __call__(self, x):
            return 0.0

    # Test with armijo_iterations=0 to disable Armijo search during iterations
    maprl = maprl_module.MAPRL(
        initial_estimate=img,
        data_fidelity=ZeroFunctional(),
        prior=ZeroFunctional(),
        step_size=2.0,
        relaxation_eta=0.5,
        initial_line_search=False,
        armijo_iterations=0,
    )
    maprl.iteration = 3
    assert maprl.step_size() == pytest.approx(2.0 / (1 + 0.5 * 3))

    # Test within Armijo iterations window
    maprl2 = maprl_module.MAPRL(
        initial_estimate=img,
        data_fidelity=ZeroFunctional(),
        prior=ZeroFunctional(),
        step_size=2.0,
        relaxation_eta=0.5,
        initial_line_search=False,
        armijo_iterations=5,
    )
    maprl2.iteration = 3
    # Within armijo_iterations window, it should use _current_step_size
    assert maprl2.step_size() == pytest.approx(2.0)

    # After armijo_iterations window, it should use relaxation formula
    maprl2.iteration = 6
    assert maprl2.step_size() == pytest.approx(2.0 / (1 + 0.5 * 6))


def test_maprl_update_applies_scaled_gradient_and_projection(maprl_module):
    target = DummyImage(np.full((2, 2), 0.25))

    class QuadraticFunctional:
        def gradient(self, x):
            return x - target

        def __call__(self, x):
            diff = x.as_array() - target.as_array()
            return 0.5 * float(np.sum(diff**2))

    zero_prior = QuadraticFunctional()  # behaves like zero gradient when using target
    zero_prior.gradient = lambda x: DummyImage(np.zeros_like(x.as_array()))

    initial = DummyImage(np.array([[1.0, -0.5], [0.25, -1.0]]))
    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=QuadraticFunctional(),
        prior=zero_prior,
        step_size=0.5,
        relaxation_eta=0.0,
        eps=0.0,
        initial_line_search=False,
    )

    maprl.update()
    updated = maprl.x.as_array()
    grad = initial.as_array() - target.as_array()
    expected = initial.as_array() - (initial.as_array()) * grad * 0.5
    expected = np.maximum(expected, 0.0)
    assert np.allclose(updated, expected)

    maprl.update_objective()
    assert len(maprl.loss) == 1
    assert maprl.loss[0] == pytest.approx(
        maprl.data_fidelity(maprl.x) + maprl.prior(maprl.x)
    )


def test_maprl_initial_line_search_reduces_objective(maprl_module):
    target = DummyImage(np.full((2, 2), 2.0))

    class QuadraticFunctional:
        def gradient(self, x):
            return x - target

        def __call__(self, x):
            diff = x.as_array() - target.as_array()
            return 0.5 * float(np.sum(diff**2))

    zero_prior = QuadraticFunctional()
    zero_prior.gradient = lambda x: DummyImage(np.zeros_like(x.as_array()))

    initial = DummyImage(np.ones((2, 2)))
    step_guess = 10.0
    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=QuadraticFunctional(),
        prior=zero_prior,
        step_size=step_guess,
        relaxation_eta=0.0,
        eps=0.0,
        initial_line_search=True,
        armijo_beta=0.5,
        armijo_sigma=1e-4,
    )

    assert maprl.initial_step_size <= step_guess

    grad = maprl.data_fidelity.gradient(initial) + maprl.prior.gradient(initial)
    direction = (initial + maprl.eps) * grad
    slope = maprl_module.MAPRL._inner_product(direction, grad)
    candidate = initial - direction * maprl.initial_step_size
    candidate.maximum(0, out=candidate)

    initial_loss = maprl.data_fidelity(initial) + maprl.prior(initial)
    candidate_loss = maprl.data_fidelity(candidate) + maprl.prior(candidate)

    assert candidate_loss <= initial_loss - maprl.armijo_sigma * maprl.initial_step_size * slope + 1e-8


def test_maprl_armijo_iterations_perform_line_search(maprl_module):
    """Test that Armijo line search is performed for the first armijo_iterations."""
    target = DummyImage(np.full((2, 2), 2.0))

    class QuadraticFunctional:
        def __init__(self):
            self.gradient_calls = 0
            self.eval_calls = 0

        def gradient(self, x):
            self.gradient_calls += 1
            return x - target

        def __call__(self, x):
            self.eval_calls += 1
            diff = x.as_array() - target.as_array()
            return 0.5 * float(np.sum(diff**2))

    data_fidelity = QuadraticFunctional()
    zero_prior = QuadraticFunctional()
    zero_prior.gradient = lambda x: DummyImage(np.zeros_like(x.as_array()))
    zero_prior.__call__ = lambda x: 0.0

    initial = DummyImage(np.ones((2, 2)))

    # Test with armijo_iterations=3 to verify line search happens in first 3 iterations
    maprl = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fidelity,
        prior=zero_prior,
        step_size=10.0,  # Intentionally large to trigger backtracking
        relaxation_eta=0.1,
        eps=0.0,
        initial_line_search=True,
        armijo_beta=0.5,
        armijo_sigma=1e-4,
        armijo_max_iter=20,
        armijo_iterations=3,  # Only first 3 iterations use Armijo
    )

    initial_step = maprl.initial_step_size
    losses = []
    step_sizes = []

    # Run 6 iterations
    for i in range(1, 7):
        maprl.iteration = i
        eval_calls_before = data_fidelity.eval_calls

        maprl.update()
        step_after = maprl.step_size()

        maprl.update_objective()
        losses.append(maprl.loss[-1])
        step_sizes.append(step_after)

        eval_calls_after = data_fidelity.eval_calls

        if i <= 3:
            # During Armijo iterations, we expect multiple evaluations (line search)
            # At minimum: reference loss + at least one candidate
            assert eval_calls_after - eval_calls_before >= 2, \
                f"Iteration {i}: Expected multiple loss evaluations during Armijo search, got {eval_calls_after - eval_calls_before}"
            # Step size should be found by Armijo (likely smaller than initial_step)
            # and should stay constant within the Armijo window
            assert step_after == maprl._current_step_size
        else:
            # After Armijo iterations, only one evaluation for update_objective
            assert eval_calls_after - eval_calls_before == 1, \
                f"Iteration {i}: Expected single evaluation, got {eval_calls_after - eval_calls_before}"
            # Step size should follow relaxation schedule
            expected_step = initial_step / (1 + 0.1 * i)
            assert step_after == pytest.approx(expected_step)

    # Verify that losses are decreasing (convergence)
    for i in range(1, len(losses)):
        assert losses[i] <= losses[i-1] + 1e-6, \
            f"Loss increased from iteration {i} to {i+1}: {losses[i-1]:.6f} -> {losses[i]:.6f}"

    # Verify step sizes behavior
    # First 3 iterations use Armijo-found steps, which can vary per iteration
    # but should be less than or equal to initial_step (due to backtracking)
    for i in range(3):
        assert step_sizes[i] <= initial_step, \
            f"Armijo iteration {i+1}: step {step_sizes[i]:.6f} should be <= initial {initial_step:.6f}"

    # Iterations 4-6 should follow relaxation schedule
    for i in range(3, 6):
        expected_step = initial_step / (1 + 0.1 * (i + 1))
        assert step_sizes[i] == pytest.approx(expected_step), \
            f"Iteration {i+1}: step {step_sizes[i]:.6f} should match relaxation schedule {expected_step:.6f}"
