import importlib
import numpy as np
import pytest


class DummyImage:
    __array_priority__ = 1000

    def __init__(self, data):
        self.array = np.array(data, dtype=np.float64)

    def as_array(self):
        return self.array

    @property
    def ndim(self):
        return self.array.ndim

    def clone(self):
        return DummyImage(self.array.copy())

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
        self._voxel_sizes = voxel_sizes

    def voxel_sizes(self):
        return self._voxel_sizes


@pytest.fixture
def gradient_module():
    module = importlib.import_module("src.gradient")
    return importlib.reload(module)


@pytest.fixture
def directional_module():
    module = importlib.import_module("src.directional_operator")
    return importlib.reload(module)


@pytest.fixture
def gaussian_module():
    module = importlib.import_module("src.gaussian_blurring")
    return importlib.reload(module)


@pytest.fixture
def maprl_module():
    module = importlib.import_module("src.map_rl")
    return importlib.reload(module)


def test_gradient_forward_neumann_direct(gradient_module):
    torch = pytest.importorskip("torch")
    data = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    image = DummyImage(data)

    grad_op = gradient_module.Gradient(voxel_sizes=(1.0, 1.0), method="forward")
    result = grad_op.direct(image)
    result_arr = np.asarray(result.as_array(), dtype=np.float32)

    expected_axis0 = np.array([[2.0, 3.0], [0.0, 0.0]], dtype=np.float32)
    expected_axis1 = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    expected = np.stack([expected_axis0, expected_axis1], axis=-1)
    assert result_arr.shape == expected.shape
    assert np.allclose(result_arr, expected)


def test_gradient_adjoint_matches_manual_divergence(gradient_module):
    torch = pytest.importorskip("torch")
    data = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    image = DummyImage(data)
    grad_op = gradient_module.Gradient(voxel_sizes=(1.0, 1.0), method="forward")
    grad_field = grad_op.direct(image).as_array()

    gradient_tensor = torch.tensor(grad_field)
    out = DummyImage(np.zeros_like(data))
    result = grad_op.adjoint(gradient_tensor, out=out)

    components = []
    for axis in range(gradient_tensor.size(-1)):
        components.append(-grad_op.backward_diff(gradient_tensor[..., axis], axis))
    expected = torch.stack(components, dim=-1).sum(dim=-1).cpu().numpy()
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

    maprl = maprl_module.MAPRL(
        initial_estimate=img,
        data_fidelity=ZeroFunctional(),
        prior=ZeroFunctional(),
        step_size=2.0,
        relaxation_eta=0.5,
    )
    maprl.iteration = 3
    assert maprl.step_size() == pytest.approx(2.0 / (1 + 0.5 * 3))


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
