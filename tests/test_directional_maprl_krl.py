import importlib
import sys
import types

import numpy as np
import pytest


def _ensure_cil_stubs():
    """Provide lightweight stand-ins for CIL modules when unavailable."""
    try:
        import cil.framework  # type: ignore
        import cil.optimisation.operators  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    cil_pkg = sys.modules.get("cil")
    if cil_pkg is None:
        cil_pkg = types.ModuleType("cil")
        cil_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["cil"] = cil_pkg

    framework = types.ModuleType("cil.framework")

    class DummyBlockGeometry:
        def __init__(self, *geometries):
            self.components = geometries

    framework.BlockGeometry = DummyBlockGeometry  # type: ignore[attr-defined]
    sys.modules["cil.framework"] = framework
    setattr(cil_pkg, "framework", framework)

    optimisation_pkg = sys.modules.get("cil.optimisation")
    if optimisation_pkg is None:
        optimisation_pkg = types.ModuleType("cil.optimisation")
        optimisation_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["cil.optimisation"] = optimisation_pkg
    setattr(cil_pkg, "optimisation", optimisation_pkg)

    operators_module = types.ModuleType("cil.optimisation.operators")

    class DummyLinearOperator:
        def __init__(self, domain_geometry=None, range_geometry=None, **_):
            self.domain_geometry = domain_geometry
            self.range_geometry = range_geometry

    operators_module.LinearOperator = DummyLinearOperator  # type: ignore[attr-defined]
    sys.modules["cil.optimisation.operators"] = operators_module
    setattr(optimisation_pkg, "operators", operators_module)


class DummyImage:
    __array_priority__ = 1000

    def __init__(self, data, geometry=None):
        self.array = np.array(data, dtype=np.float64)
        self.geometry = geometry if geometry is not None else ("geom", self.array.shape)

    def as_array(self):
        return self.array

    def clone(self):
        return DummyImage(self.array.copy(), self.geometry)

    def fill(self, values):
        if isinstance(values, DummyImage):
            self.array[...] = values.array
        else:
            self.array[...] = values

    def power(self, exponent):
        return DummyImage(np.power(self.array, exponent), self.geometry)

    def sqrt(self):
        return DummyImage(np.sqrt(self.array), self.geometry)

    def maximum(self, value, out=None):
        value_arr = value.array if isinstance(value, DummyImage) else value
        result = np.maximum(self.array, value_arr)
        if out is None:
            return DummyImage(result, self.geometry)
        out.array = result
        return out

    def _coerce(self, other):
        return other.array if isinstance(other, DummyImage) else other

    def __add__(self, other):
        return DummyImage(self.array + self._coerce(other), self.geometry)

    def __radd__(self, other):
        return DummyImage(self._coerce(other) + self.array, self.geometry)

    def __sub__(self, other):
        return DummyImage(self.array - self._coerce(other), self.geometry)

    def __rsub__(self, other):
        return DummyImage(self._coerce(other) - self.array, self.geometry)

    def __mul__(self, other):
        return DummyImage(self.array * self._coerce(other), self.geometry)

    def __rmul__(self, other):
        return DummyImage(self._coerce(other) * self.array, self.geometry)

    def __truediv__(self, other):
        return DummyImage(self.array / self._coerce(other), self.geometry)

    def __rtruediv__(self, other):
        return DummyImage(self._coerce(other) / self.array, self.geometry)

    def __neg__(self):
        return DummyImage(-self.array, self.geometry)

    def __iadd__(self, other):
        self.array += self._coerce(other)
        return self

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)


class DummyBlock:
    def __init__(self, *containers):
        self.containers = tuple(containers)
        self.geometry = tuple(container.geometry for container in self.containers)

    def clone(self):
        return DummyBlock(*[container.clone() for container in self.containers])

    def fill(self, other):
        if isinstance(other, DummyBlock):
            for dst, src in zip(self.containers, other.containers):
                dst.fill(src)
        else:
            for dst in self.containers:
                dst.fill(other)
        return self

    def pnorm(self):
        squares = sum(container.as_array() ** 2 for container in self.containers)
        return DummyImage(np.sqrt(squares))

    def _apply(self, other, op):
        if isinstance(other, DummyBlock):
            return DummyBlock(*[op(a, b) for a, b in zip(self.containers, other.containers)])
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


@pytest.fixture(scope="module")
def directional_module():
    _ensure_cil_stubs()
    module = importlib.import_module("krl.operators.directional")
    return importlib.reload(module)


@pytest.fixture(scope="module")
def maprl_module():
    module = importlib.import_module("krl.algorithms.maprl")
    return importlib.reload(module)


def test_directional_operator_xi_normalization(directional_module):
    anatomical = DummyBlock(
        DummyImage([[3.0, 4.0], [0.0, 5.0]], geometry="g0"),
        DummyImage([[4.0, 0.0], [0.0, 0.0]], geometry="g1"),
    )
    operator = directional_module.DirectionalOperator(anatomical, gamma=0.5, eta=0.2)

    sum_squares = sum(container.as_array() ** 2 for container in anatomical.containers)
    expected_norm = np.sqrt(sum_squares + 0.2**2)

    for xi_component, anatomical_component in zip(operator.xi.containers, anatomical.containers):
        assert np.allclose(
            xi_component.as_array(),
            anatomical_component.as_array() / expected_norm,
        )


def test_directional_operator_direct_writes_to_out(directional_module):
    anatomical = DummyBlock(
        DummyImage([[1.0, 2.0]], geometry="g0"),
        DummyImage([[0.5, 0.5]], geometry="g1"),
    )
    operator = directional_module.DirectionalOperator(anatomical, gamma=0.3, eta=0.1)

    test_block = DummyBlock(
        DummyImage([[2.0, -1.0]], geometry="g0"),
        DummyImage([[0.0, 1.5]], geometry="g1"),
    )
    out = test_block.clone()
    operator.direct(test_block, out=out)

    dot_val = operator.dot(operator.xi, test_block)
    expected = test_block - operator.gamma * operator.xi * dot_val

    for out_comp, exp_comp in zip(out.containers, expected.containers):
        assert np.allclose(out_comp.as_array(), exp_comp.as_array())


def test_directional_operator_dot_resets_accumulator(directional_module):
    anatomical = DummyBlock(
        DummyImage([[1.0, 1.0]], geometry="g0"),
        DummyImage([[1.0, 1.0]], geometry="g1"),
    )
    operator = directional_module.DirectionalOperator(anatomical, gamma=1.0, eta=0.01)

    block_a = DummyBlock(
        DummyImage([[2.0, 3.0]], geometry="g0"),
        DummyImage([[4.0, 5.0]], geometry="g1"),
    )
    block_b = DummyBlock(
        DummyImage([[0.5, 1.0]], geometry="g0"),
        DummyImage([[1.5, 2.0]], geometry="g1"),
    )

    dot_a = operator.dot(block_a, block_a).as_array()
    expected_a = (
        block_a.containers[0].as_array() * block_a.containers[0].as_array()
        + block_a.containers[1].as_array() * block_a.containers[1].as_array()
    )
    assert np.allclose(dot_a, expected_a)

    dot_b = operator.dot(block_a, block_b).as_array()
    expected_b = (
        block_a.containers[0].as_array() * block_b.containers[0].as_array()
        + block_a.containers[1].as_array() * block_b.containers[1].as_array()
    )
    assert np.allclose(dot_b, expected_b)


def test_maprl_update_combines_data_and_prior_gradients(maprl_module):
    initial = DummyImage([[1.0, 2.0]])

    class DataFunctional:
        def gradient(self, x):
            return DummyImage([[0.2, -0.1]])

        def __call__(self, x):
            return float(np.sum(x.as_array()))

    class PriorFunctional:
        def gradient(self, x):
            return DummyImage([[-0.3, 0.5]])

        def __call__(self, x):
            return float(np.sum(x.as_array() ** 2))

    algorithm = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=DataFunctional(),
        prior=PriorFunctional(),
        step_size=0.4,
        relaxation_eta=0.0,
        eps=0.2,
    )
    algorithm.iteration = 1

    algorithm.update()
    combined_grad = np.array([[0.2, -0.1]]) + np.array([[-0.3, 0.5]])
    expected = initial.as_array() - (initial.as_array() + 0.2) * combined_grad * 0.4
    expected = np.maximum(expected, 0.0)

    assert np.allclose(algorithm.x.as_array(), expected)
    assert np.allclose(initial.as_array(), np.array([[1.0, 2.0]]))


def test_maprl_update_projects_negative_values(maprl_module):
    initial = DummyImage([[0.1, 0.1]])

    class PositiveGradient:
        def gradient(self, x):
            return DummyImage([[5.0, 5.0]])

        def __call__(self, x):
            return float(np.sum(x.as_array()))

    zero_prior = PositiveGradient()
    zero_prior.gradient = lambda x: DummyImage([[0.0, 0.0]])  # type: ignore[attr-defined]

    algorithm = maprl_module.MAPRL(
        initial_estimate=initial,
        data_fidelity=PositiveGradient(),
        prior=zero_prior,
        step_size=1.0,
        relaxation_eta=0.0,
        eps=0.0,
    )

    algorithm.update()
    assert np.allclose(algorithm.x.as_array(), np.zeros_like(initial.as_array()))
