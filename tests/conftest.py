import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _install_numba_stub() -> None:
    if "numba" in sys.modules:
        return

    numba = types.ModuleType("numba")

    def _jit(*args, **kwargs):
        def decorator(func):
            return func

        # Support usage both with and without arguments
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return decorator(args[0])
        return decorator

    numba.jit = _jit  # type: ignore[attr-defined]
    numba.njit = _jit  # type: ignore[attr-defined]
    numba.prange = range  # type: ignore[attr-defined]
    sys.modules["numba"] = numba


def _install_cil_stubs() -> None:
    if "cil" in sys.modules:
        return

    cil = types.ModuleType("cil")
    cil.__path__ = []  # type: ignore[attr-defined]
    sys.modules["cil"] = cil

    optimisation = types.ModuleType("cil.optimisation")
    optimisation.__path__ = []  # type: ignore[attr-defined]
    sys.modules["cil.optimisation"] = optimisation
    cil.optimisation = optimisation  # type: ignore[attr-defined]

    operators = types.ModuleType("cil.optimisation.operators")
    sys.modules["cil.optimisation.operators"] = operators

    class LinearOperator:  # Minimal stub to satisfy imports
        def __init__(self, domain_geometry=None, range_geometry=None, **kwargs):
            self.domain_geometry = domain_geometry
            self.range_geometry = range_geometry

    operators.LinearOperator = LinearOperator  # type: ignore[attr-defined]
    optimisation.operators = operators  # type: ignore[attr-defined]

    algorithms = types.ModuleType("cil.optimisation.algorithms")
    sys.modules["cil.optimisation.algorithms"] = algorithms

    class Algorithm:  # Minimal stub that mimics the CIL API portions we need
        def __init__(self, **kwargs):
            self.iteration = 0
            self.loss = []
            self.solution = None

        def run(self, iterations=1, callbacks=None, verbose=0):
            callbacks = callbacks or []
            for _ in range(iterations):
                self.update()
                for callback in callbacks:
                    callback(self)
                self.iteration += 1
            self.solution = getattr(self, "x", None)

        def update(self):  # pragma: no cover - to be implemented by subclasses
            raise NotImplementedError

    algorithms.Algorithm = Algorithm  # type: ignore[attr-defined]
    optimisation.algorithms = algorithms  # type: ignore[attr-defined]

    framework = types.ModuleType("cil.framework")
    sys.modules["cil.framework"] = framework

    class BlockGeometry:
        def __init__(self, *args, **kwargs):
            self.blocks = args

    framework.BlockGeometry = BlockGeometry  # type: ignore[attr-defined]
    cil.framework = framework  # type: ignore[attr-defined]


_install_numba_stub()
_install_cil_stubs()


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return

    import numpy as np

    torch = types.ModuleType("torch")
    torch.__dict__["__all__"] = []  # type: ignore[assignment]

    class TorchTensor:
        __array_priority__ = 1000

        def __init__(self, array, *, copy=True):
            self.array = np.array(array, dtype=np.float64, copy=copy)

        def clone(self):
            return TorchTensor(self.array.copy())

        def numpy(self):
            return self.array.copy()

        def cpu(self):
            return self

        def cuda(self):
            return self

        def select(self, dim, index):
            slices = [slice(None)] * self.array.ndim
            slices[dim] = index
            return TorchTensor(self.array[tuple(slices)], copy=False)

        def unsqueeze(self, dim):
            return TorchTensor(np.expand_dims(self.array, axis=dim), copy=False)

        def zero_(self):
            self.array[...] = 0
            return self

        def copy_(self, other):
            source = other.array if isinstance(other, TorchTensor) else np.asarray(other)
            self.array[...] = source
            return self

        def flip(self, dim):
            return TorchTensor(np.flip(self.array, axis=dim))

        def size(self, dim=None):
            return self.array.shape if dim is None else self.array.shape[dim]

        def sum(self, dim=None):
            return TorchTensor(np.sum(self.array, axis=dim))

        def __array__(self, dtype=None):
            return np.asarray(self.array, dtype=dtype)

        def __mul__(self, other):
            return TorchTensor(self.array * _coerce_array(other))

        __rmul__ = __mul__

        def __truediv__(self, other):
            return TorchTensor(self.array / _coerce_array(other))

        def __itruediv__(self, other):
            self.array /= _coerce_array(other)
            return self

        def __add__(self, other):
            return TorchTensor(self.array + _coerce_array(other))

        __radd__ = __add__

        def __sub__(self, other):
            return TorchTensor(self.array - _coerce_array(other))

        def __rsub__(self, other):
            return TorchTensor(_coerce_array(other) - self.array)

        def __neg__(self):
            return TorchTensor(-self.array)

        def __getitem__(self, item):
            return TorchTensor(self.array[item], copy=False)

    def _coerce_array(value):
        if isinstance(value, TorchTensor):
            return value.array
        return np.asarray(value, dtype=np.float64)

    def tensor(data, device=None, dtype=None):
        return TorchTensor(data)

    def stack(tensors, dim=0):
        arrays = [_coerce_array(t) for t in tensors]
        return TorchTensor(np.stack(arrays, axis=dim))

    def diff(data, n=1, dim=0, append=None):
        arr = _coerce_array(data)
        if n != 1:
            raise NotImplementedError("Stub torch.diff only supports n=1")
        diff_arr = np.diff(arr, axis=dim)
        if append is not None:
            append_arr = _coerce_array(append)
            append_shape = list(arr.shape)
            append_shape[dim] = 1
            append_arr = np.reshape(append_arr, append_shape)
            diff_arr = np.concatenate((diff_arr, append_arr), axis=dim)
        return TorchTensor(diff_arr)

    def flip(data, dims):
        arr = _coerce_array(data)
        if isinstance(dims, int):
            dims = (dims,)
        flipped = arr
        for axis in dims:
            flipped = np.flip(flipped, axis=axis)
        return TorchTensor(flipped)

    class _CUDA:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    torch.Tensor = TorchTensor  # type: ignore[attr-defined]
    torch.tensor = tensor  # type: ignore[attr-defined]
    torch.stack = stack  # type: ignore[attr-defined]
    torch.diff = diff  # type: ignore[attr-defined]
    torch.flip = flip  # type: ignore[attr-defined]
    torch.device = lambda *args, **kwargs: "cpu"  # type: ignore[assignment]
    torch.cuda = _CUDA()  # type: ignore[attr-defined]

    sys.modules["torch"] = torch


def _install_sirf_stub() -> None:
    if "sirf" in sys.modules:
        return

    sirf = types.ModuleType("sirf")
    sys.modules["sirf"] = sirf

    stir = types.ModuleType("sirf.STIR")

    class TruncateToCylinderProcessor:
        def __call__(self, image):
            return image

    stir.TruncateToCylinderProcessor = TruncateToCylinderProcessor  # type: ignore[attr-defined]
    sys.modules["sirf.STIR"] = stir
    sirf.STIR = stir  # type: ignore[attr-defined]


_install_torch_stub()
_install_sirf_stub()
