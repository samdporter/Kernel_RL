"""L-BFGS-B optimiser tailored for CIL ImageData objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from scipy.optimize import Bounds, minimize

try:
    from cil.framework import ImageData
except ImportError:  # pragma: no cover
    from typing import Any

    ImageData = Any  # type: ignore


@dataclass
class LBFGSBOptions:
    """Configuration container for the L-BFGS-B optimiser."""

    max_linesearch: int = 20
    ftol: float = 1e-6
    gtol: float = 1e-6
    enforce_non_negativity: bool = True


class _ImageArrayAdapter:
    """Utility class to convert between ImageData objects and flat arrays."""

    def __init__(self, template: ImageData):
        self.shape = template.shape
        self.size = int(np.prod(self.shape))
        self._working_image = template.geometry.allocate(value=0)

    def array_to_image(self, array: np.ndarray) -> ImageData:
        self._working_image.fill(array.reshape(self.shape))
        return self._working_image

    @staticmethod
    def image_to_array(image: ImageData) -> np.ndarray:
        return np.asarray(image.as_array(), dtype=np.float64).ravel(order="C")


class LBFGSBOptimizer:
    """
    Lightweight wrapper around SciPy's L-BFGS-B for use inside CIL pipelines.

    Parameters
    ----------
    initial_estimate : ImageData
        Starting image for the optimisation.
    data_fidelity : Function
        CIL function that evaluates the data term and provides gradients.
    prior : Function, optional
        Additional regularisation term (e.g. DTV). Set to ``None`` for
        unregularised optimisation.
    options : LBFGSBOptions, optional
        Numerical options for the optimiser.
    """

    def __init__(
        self,
        initial_estimate: ImageData,
        data_fidelity,
        prior=None,
        options: Optional[LBFGSBOptions] = None,
        **kwargs,
    ) -> None:
        self.initial_estimate = initial_estimate.clone()
        self._solution = initial_estimate.clone()
        self._solution.fill(self.initial_estimate.as_array())
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.options = options or LBFGSBOptions()
        self.result = None

        self._adapter = _ImageArrayAdapter(self._solution)
        self._objective: List[float] = []
        self.loss = self._objective
        self._last_fun_value: Optional[float] = None
        self.iteration = 0
        self.configured = True

    @property
    def solution(self) -> ImageData:
        return self._solution

    @solution.setter
    def solution(self, value: ImageData) -> None:
        self._solution = value

    @property
    def objective(self) -> List[float]:
        return self._objective

    def run(self, iterations: int, callbacks: Optional[Iterable] = None, verbose: int = 0):
        """Execute the optimisation for a maximum number of iterations."""
        callbacks = list(callbacks) if callbacks else []
        self.iteration = 0
        self._objective = []
        self.loss = self._objective
        self._last_fun_value = None

        # Reset solution to the starting point
        self.solution.fill(self.initial_estimate.as_array())
        x0 = _ImageArrayAdapter.image_to_array(self.solution)

        initial_obj = self._objective_from_array(x0)
        self._objective.append(initial_obj)

        bounds = None
        if self.options.enforce_non_negativity:
            bounds = Bounds(lb=0, ub=np.inf)

        def fun(x):
            value = self._objective_from_array(x)
            self._last_fun_value = value
            return value

        def jac(x):
            return self._gradient_from_array(x)

        def scipy_callback(xk):
            self.iteration += 1
            self.solution.fill(xk.reshape(self.solution.shape))
            if self.options.enforce_non_negativity:
                with np.errstate(invalid="ignore"):
                    self.solution.maximum(0, out=self.solution)
            current_value = self._last_fun_value
            if current_value is None:
                current_value = self._objective_from_array(xk)
            self._objective.append(current_value)
            for cb in callbacks:
                cb(self)
            return False

        options = {
            "maxiter": iterations,
            "maxls": self.options.max_linesearch,
            "ftol": self.options.ftol,
            "gtol": self.options.gtol,
            "disp": verbose > 0,
        }

        self.result = minimize(
            fun=fun,
            jac=jac,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            callback=scipy_callback,
            options=options,
        )

        final_x = self.result.x
        self.solution.fill(final_x.reshape(self.solution.shape))
        if self.options.enforce_non_negativity:
            with np.errstate(invalid="ignore"):
                self.solution.maximum(0, out=self.solution)

        final_value = self._objective_from_array(final_x)
        if not self._objective or not np.isclose(self._objective[-1], final_value):
            self._objective.append(final_value)

        return self

    def _objective_from_array(self, array: np.ndarray) -> float:
        image = self._adapter.array_to_image(array)
        value = float(self.data_fidelity(image))
        if self.prior is not None:
            value += float(self.prior(image))
        return value

    def _gradient_from_array(self, array: np.ndarray) -> np.ndarray:
        image = self._adapter.array_to_image(array)
        grad = self.data_fidelity.gradient(image)
        grad_arr = np.asarray(grad.as_array(), dtype=np.float64)
        if self.prior is not None:
            grad_prior = self.prior.gradient(image)
            grad_arr = grad_arr + np.asarray(grad_prior.as_array(), dtype=np.float64)
        return grad_arr.ravel(order="C")
