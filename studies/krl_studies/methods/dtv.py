"""MAP-RL with directional TV, following examples/pipelines/run_deconv.py."""

from __future__ import annotations

from collections.abc import Iterator

import cil.optimisation.functions as fn
import cil.optimisation.operators as op
import numpy as np
from cil.optimisation.operators import GradientOperator
from cil.optimisation.utilities.callbacks import Callback
from krl.algorithms.lbfgsb import LBFGSBOptimizer, LBFGSBOptions
from krl.operators.directional import DirectionalOperator
from krl.utils import get_array

from krl_studies.methods.base import Iterate, Method
from krl_studies.methods.richardson_lucy import _blur_op

_WRAPPER_KEYS = ("alpha", "fwhm_mm", "backend", "lbfgs_max_linesearch", "lbfgs_ftol", "lbfgs_gtol")


class _CaptureSolution(Callback):
    def __init__(self, sink: list):
        super().__init__()
        self._sink = sink

    def __call__(self, algorithm) -> None:
        if int(algorithm.iteration) < 1:
            return
        arr = get_array(algorithm.solution).astype(np.float32, copy=True)
        arr[arr < 0] = 0.0
        obj = None
        objective = getattr(algorithm, "objective", None)
        if objective:
            try:
                obj = float(objective[-1])
            except (TypeError, ValueError, IndexError):
                obj = None
        self._sink.append(Iterate(int(algorithm.iteration), arr, obj))


class DTVMethod(Method):
    name = "dtv"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        """Stream MAP-RL/dTV iterates.

        L-BFGS-B may converge before n_iterations (gtol/ftol), yielding fewer
        iterates than requested; consumers must not assume exact counts.
        """
        unknown = set(params) - set(_WRAPPER_KEYS)
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}; dtv accepts {sorted(_WRAPPER_KEYS)}")
        if guidance is None:
            raise ValueError("dtv requires anatomical guidance image")

        blur = _blur_op(observed, float(params["fwhm_mm"]), params.get("backend", "numba"))

        fidelity = fn.KullbackLeibler(b=observed, eta=observed.geometry.allocate(value=1e-2))
        data_fidelity = fn.OperatorCompositionFunction(fidelity, blur)

        grad = GradientOperator(observed.geometry, method="forward", bnd_cond="Neumann")
        grad_ref = grad.direct(guidance)
        directional = op.CompositionOperator(DirectionalOperator(grad_ref), grad)
        prior = float(params["alpha"]) * fn.OperatorCompositionFunction(
            fn.SmoothMixedL21Norm(epsilon=float(observed.max()) * 1e-2), directional
        )

        options = LBFGSBOptions(
            max_linesearch=int(params.get("lbfgs_max_linesearch", 20)),
            ftol=float(params.get("lbfgs_ftol", 1e-6)),
            gtol=float(params.get("lbfgs_gtol", 1e-6)),
            enforce_non_negativity=True,
        )
        optimizer = LBFGSBOptimizer(
            initial_estimate=observed,
            data_fidelity=data_fidelity,
            prior=prior,
            options=options,
        )
        captured: list[Iterate] = []

        def generate():
            optimizer.run(
                verbose=0,
                iterations=int(n_iterations),
                callbacks=[_CaptureSolution(captured)],
            )
            yield from captured

        return generate()
