"""MAP-RL with directional TV, following examples/pipelines/run_deconv.py."""

from __future__ import annotations

from collections.abc import Iterator

import cil.optimisation.functions as fn
import cil.optimisation.operators as op
import numpy as np
from cil.optimisation.operators import BlurringOperator, GradientOperator
from cil.optimisation.utilities.callbacks import Callback
from krl.algorithms.lbfgsb import LBFGSBOptimizer, LBFGSBOptions
from krl.operators.directional import DirectionalOperator
from krl.utils import get_array

from krl_studies.methods.base import Iterate, Method
from krl_studies.methods.richardson_lucy import FWHM_TO_SIGMA, _blur_op

_WRAPPER_KEYS = ("alpha", "fwhm_mm", "backend", "lbfgs_max_linesearch", "lbfgs_ftol", "lbfgs_gtol")


def _psf_kernel(kernel_size: int, sigma: tuple[float, float, float], voxel: tuple[float, float, float]):
    axes = [np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size) for _ in range(3)]
    sig_vox = [sigma[i] / voxel[i] for i in range(3)]
    gauss = [np.exp(-0.5 * ax**2 / sv**2) for ax, sv in zip(axes, sig_vox)]
    k = (
        np.outer(gauss[0], gauss[1]).reshape(kernel_size, kernel_size, 1)
        * gauss[2].reshape(1, 1, kernel_size)
    )
    return (k / k.sum()).astype(np.float32)


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
        unknown = set(params) - set(_WRAPPER_KEYS)
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}; dtv accepts {sorted(_WRAPPER_KEYS)}")
        if guidance is None:
            raise ValueError("dtv requires anatomical guidance image")

        sigma = tuple(float(params["fwhm_mm"]) * FWHM_TO_SIGMA for _ in range(3))
        try:
            blur = _blur_op(observed, float(params["fwhm_mm"]), params.get("backend", "numba"))
        except (ImportError, AttributeError):
            voxel = (
                observed.geometry.voxel_size_z,
                observed.geometry.voxel_size_y,
                observed.geometry.voxel_size_x,
            )
            blur = BlurringOperator(_psf_kernel(5, sigma, voxel), observed)

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
