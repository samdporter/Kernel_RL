"""RL / KRL / HKRL wrappers around krl.RichardsonLucy."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from cil.optimisation.utilities.callbacks import Callback
from krl.algorithms.richardson_lucy import RichardsonLucy
from krl.operators.blurring import create_gaussian_blur
from krl.operators.kernel_operator import get_kernel_operator
from krl.utils import get_array

from krl_studies.methods.base import Iterate, Method

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _blur_op(observed, fwhm_mm: float, backend: str = "numba"):
    sigma = tuple(fwhm_mm * FWHM_TO_SIGMA for _ in range(3))
    return create_gaussian_blur(sigma, observed.geometry, backend=backend)


class _Capture(Callback):
    """Record solution (and objective when available) every iteration."""

    def __init__(self, sink: list):
        super().__init__()
        self._sink = sink

    def __call__(self, algorithm) -> None:
        # CIL fires the callback once before the first update (iteration 0,
        # initial estimate); only post-update iterates belong to the stream.
        if int(algorithm.iteration) < 1:
            return
        obj = None
        loss = getattr(algorithm, "loss", None)
        if loss:
            try:
                obj = float(loss[-1])
            except (TypeError, ValueError):
                obj = None
        self._sink.append(
            Iterate(
                iteration=int(algorithm.iteration),
                image=get_array(algorithm.solution).astype(np.float32, copy=True),
                objective=obj,
            )
        )


class RLMethod(Method):
    name = "rl"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        _validate_params(params)
        blur = _blur_op(observed, float(params["fwhm_mm"]), params.get("backend", "numba"))
        captured: list[Iterate] = []
        algo = RichardsonLucy(
            initial_estimate=observed,
            blurring_operator=blur,
            observed_data=observed,
            epsilon=float(params.get("epsilon", 1e-10)),
            update_objective_interval=1,
        )

        def generate():
            algo.run(iterations=int(n_iterations), verbose=0, callbacks=[_Capture(captured)])
            yield from captured

        return generate()


def _kernel_params(params: dict[str, Any]) -> dict[str, Any]:
    """Whitelist of supported KernelOperator settings."""
    allowed = (
        "num_neighbours",
        "sigma_anat",
        "sigma_dist",
        "sigma_emission",
        "distance_weighting",
        "normalize_features",
        "normalize_kernel",
        "use_mask",
        "mask_k",
        "recalc_mask",
        "hybrid",
    )
    return {k: params[k] for k in allowed if k in params}


_WRAPPER_KEYS = ("fwhm_mm", "backend", "epsilon", "freeze_iteration")


def _validate_params(params: dict[str, Any]) -> None:
    unknown = set(params) - set(_WRAPPER_KEYS) - set(_kernel_params(params))
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {sorted(unknown)}; wrapper accepts "
            f"{sorted(set(_WRAPPER_KEYS))} and kernel accepts {sorted(set(_kernel_params(params)))}"
        )


class _KernelMethod(Method):
    def _run_kernel(self, observed, guidance, params, n_iterations, freeze_iteration):
        _validate_params(params)
        if guidance is None:
            raise ValueError(f"{self.name} requires anatomical guidance image")
        blur = _blur_op(observed, float(params["fwhm_mm"]), params.get("backend", "numba"))
        kernel_op = get_kernel_operator(observed, backend=params.get("backend", "numba"))
        kernel_op.set_parameters(_kernel_params(params))
        kernel_op.set_anatomical_image(guidance)

        algo = RichardsonLucy(
            initial_estimate=observed,
            blurring_operator=blur,
            observed_data=observed,
            kernel_operator=kernel_op,
            freeze_iteration=int(freeze_iteration),
            epsilon=float(params.get("epsilon", 1e-10)),
            update_objective_interval=1,
        )

        captured: list[Iterate] = []

        class _CaptureEmission(Callback):
            def __call__(self, algorithm) -> None:
                # CIL fires the callback once before the first update (iteration 0,
                # initial estimate); only post-update iterates belong to the stream.
                if int(algorithm.iteration) < 1:
                    return
                # Capture emission-domain iterate using current kernel state
                latent = algorithm.x.geometry.allocate()
                latent.fill(algorithm.x.as_array().copy())
                deconv = kernel_op.direct(latent)
                arr = get_array(deconv).astype(np.float32)
                arr[arr < 0] = 0.0
                obj = None
                loss = getattr(algorithm, "loss", None)
                if loss:
                    try:
                        obj = float(loss[-1])
                    except (TypeError, ValueError, IndexError):
                        obj = None
                captured.append(
                    Iterate(
                        iteration=int(algorithm.iteration),
                        image=arr,
                        objective=obj,
                    )
                )

        def generate():
            algo.run(iterations=int(n_iterations), verbose=0, callbacks=[_CaptureEmission()])
            yield from captured

        return generate()


class KRLMethod(_KernelMethod):
    name = "krl"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        return self._run_kernel(
            observed, guidance, params, n_iterations, freeze_iteration=params.get("freeze_iteration", 0)
        )


class HKRLMethod(_KernelMethod):
    name = "hkrl"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        # Enable hybrid if user provides sigma_emission > 0; otherwise raise if freeze requested
        freeze_it = int(params.get("freeze_iteration", 1))
        sigma_em = float(params.get("sigma_emission", 0.0))
        if sigma_em <= 0 and freeze_it > 0:
            raise ValueError("HKRL with freeze_iteration > 0 requires sigma_emission > 0 (hybrid mode)")
        kernel_params = dict(_kernel_params(params))
        kernel_params["hybrid"] = sigma_em > 0
        # Pass through so wrapper validates
        params_with_hybrid = dict(params)
        params_with_hybrid["hybrid"] = kernel_params["hybrid"]
        return self._run_kernel(
            observed, guidance, params_with_hybrid, n_iterations, freeze_iteration=freeze_it
        )
