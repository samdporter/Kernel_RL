"""Compare RL, KRL, directional-TV MAPRL and L-BFGS-B Poisson deconvolution.

A self-contained 3D deconvolution demo on a synthetic two-blob phantom:

* ``RL``       -- plain Richardson-Lucy deconvolution.
* ``KRL``      -- kernelised RL using the anatomical image as guidance.
* ``RL-dTV``   -- MAPRL with a Poisson deconvolution fidelity, a directional
  total-variation prior and a sensitivity-normalised EM preconditioner.
* ``L-BFGS-B`` -- the same fidelity and prior optimised with L-BFGS-B.

Run with ``python examples/rl_krl_dtv.py`` or, for a quick check,
``python examples/rl_krl_dtv.py --shape 6 8 8 --iterations 2``.
"""

import argparse

import numpy as np
from cil.framework import ImageGeometry
from cil.optimisation.functions import (
    KullbackLeibler,
    OperatorCompositionFunction,
    SmoothMixedL21Norm,
)
from cil.optimisation.operators import CompositionOperator, GradientOperator

from krl import (
    MAPRL,
    DirectionalOperator,
    LBFGSBOptimizer,
    LBFGSBOptions,
    RichardsonLucy,
    create_gaussian_blur,
    get_kernel_operator,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50, help="number of iterations (>= 1)")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=[8, 16, 16],
        metavar=("Z", "Y", "X"),
        help="volume shape in voxels, each >= 2",
    )
    args = parser.parse_args(argv)

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if any(size < 2 for size in args.shape):
        parser.error("--shape entries must each be >= 2")
    return args


def _make_phantom(geometry):
    z, y, x = geometry.shape
    axes = [np.linspace(0.0, 1.0, size, dtype=np.float32) for size in (z, y, x)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([zz, yy, xx], axis=-1)

    g1 = np.exp(-np.sum((coords - (0.35, 0.40, 0.40)) ** 2, axis=-1) / 0.02)
    g2 = np.exp(-np.sum((coords - (0.65, 0.65, 0.65)) ** 2, axis=-1) / 0.015)

    ground_truth = geometry.allocate(0.0)
    ground_truth.fill((0.1 + 50.0 * g1 + 30.0 * g2).astype(np.float32))
    anatomical = geometry.allocate(0.0)
    anatomical.fill((g1 + g2).astype(np.float32))
    return ground_truth, anatomical


def _nrmse_percent(image, truth):
    difference = image.as_array() - truth.as_array()
    return float(100.0 * np.sqrt(np.mean(difference**2)) / truth.as_array().max())


def _record(results, name, reconstruction, objectives, ground_truth, iterations, **extra):
    nrmse_percent = _nrmse_percent(reconstruction, ground_truth)
    history = ", ".join(f"{value:.6g}" for value in objectives)
    print(f"{name}: objective history, initial included = [{history}]")
    print(f"{name}: final NRMSE = {nrmse_percent:.3f}% (iterations = {iterations})")
    entry = {
        "reconstruction": reconstruction,
        "objectives": objectives,
        "nrmse_percent": nrmse_percent,
        "iterations": iterations,
    }
    entry.update(extra)
    results[name] = entry


def _run_richardson_lucy(blur, observed, iterations, kernel_operator=None):
    algorithm = RichardsonLucy(
        initial_estimate=observed.clone(),
        blurring_operator=blur,
        observed_data=observed,
        kernel_operator=kernel_operator,
        update_objective_interval=1,
    )
    algorithm.run(iterations, verbose=0)
    return algorithm.get_output(), list(algorithm.loss)


def _build_fidelity_and_prior(geometry, anatomical, observed, blur):
    fidelity = OperatorCompositionFunction(
        KullbackLeibler(b=observed, eta=geometry.allocate(1e-6)), blur
    )
    gradient = GradientOperator(geometry, method="forward", bnd_cond="Neumann")
    anatomical_gradient = gradient.direct(anatomical)
    directional = DirectionalOperator(anatomical_gradient, gamma=1.0, eta=0.05)
    directional_gradient = CompositionOperator(directional, gradient)
    prior = 0.01 * OperatorCompositionFunction(
        SmoothMixedL21Norm(epsilon=0.01 * float(observed.max())), directional_gradient
    )
    return fidelity, prior


def _make_em_preconditioner(geometry, blur):
    """Sensitivity-normalised EM preconditioner ``P(x) = (x + 1e-6) / max(Bᵀ1, 1e-6)``."""
    sensitivity = blur.adjoint(geometry.allocate(1.0))
    sensitivity.maximum(1e-6)

    def preconditioner(image):
        return (image + 1e-6) / sensitivity

    return preconditioner


def _run_rl_dtv(fidelity, prior, observed, iterations, preconditioner):
    # eps is only used on the unpreconditioned update path; with a callable
    # preconditioner the DTV prior gradient is also scaled by P (EM-MAP
    # convention), so eps does not enter the preconditioned update.
    algorithm = MAPRL(
        initial_estimate=observed.clone(),
        data_fidelity=fidelity,
        prior=prior,
        step_size=1.0,
        relaxation_eta=0.0,
        eps=1e-6,
        preconditioner=preconditioner,
        preconditioner_update_initial=0,
        preconditioner_update_interval=1,
        initial_line_search=False,
        armijo_iterations=iterations,
        armijo_update_initial=0,
        armijo_update_interval=0,
        armijo_beta=0.5,
        armijo_sigma=1e-4,
        armijo_max_iter=30,
        update_objective_interval=1,
    )
    algorithm.run(iterations, verbose=0)
    return algorithm.x, list(algorithm.loss), algorithm.step_size()


def _run_lbfgsb(fidelity, prior, observed, iterations):
    optimizer = LBFGSBOptimizer(
        initial_estimate=observed.clone(),
        data_fidelity=fidelity,
        prior=prior,
        options=LBFGSBOptions(
            max_linesearch=20,
            ftol=1e-6,
            gtol=1e-6,
            enforce_non_negativity=True,
        ),
    )
    optimizer.run(iterations=iterations, verbose=0)
    return optimizer


def main(argv=None):
    args = _parse_args(argv)
    z, y, x = args.shape
    geometry = ImageGeometry(voxel_num_x=x, voxel_num_y=y, voxel_num_z=z, dtype=np.float32)

    ground_truth, anatomical = _make_phantom(geometry)
    blur = create_gaussian_blur(sigma=(1, 1, 1), geometry=geometry, backend="numba")
    observed = blur.direct(ground_truth)

    print("RL-dTV — MAPRL with Poisson deconvolution fidelity and directional TV.")
    print(f"observed (blurred input): NRMSE = {_nrmse_percent(observed, ground_truth):.3f}%")

    fidelity, prior = _build_fidelity_and_prior(geometry, anatomical, observed, blur)
    preconditioner = _make_em_preconditioner(geometry, blur)

    results = {}

    rl_reconstruction, rl_objectives = _run_richardson_lucy(blur, observed, args.iterations)
    _record(results, "RL", rl_reconstruction, rl_objectives, ground_truth, args.iterations)

    kernel = get_kernel_operator(
        geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        normalize_kernel=True,
        use_mask=False,
        hybrid=False,
    )
    kernel.set_anatomical_image(anatomical)
    krl_reconstruction, krl_objectives = _run_richardson_lucy(
        blur, observed, args.iterations, kernel_operator=kernel
    )
    _record(results, "KRL", krl_reconstruction, krl_objectives, ground_truth, args.iterations)

    dtv_reconstruction, dtv_objectives, dtv_step = _run_rl_dtv(
        fidelity, prior, observed, args.iterations, preconditioner
    )
    _record(
        results,
        "RL-dTV",
        dtv_reconstruction,
        dtv_objectives,
        ground_truth,
        args.iterations,
        final_step_size=dtv_step,
    )

    lbfgsb = _run_lbfgsb(fidelity, prior, observed, args.iterations)
    result = lbfgsb.result
    print(
        "L-BFGS-B: scipy_nfev = "
        f"{result.nfev}, scipy_njev = {result.njev}, success = {result.success}, "
        f"status = {result.status}, message = {result.message!r}"
    )
    _record(
        results,
        "L-BFGS-B",
        lbfgsb.solution,
        list(lbfgsb.objective),
        ground_truth,
        result.nit,
        scipy_nfev=result.nfev,
        scipy_njev=result.njev,
        success=result.success,
        status=result.status,
        message=result.message,
    )

    return results


if __name__ == "__main__":
    main()
