#!/usr/bin/env python3
"""CLI entry point for running PET deconvolution experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import logging

import matplotlib.pyplot as plt
import numpy as np

# Import CIL (required)
try:
    from cil.framework import ImageData
    import cil.optimisation.functions as fn
    import cil.optimisation.operators as op
    from cil.optimisation.operators import BlurringOperator, GradientOperator
    from cil.optimisation.utilities.callbacks import Callback
    import cil.optimisation.algorithms as alg
    from cil.utilities.display import show2D
except ImportError as e:
    raise ImportError(
        "CIL is required. Please install it with:\n"
        "  conda install -c conda-forge -c ccpi cil"
    ) from e

from krl.cli.config import (
    KernelParameters,
    PipelineConfig,
    configure_matplotlib,
    parse_common_args,
)
from krl.operators.directional import DirectionalOperator
from krl.operators.blurring import create_gaussian_blur
from krl.algorithms.lbfgsb import LBFGSBOptimizer, LBFGSBOptions
from krl.algorithms.richardson_lucy import RichardsonLucy
from krl.operators.kernel_operator import get_kernel_operator
from krl.utils import load_image, save_image
from krl.callbacks import NRMSECallback, SaveIterationCallback


LOGGER = logging.getLogger(__name__)


def fwhm_to_sigma(fwhm: Tuple[float, float, float]) -> Tuple[float, float, float]:
    scale = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return tuple(value * scale for value in fwhm)


def geometry_voxel_sizes(geometry) -> Tuple[float, float, float]:
    """Return voxel sizes as (z, y, x) for a CIL ImageGeometry."""
    return (
        float(geometry.voxel_size_z),
        float(geometry.voxel_size_y),
        float(geometry.voxel_size_x),
    )


def psf(kernel_size: int, fwhm: Tuple[float, float, float], voxel_size: Tuple[float, float, float]) -> np.ndarray:
    sigma = fwhm_to_sigma(fwhm)
    sigma_voxels = [sigma[i] / voxel_size[i] for i in range(3)]
    axes = [
        np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        for _ in range(3)
    ]
    gauss = [np.exp(-0.5 * ax**2 / sigma_voxels[i] ** 2) for i, ax in enumerate(axes)]
    kernel = (
        np.outer(gauss[0], gauss[1]).reshape(kernel_size, kernel_size, 1)
        * gauss[2].reshape(1, 1, kernel_size)
    )
    return kernel / np.sum(kernel)


def load_images(config: PipelineConfig) -> Dict[str, ImageData]:
    """Load emission and guidance images from file."""
    images = {
        "OSEM": load_image(config.emission_path()),
        "T1": load_image(config.guidance_path()),
    }

    if config.flip_emission:
        arr = images["OSEM"].as_array()
        images["OSEM"].fill(np.flip(arr, axis=0))

    if config.flip_guidance:
        arr = images["T1"].as_array()
        images["T1"].fill(np.flip(arr, axis=0))

    # Load ground truth if available
    if config.ground_truth_path() is not None:
        gt_path = config.ground_truth_path()
        if gt_path.exists():
            images["ground_truth"] = load_image(gt_path)
            print(f"Loaded ground truth from {gt_path}")
        else:
            print(f"Warning: Ground truth file specified but not found: {gt_path}")

    return images


def save_reference_figures(images: Mapping[str, ImageData], output_dir: Path, osem_vmax: float) -> None:
    """Save reference comparison figures."""
    t1_vmax = float(np.max(images["T1"].as_array()))
    fig = show2D(
        [images["OSEM"], images["T1"]],
        title=["OSEM", "T1"],
        origin="upper",
        num_cols=2,
        fix_range=[(0, osem_vmax), (0, t1_vmax)],
    )
    fig.save(output_dir / "OSEM_T1.png")

    slices = [images["OSEM"].shape[0] // 4, 3 * images["OSEM"].shape[0] // 4]
    fig, ax = plt.subplots(len(slices), 2, figsize=(10, 5))
    for row, z in enumerate(slices):
        ax[row, 0].imshow(images["OSEM"].as_array()[z], cmap="gray", vmin=0, vmax=osem_vmax)
        ax[row, 0].set_title("OSEM")
        ax[row, 1].imshow(images["T1"].as_array()[z], cmap="gray", vmin=0, vmax=t1_vmax)
        ax[row, 1].set_title("T1")
    for axis in np.atleast_1d(ax).ravel():
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "OSEM_T1_quarter_slices.png")
    plt.close(fig)


def create_blurring_operator(config: PipelineConfig, emission: ImageData):
    """Create Gaussian blurring operator for PSF."""
    try:
        return create_gaussian_blur(
            fwhm_to_sigma(config.fwhm),
            emission.geometry,
            backend="auto",
        )
    except (ImportError, AttributeError):
        # Fallback to CIL's BlurringOperator
        kernel = psf(
            config.psf_kernel_size,
            config.fwhm,
            geometry_voxel_sizes(emission.geometry),
        )
        return BlurringOperator(kernel, emission)


def prepare_kernel_operator(
    config: PipelineConfig,
    kernel_params: KernelParameters,
    emission: ImageData,
    guidance: ImageData,
):
    operator = get_kernel_operator(emission, backend=config.backend)
    operator.set_parameters(kernel_params.as_operator_kwargs())
    operator.set_anatomical_image(guidance)
    return operator


def richardson_lucy(
    observed: ImageData,
    blur_op,
    iterations: int,
    freeze_iteration: int = 0,
    *,
    epsilon: float = 1e-10,
    kernel_operator=None,
    save_dir: Path,
    save_interval: int = 10,
    save_first_n: int = 5,
    tag: str = "",
    callbacks=None,
    use_uniform_initial: bool = False,
):
    """
    Run Richardson-Lucy deconvolution using the RichardsonLucy algorithm class.

    This is a convenience wrapper around the RichardsonLucy class that maintains
    backward compatibility with the old function-based interface.

    Parameters
    ----------
    observed : ImageData
        Observed blurred image
    blur_op : LinearOperator
        Blurring operator (PSF)
    iterations : int
        Number of RL iterations
    freeze_iteration : int, optional
        Freeze kernel operator at this iteration (for KRL/HKRL)
    epsilon : float, optional
        Small value to avoid division by zero
    kernel_operator : LinearOperator, optional
        Kernel operator for KRL/HKRL
    save_dir : Path
        Directory to save intermediate results
    save_interval : int, optional
        Save every N iterations
    save_first_n : int, optional
        Save first N iterations before using interval
    tag : str, optional
        Prefix for saved filenames
    callbacks : list, optional
        List of callback functions
    use_uniform_initial : bool, optional
        Use uniform image as initial estimate instead of observed image

    Returns
    -------
    current : ImageData
        Latent image (for KRL) or reconstructed image (for standard RL)
    objective_values : list
        Objective function values per iteration
    """
    # Prepare callbacks list
    all_callbacks = []

    # Add save callback if needed
    if save_interval > 0:
        save_callback = SaveIterationCallback(
            output_dir=save_dir,
            interval=save_interval,
            prefix=f"{tag}deconv_iter",
            kernel_operator=kernel_operator,
            save_first_n=save_first_n,
        )
        all_callbacks.append(save_callback)

    # Add user-provided callbacks
    if callbacks is not None:
        all_callbacks.extend(callbacks)

    # Determine initial estimate
    if use_uniform_initial:
        initial_estimate = observed.geometry.allocate(value=float(np.mean(observed.as_array())))
    else:
        initial_estimate = observed

    # Create and run the algorithm
    rl = RichardsonLucy(
        initial_estimate=initial_estimate,
        blurring_operator=blur_op,
        observed_data=observed,
        kernel_operator=kernel_operator,
        freeze_iteration=freeze_iteration,
        epsilon=epsilon,
        update_objective_interval=1,
    )

    rl.run(
        iterations=iterations,
        verbose=0,
        callbacks=all_callbacks if all_callbacks else None,
    )

    return rl.x, rl.loss


def save_objective(values: Iterable[float], output: Path, title: str) -> None:
    """Save objective values as both CSV and plot.

    Parameters
    ----------
    values : Iterable[float]
        Objective values per iteration
    output : Path
        Output path for the plot (CSV will have same name with .csv extension)
    title : str
        Plot title
    """
    values_list = list(values)

    # Save as CSV
    csv_path = output.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("iteration,objective\n")
        for i, val in enumerate(values_list, start=1):
            f.write(f"{i},{val:.8e}\n")

    # Save plot
    plt.figure()
    plt.plot(values_list)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def save_comparison(
    result: ImageData,
    reference: ImageData,
    output: Path,
    title: str,
    ranges=None,
) -> None:
    """Save comparison figure of two images."""
    fig = show2D(
        [result, reference],
        title=[title, "OSEM"],
        origin="upper",
        num_cols=2,
        fix_range=ranges,
    )
    fig.save(output)


def save_profile_plot(
    images: Mapping[str, ImageData],
    reconstructions: Mapping[str, ImageData],
    output: Path,
) -> None:
    """Save 1D profile comparison plot."""
    center_slice = images["OSEM"].shape[0] // 2
    profile_axis = images["OSEM"].shape[2] // 2
    plt.figure(figsize=(15, 5))
    for label, img in reconstructions.items():
        plt.plot(
            img.as_array()[center_slice, :, profile_axis],
            label=label,
        )
    plt.plot(
        images["OSEM"].as_array()[center_slice, :, profile_axis],
        label="OSEM",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def run_pipeline(config: PipelineConfig, kernel_params: KernelParameters) -> None:
    for line in config.summary_lines(kernel_params):
        print(line)

    configure_matplotlib(config.show_plots)
    np.random.seed(config.noise_seed)

    output_dir = config.output_directory(kernel_params)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_images(config)

    # Calculate vmax from OSEM image for consistent scaling across all plots
    osem_vmax = float(np.max(images["OSEM"].as_array()))

    save_reference_figures(images, output_dir, osem_vmax)
    print(f"size of OSEM image: {images['OSEM'].shape}")
    print(f"size of T1 image: {images['T1'].shape}")

    blur = create_blurring_operator(config, images["OSEM"])

    reconstructions: Dict[str, ImageData] = {}

    # Prepare NRMSE callbacks if ground truth is available
    ground_truth = images.get("ground_truth")

    if config.do_rl:
        rl_callbacks = []
        if ground_truth is not None:
            rl_callbacks.append(
                NRMSECallback(
                    ground_truth=ground_truth,
                    output_file=output_dir / "rl_nrmse.csv",
                    interval=1,
                    verbose=True,
                )
            )

        rl_img, rl_obj = richardson_lucy(
            images["OSEM"],
            blur,
            iterations=config.rl_iterations_standard,
            freeze_iteration=config.freeze_iteration,
            save_dir=output_dir,
            save_first_n=config.save_first_n,
            tag="rl_",
            callbacks=rl_callbacks if rl_callbacks else None,
            use_uniform_initial=config.use_uniform_initial,
        )
        with np.errstate(invalid="ignore"):
            rl_img.maximum(0, out=rl_img)
        reconstructions["RL"] = rl_img
        save_comparison(
            rl_img,
            images["OSEM"],
            output_dir / "deconv_rl_comparison.png",
            "Deconv RL",
            ranges=[(0, osem_vmax), (0, osem_vmax)],
        )
        save_objective(rl_obj, output_dir / "deconv_rl_objective.png", "RL Objective")
        save_image(rl_img, output_dir / "deconv_rl.nii.gz")

    kernel_operator = None
    if config.do_krl:
        kernel_operator = prepare_kernel_operator(
            config,
            kernel_params,
            images["OSEM"],
            images["T1"],
        )

        krl_callbacks = []
        if ground_truth is not None:
            krl_callbacks.append(
                NRMSECallback(
                    ground_truth=ground_truth,
                    output_file=output_dir / "krl_nrmse.csv",
                    interval=1,
                    kernel_operator=kernel_operator,
                    verbose=True,
                )
            )

        kalpha, kernel_obj = richardson_lucy(
            images["OSEM"],
            blur,
            iterations=config.rl_iterations_kernel,
            kernel_operator=kernel_operator,
            freeze_iteration=config.freeze_iteration if kernel_params.hybrid else 0,
            save_dir=output_dir,
            save_first_n=config.save_first_n,
            tag="kernel_",
            callbacks=krl_callbacks if krl_callbacks else None,
            use_uniform_initial=config.use_uniform_initial,
        )
        deconv_kernel = kernel_operator.direct(kalpha)
        with np.errstate(invalid="ignore"):
            deconv_kernel.maximum(0, out=deconv_kernel)
        reconstructions["KRL"] = deconv_kernel
        save_comparison(
            deconv_kernel,
            images["OSEM"],
            output_dir / "deconv_kernel_comparison.png",
            "Deconv KRL",
            ranges=[(0, osem_vmax), (0, osem_vmax)],
        )
        save_objective(
            kernel_obj, output_dir / "deconv_kernel_objective.png", "KRL Objective"
        )
        save_image(deconv_kernel, output_dir / "deconv_kernel.nii.gz")

        if getattr(kernel_operator, "mask", None) is not None:
            center = tuple(s // 2 for s in images["OSEM"].shape)
            mask_vals = kernel_operator.mask[center]  # type: ignore[index]
            plt.figure()
            plt.plot(mask_vals, "o")
            plt.tight_layout()
            plt.savefig(output_dir / "kernel_mask.png")
            plt.close()

    if config.do_drl:
        f = fn.KullbackLeibler(
            b=images["OSEM"],
            eta=images["OSEM"].geometry.allocate(value=1e-2),
        )
        df = fn.OperatorCompositionFunction(f, blur)
        grad = GradientOperator(images["OSEM"].geometry, method="forward", bnd_cond="Neumann")
        grad_ref = grad.direct(images["T1"])
        d_op = op.CompositionOperator(DirectionalOperator(grad_ref), grad)
        prior = config.alpha * fn.OperatorCompositionFunction(
            fn.SmoothMixedL21Norm(epsilon=images["OSEM"].max() * 1e-2), d_op
        )

        class SaveCallback(Callback):
            def __init__(self, interval: int, save_first_n: int = 5) -> None:
                super().__init__()
                self.interval = interval
                self.save_first_n = save_first_n

            def __call__(self, algorithm: alg.Algorithm) -> None:  # type: ignore[override]
                # Save first N iterations (1, 2, 3, 4, 5)
                if algorithm.iteration <= self.save_first_n:
                    should_save = True
                # Then save at regular intervals (10, 20, 30...)
                elif algorithm.iteration % self.interval == 0:
                    should_save = True
                else:
                    should_save = False

                if not should_save:
                    return

                with np.errstate(invalid="ignore"):
                    algorithm.solution.maximum(0, out=algorithm.solution)
                save_image(
                    algorithm.solution,
                    output_dir / f"dtv_iter_{algorithm.iteration}.nii.gz"
                )

        dtv_callbacks = [SaveCallback(10, save_first_n=config.save_first_n)]
        if ground_truth is not None:
            dtv_callbacks.append(
                NRMSECallback(
                    ground_truth=ground_truth,
                    output_file=output_dir / "dtv_nrmse.csv",
                    interval=1,
                    verbose=True,
                )
            )

        # Determine initial estimate
        if config.use_uniform_initial:
            dtv_initial = images["OSEM"].geometry.allocate(value=float(np.mean(images["OSEM"].as_array())))
        else:
            dtv_initial = images["OSEM"]

        lbfgs_options = LBFGSBOptions(
            max_linesearch=config.lbfgs_max_linesearch,
            ftol=config.lbfgs_ftol,
            gtol=config.lbfgs_gtol,
            enforce_non_negativity=True,
        )
        lbfgsb = LBFGSBOptimizer(
            initial_estimate=dtv_initial,
            data_fidelity=df,
            prior=prior,
            options=lbfgs_options,
        )
        lbfgsb.run(
            verbose=1,
            iterations=config.dtv_iterations,
            callbacks=dtv_callbacks,
        )
        deconv_dtv = lbfgsb.solution
        with np.errstate(invalid="ignore"):
            deconv_dtv.maximum(0, out=deconv_dtv)
        reconstructions["DTV"] = deconv_dtv
        save_comparison(
            deconv_dtv,
            images["OSEM"],
            output_dir / "deconv_dtv_comparison.png",
            "Deconv DTV",
            ranges=[(0, osem_vmax), (0, osem_vmax)],
        )
        save_objective(
            lbfgsb.objective,
            output_dir / "deconv_dtv_objective.png",
            "DTV Objective",
        )
        save_image(deconv_dtv, output_dir / "deconv_dtv.nii.gz")

    if reconstructions:
        save_profile_plot(images, reconstructions, output_dir / "profile_comparison.png")


def main(argv=None) -> None:
    defaults = PipelineConfig(data_path=Path("data"))
    kernel_defaults = KernelParameters()
    config, kernel_params, _ = parse_common_args(
        defaults=defaults,
        kernel_defaults=kernel_defaults,
        description="Run PET deconvolution with configurable operators.",
        argv=argv,
    )
    run_pipeline(config, kernel_params)


if __name__ == "__main__":
    main()
