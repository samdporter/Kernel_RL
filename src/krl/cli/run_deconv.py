#!/usr/bin/env python3
"""CLI entry point for running PET deconvolution experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

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
from krl.algorithms.maprl import MAPRL
from krl.operators.kernel_operator import get_kernel_operator
from krl.utils import load_image, save_image


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
        images["OSEM"].fill(np.flip(arr, axis=1))
    return images


def save_reference_figures(images: Mapping[str, ImageData], output_dir: Path) -> None:
    """Save reference comparison figures."""
    fig = show2D(
        [images["OSEM"], images["T1"]],
        title=["OSEM", "T1"],
        origin="upper",
        num_cols=2,
        fix_range=[(0, 10), (0, 5)],
    )
    fig.save(output_dir / "OSEM_T1.png")

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    mid = images["OSEM"].shape[0] // 2
    quarter = images["OSEM"].shape[0] // 4
    slices = [quarter, 3 * quarter]
    for idx, z in enumerate(slices):
        row, col = divmod(idx, 2)
        ax[row, col].imshow(images["OSEM"].as_array()[z], cmap="gray")
        ax[row, col].set_title("OSEM")
        ax[row, col + 1].imshow(images["T1"].as_array()[z], cmap="gray")
        ax[row, col + 1].set_title("T1")
    for axis in ax.flat:
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
    *,
    epsilon: float = 1e-10,
    kernel_operator=None,
    save_dir: Path,
    save_interval: int = 10,
    tag: str = "",
):
    geometry = observed.geometry
    sensitivity = geometry.allocate(value=1)
    effective_blur = blur_op
    if kernel_operator is not None:
        effective_blur = op.CompositionOperator(blur_op, kernel_operator)
        sensitivity = effective_blur.adjoint(geometry.allocate(value=1))

    current = observed.clone()
    est_blur = effective_blur.direct(current)
    objective_values = []

    for idx in range(iterations):
        current *= effective_blur.adjoint(observed / (est_blur + epsilon))
        current /= (sensitivity + epsilon)
        est_blur = effective_blur.direct(current)
        obj = (est_blur - observed * (est_blur + epsilon).log()).sum()
        objective_values.append(obj)

        if (idx + 1) % save_interval == 0:
            output = kernel_operator.direct(current) if kernel_operator else current
            save_image(output, save_dir / f"{tag}deconv_iter_{idx+1}.nii.gz")

    return current, objective_values


def save_objective(values: Iterable[float], output: Path, title: str) -> None:
    plt.figure()
    plt.plot(list(values))
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
    save_reference_figures(images, output_dir)
    print(f"size of OSEM image: {images['OSEM'].shape}")
    print(f"size of T1 image: {images['T1'].shape}")

    blur = create_blurring_operator(config, images["OSEM"])

    reconstructions: Dict[str, ImageData] = {}

    if config.do_rl:
        rl_img, rl_obj = richardson_lucy(
            images["OSEM"],
            blur,
            iterations=config.rl_iterations_standard,
            save_dir=output_dir,
            tag="rl_",
        )
        reconstructions["RL"] = rl_img
        save_comparison(
            rl_img,
            images["OSEM"],
            output_dir / "deconv_rl_comparison.png",
            "Deconv RL",
            ranges=[(0, 320), (0, 320)],
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
        kalpha, kernel_obj = richardson_lucy(
            images["OSEM"],
            blur,
            iterations=config.rl_iterations_kernel,
            kernel_operator=kernel_operator,
            save_dir=output_dir,
            tag="kernel_",
        )
        deconv_kernel = kernel_operator.direct(kalpha)
        reconstructions["KRL"] = deconv_kernel
        save_comparison(
            deconv_kernel,
            images["OSEM"],
            output_dir / "deconv_kernel_comparison.png",
            "Deconv KRL",
            ranges=[(0, 320), (0, 320)],
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
            eta=images["OSEM"].geometry.allocate(value=1e-6),
        )
        df = fn.OperatorCompositionFunction(f, blur)
        grad = GradientOperator(images["OSEM"].geometry, method="forward", bnd_cond="Neumann")
        grad_ref = grad.direct(images["T1"])
        d_op = op.CompositionOperator(DirectionalOperator(grad_ref), grad)
        prior = config.alpha * fn.OperatorCompositionFunction(
            fn.SmoothMixedL21Norm(epsilon=1e-4), d_op
        )

        class SaveCallback(Callback):
            def __init__(self, interval: int) -> None:
                super().__init__()
                self.interval = interval

            def __call__(self, algorithm: alg.Algorithm) -> None:  # type: ignore[override]
                if algorithm.iteration % self.interval == 0:
                    save_image(
                        algorithm.solution,
                        output_dir / f"dtv_iter_{algorithm.iteration}.nii.gz"
                    )

        maprl = MAPRL(
            initial_estimate=images["OSEM"],
            data_fidelity=df,
            prior=prior,
            step_size=config.step_size,
            relaxation_eta=config.relaxation_eta,
            update_objective_interval=config.update_obj_interval,
        )
        maprl.run(
            verbose=1,
            iterations=config.dtv_iterations,
            callbacks=[SaveCallback(10)],
        )
        deconv_dtv = maprl.solution
        reconstructions["DTV"] = deconv_dtv
        save_comparison(
            deconv_dtv,
            images["OSEM"],
            output_dir / "deconv_dtv_comparison.png",
            "Deconv DTV",
            ranges=[(0, 320), (0, 320)],
        )
        save_objective(
            maprl.objective,
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
