#!/usr/bin/env python3
"""Compare standard RL against kernel-guided RL with configurable parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import sirf.STIR as pet

from src.deconv_cli import (
    KernelParameters,
    PipelineConfig,
    configure_matplotlib,
    parse_common_args,
)
from run_deconv import (
    create_blurring_operator,
    load_images,
    prepare_kernel_operator,
    richardson_lucy,
    save_comparison,
    save_objective,
    save_profile_plot,
    save_reference_figures,
)


def run_pipeline(config: PipelineConfig, kernel_params: KernelParameters) -> None:
    for line in config.summary_lines(kernel_params):
        print(line)

    configure_matplotlib(config.show_plots)
    np.random.seed(config.noise_seed)
    pet.MessageRedirector()

    output_dir = config.output_directory(kernel_params)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_images(config)
    save_reference_figures(images, output_dir)

    blur = create_blurring_operator(config, images["OSEM"])
    reconstructions: Dict[str, pet.ImageData] = {}

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
        rl_img.write(str(output_dir / "deconv_rl.hv"))
        rl_img.write(str(output_dir / "deconv_rl.nii"))

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
            kernel_obj,
            output_dir / "deconv_kernel_objective.png",
            "KRL Objective",
        )
        deconv_kernel.write(str(output_dir / "deconv_kernel.hv"))
        deconv_kernel.write(str(output_dir / "deconv_kernel.nii"))

    if reconstructions:
        save_profile_plot(images, reconstructions, output_dir / "profile_comparison.png")


def main(argv=None) -> None:
    defaults = PipelineConfig(
        data_path=Path("data"),
        rl_iterations_standard=20,
        rl_iterations_kernel=100,
        fwhm=(4.0, 4.0, 4.0),
        do_drl=False,
        save_suffix="kl_comparison",
    )
    kernel_defaults = KernelParameters(
        num_neighbours=5,
        sigma_anat=0.5,
        sigma_dist=3.0,
    )
    config, kernel_params, _ = parse_common_args(
        defaults=defaults,
        kernel_defaults=kernel_defaults,
        description="Compare RL and Kernel RL reconstructions.",
        argv=argv,
        include_drl=False,
    )
    run_pipeline(config, kernel_params)


if __name__ == "__main__":
    main()
