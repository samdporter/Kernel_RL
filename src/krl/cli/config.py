from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return int(value)


def _tuple3(values: Sequence[float]) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three values for FWHM.")
    return float(values[0]), float(values[1]), float(values[2])


@dataclass
class KernelParameters:
    num_neighbours: int = 9
    sigma_anat: float = 1.0
    sigma_dist: float = 3.0
    sigma_emission: float = 1.0
    distance_weighting: bool = False
    normalize_features: bool = True
    normalize_kernel: bool = True
    use_mask: bool = True
    mask_k: Optional[int] = 48
    recalc_mask: bool = False
    hybrid: bool = True

    def as_operator_kwargs(self) -> dict:
        return {
            "num_neighbours": self.num_neighbours,
            "sigma_anat": self.sigma_anat,
            "sigma_dist": self.sigma_dist,
            "sigma_emission": self.sigma_emission,
            "distance_weighting": self.distance_weighting,
            "normalize_features": self.normalize_features,
            "normalize_kernel": self.normalize_kernel,
            "use_mask": self.use_mask,
            "mask_k": self.mask_k,
            "recalc_mask": self.recalc_mask,
            "hybrid": self.hybrid,
        }


@dataclass
class PipelineConfig:
    data_path: Path = Path("data")
    output_root: Path = Path("results")
    emission_file: str = "OSEM_b1337_n5.hv"
    guidance_file: str = "T1_b1337.hv"
    ground_truth_file: Optional[str] = None
    backend: str = "numba"
    show_plots: bool = False
    flip_emission: bool = False
    flip_guidance: bool = False
    do_rl: bool = True
    do_krl: bool = True
    do_drl: bool = True
    save_suffix: str = "brain_simple"

    noise_seed: int = 5
    bw_seed: int = 1337
    rl_iterations_kernel: int = 100
    rl_iterations_standard: int = 100
    freeze_iteration: int = 0
    dtv_iterations: int = 100
    alpha: float = 0.1
    step_size: float = 0.2
    relaxation_eta: float = 0.0
    update_obj_interval: int = 1
    armijo_iterations: int = 1000
    armijo_update_initial: int = 10
    armijo_update_interval: int = 25
    preconditioner_update_initial: int = 10
    preconditioner_update_interval: int = 25
    psf_kernel_size: int = 5
    fwhm: Tuple[float, float, float] = (6.0, 6.0, 6.0)
    save_first_n: int = 5
    use_uniform_initial: bool = False

    def emission_path(self) -> Path:
        return self.data_path / self.emission_file

    def guidance_path(self) -> Path:
        return self.data_path / self.guidance_file

    def ground_truth_path(self) -> Optional[Path]:
        if self.ground_truth_file is None:
            return None
        return self.data_path / self.ground_truth_file

    def output_directory(self, kernel: KernelParameters) -> Path:
        parts = [self.save_suffix]
        if self.do_rl:
            parts.append("RL")
        if self.do_krl:
            parts.append(
                f"KRL_s{self._fmt(kernel.sigma_anat)}"
                f"_sd_{self._fmt(kernel.sigma_dist)}"
                f"_k{kernel.num_neighbours}"
            )
        if self.do_drl:
            parts.append(
                f"DRL_a{self._fmt(self.alpha)}"
                f"_s{self._fmt(self.step_size)}"
                f"_eta{self._fmt(self.relaxation_eta)}"
            )
        suffix = "_".join(parts) + f"_G{self._fmt(self.fwhm[0])}mm"
        if self.use_uniform_initial:
            suffix += "_init-uniform"
        else:
            suffix += "_init-blurred"
        return self.output_root / suffix

    def summary_lines(self, kernel: KernelParameters) -> Iterable[str]:
        yield "Deconvolution configuration:"
        yield f"  data_path: {self.data_path}"
        yield f"  emission_file: {self.emission_file}"
        yield f"  guidance_file: {self.guidance_file}"
        yield f"  output_root: {self.output_root}"
        yield f"  backend: {self.backend}"
        yield f"  show_plots: {self.show_plots}"
        yield f"  flip_emission: {self.flip_emission}"
        yield f"  flip_guidance: {self.flip_guidance}"
        yield f"  run RL: {self.do_rl}"
        yield f"  run KRL: {self.do_krl}"
        yield f"  run DRL: {self.do_drl}"
        yield f"  rl_iterations_standard: {self.rl_iterations_standard}"
        yield f"  rl_iterations_kernel: {self.rl_iterations_kernel}"
        yield f"  freeze_iteration: {self.freeze_iteration}"
        yield f"  dtv_iterations: {self.dtv_iterations}"
        yield f"  psf_kernel_size: {self.psf_kernel_size}"
        yield f"  fwhm: {self.fwhm}"
        yield f"  use_uniform_initial: {self.use_uniform_initial}"
        yield "Kernel parameters:"
        for key, value in kernel.as_operator_kwargs().items():
            yield f"  {key}: {value}"

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value}".replace(".", "_")


def configure_matplotlib(show_plots: bool) -> None:
    if not show_plots:
        import matplotlib.pyplot as plt

        plt.show = lambda *args, **kwargs: None


def parse_common_args(
    *,
    defaults: PipelineConfig,
    kernel_defaults: KernelParameters,
    description: str,
    argv: Optional[Sequence[str]] = None,
    include_drl: bool = True,
) -> Tuple[PipelineConfig, KernelParameters, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--data-path", type=Path, default=defaults.data_path)
    parser.add_argument("--emission-file", default=defaults.emission_file)
    parser.add_argument("--guidance-file", default=defaults.guidance_file)
    parser.add_argument("--ground-truth-file", default=defaults.ground_truth_file, help="Ground truth image for NRMSE calculation (optional)")
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument(
        "--backend", choices=["auto", "numba"], default=defaults.backend
    )
    parser.add_argument("--save-suffix", default=defaults.save_suffix)
    parser.add_argument("--fwhm", type=float, nargs=3, default=list(defaults.fwhm), metavar=("FX", "FY", "FZ"))
    parser.add_argument("--psf-kernel-size", type=int, default=defaults.psf_kernel_size)
    parser.add_argument("--save-first-n", type=int, default=defaults.save_first_n, help="Save first N iterations before using interval")
    parser.add_argument("--use-uniform-initial", dest="use_uniform_initial", action="store_true", default=defaults.use_uniform_initial, help="Use uniform image as initial estimate instead of blurred image")
    parser.add_argument("--use-blurred-initial", dest="use_uniform_initial", action="store_false", help="Use blurred image as initial estimate (default)")

    parser.add_argument("--noise-seed", type=int, default=defaults.noise_seed)
    parser.add_argument("--bw-seed", type=int, default=defaults.bw_seed)
    parser.add_argument("--rl-iterations-standard", type=int, default=defaults.rl_iterations_standard)
    parser.add_argument("--rl-iterations-kernel", type=int, default=defaults.rl_iterations_kernel)
    parser.add_argument("--freeze-iteration", type=int, default=defaults.freeze_iteration)
    if include_drl:
        parser.add_argument("--dtv-iterations", type=int, default=defaults.dtv_iterations)
        parser.add_argument("--alpha", type=float, default=defaults.alpha)
        parser.add_argument("--step-size", type=float, default=defaults.step_size)
        parser.add_argument("--relaxation-eta", type=float, default=defaults.relaxation_eta)
        parser.add_argument("--update-obj-interval", type=int, default=defaults.update_obj_interval)
        parser.add_argument("--armijo-iterations", type=int, default=defaults.armijo_iterations, help="Maximum iteration number for Armijo line search (0 = disabled)")
        parser.add_argument("--armijo-update-initial", type=int, default=defaults.armijo_update_initial, help="Perform Armijo line search every iteration for first N iterations")
        parser.add_argument("--armijo-update-interval", type=int, default=defaults.armijo_update_interval, help="Perform Armijo line search every N iterations after initial period")
        parser.add_argument("--preconditioner-update-initial", type=int, default=defaults.preconditioner_update_initial, help="Update preconditioner every iteration for first N iterations")
        parser.add_argument("--preconditioner-update-interval", type=int, default=defaults.preconditioner_update_interval, help="Update preconditioner every N iterations after initial period")

    parser.add_argument("--show-plots", dest="show_plots", action="store_true", default=defaults.show_plots)
    parser.add_argument("--no-show-plots", dest="show_plots", action="store_false")
    parser.add_argument("--flip-emission", dest="flip_emission", action="store_true", default=defaults.flip_emission)
    parser.add_argument("--no-flip-emission", dest="flip_emission", action="store_false")
    parser.add_argument("--flip-guidance", dest="flip_guidance", action="store_true", default=defaults.flip_guidance)
    parser.add_argument("--no-flip-guidance", dest="flip_guidance", action="store_false")

    parser.add_argument("--enable-rl", dest="do_rl", action="store_true", default=defaults.do_rl)
    parser.add_argument("--disable-rl", dest="do_rl", action="store_false")
    parser.add_argument("--enable-krl", dest="do_krl", action="store_true", default=defaults.do_krl)
    parser.add_argument("--disable-krl", dest="do_krl", action="store_false")
    if include_drl:
        parser.add_argument("--enable-drl", dest="do_drl", action="store_true", default=defaults.do_drl)
        parser.add_argument("--disable-drl", dest="do_drl", action="store_false")
    else:
        parser.set_defaults(do_drl=False)

    # Kernel arguments
    parser.add_argument("--kernel-num-neighbours", type=int, default=kernel_defaults.num_neighbours)
    parser.add_argument("--kernel-sigma-anat", type=float, default=kernel_defaults.sigma_anat)
    parser.add_argument("--kernel-sigma-dist", type=float, default=kernel_defaults.sigma_dist)
    parser.add_argument("--kernel-sigma-emission", type=float, default=kernel_defaults.sigma_emission)
    parser.add_argument("--kernel-distance-weighting", dest="kernel_distance_weighting", action="store_true", default=kernel_defaults.distance_weighting)
    parser.add_argument("--kernel-no-distance-weighting", dest="kernel_distance_weighting", action="store_false")
    parser.add_argument("--kernel-normalize-features", dest="kernel_normalize_features", action="store_true", default=kernel_defaults.normalize_features)
    parser.add_argument("--kernel-dont-normalize-features", dest="kernel_normalize_features", action="store_false")
    parser.add_argument("--kernel-normalize", dest="kernel_normalize_kernel", action="store_true", default=kernel_defaults.normalize_kernel)
    parser.add_argument("--kernel-dont-normalize", dest="kernel_normalize_kernel", action="store_false")
    parser.add_argument("--kernel-use-mask", dest="kernel_use_mask", action="store_true", default=kernel_defaults.use_mask)
    parser.add_argument("--kernel-no-mask", dest="kernel_use_mask", action="store_false")
    parser.add_argument("--kernel-mask-k", type=_parse_optional_int, default=kernel_defaults.mask_k)
    parser.add_argument("--kernel-recalc-mask", dest="kernel_recalc_mask", action="store_true", default=kernel_defaults.recalc_mask)
    parser.add_argument("--kernel-no-recalc-mask", dest="kernel_recalc_mask", action="store_false")
    parser.add_argument("--kernel-hybrid", dest="kernel_hybrid", action="store_true", default=kernel_defaults.hybrid)
    parser.add_argument("--kernel-no-hybrid", dest="kernel_hybrid", action="store_false")

    args = parser.parse_args(argv)

    config = PipelineConfig(
        data_path=args.data_path,
        emission_file=args.emission_file,
        guidance_file=args.guidance_file,
        ground_truth_file=args.ground_truth_file,
        output_root=args.output_root,
        backend=args.backend,
        show_plots=args.show_plots,
        flip_emission=args.flip_emission,
        flip_guidance=args.flip_guidance,
        do_rl=args.do_rl,
        do_krl=args.do_krl,
        do_drl=args.do_drl,
        save_suffix=args.save_suffix,
        noise_seed=args.noise_seed,
        bw_seed=args.bw_seed,
        rl_iterations_kernel=args.rl_iterations_kernel,
        rl_iterations_standard=args.rl_iterations_standard,
        freeze_iteration=args.freeze_iteration,
        dtv_iterations=getattr(args, "dtv_iterations", defaults.dtv_iterations),
        alpha=getattr(args, "alpha", defaults.alpha),
        step_size=getattr(args, "step_size", defaults.step_size),
        relaxation_eta=getattr(args, "relaxation_eta", defaults.relaxation_eta),
        update_obj_interval=getattr(args, "update_obj_interval", defaults.update_obj_interval),
        armijo_iterations=getattr(args, "armijo_iterations", defaults.armijo_iterations),
        armijo_update_initial=getattr(args, "armijo_update_initial", defaults.armijo_update_initial),
        armijo_update_interval=getattr(args, "armijo_update_interval", defaults.armijo_update_interval),
        preconditioner_update_initial=getattr(args, "preconditioner_update_initial", defaults.preconditioner_update_initial),
        preconditioner_update_interval=getattr(args, "preconditioner_update_interval", defaults.preconditioner_update_interval),
        psf_kernel_size=args.psf_kernel_size,
        fwhm=_tuple3(args.fwhm),
        save_first_n=args.save_first_n,
        use_uniform_initial=args.use_uniform_initial,
    )

    kernel = KernelParameters(
        num_neighbours=args.kernel_num_neighbours,
        sigma_anat=args.kernel_sigma_anat,
        sigma_dist=args.kernel_sigma_dist,
        sigma_emission=args.kernel_sigma_emission,
        distance_weighting=args.kernel_distance_weighting,
        normalize_features=args.kernel_normalize_features,
        normalize_kernel=args.kernel_normalize_kernel,
        use_mask=args.kernel_use_mask,
        mask_k=args.kernel_mask_k,
        recalc_mask=args.kernel_recalc_mask,
        hybrid=args.kernel_hybrid,
    )

    return config, kernel, args
