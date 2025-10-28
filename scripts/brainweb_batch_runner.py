#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

# Optional YAML support
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from krl.utils import prepare_brainweb_pet_dataset


# ------------------------------ Scenario runners -----------------------------

def run_cohort20_same_noise_fwhm(cfg: dict) -> None:
    from pathlib import Path
    out_root = Path(cfg["out_root"])
    prepare_brainweb_pet_dataset(
        out_dir=out_root,
        subject_ids=cfg["subject_ids"],
        mr_modality=cfg.get("mr_modality", "T1"),
        fwhm_mm=float(cfg["fwhm_mm"]),
        noise_model=cfg.get("noise_model", "poisson"),
        poisson_scale=float(cfg.get("poisson_scale", 1e5)),
        gaussian_sigma=cfg.get("gaussian_sigma"),
        replicate_subject_id=None,
        n_realisations=1,
        seed=int(cfg.get("seed", 1337)),
    )


def run_one_patient_multi_noise_realisations(cfg: dict) -> None:
    from pathlib import Path
    out_root = Path(cfg["out_root"])
    pid = int(cfg["patient_id"])
    prepare_brainweb_pet_dataset(
        out_dir=out_root,
        subject_ids=[pid],
        mr_modality=cfg.get("mr_modality", "T1"),
        fwhm_mm=float(cfg["fwhm_mm"]),
        noise_model=cfg.get("noise_model", "poisson"),
        poisson_scale=float(cfg.get("poisson_scale", 1e5)),
        gaussian_sigma=cfg.get("gaussian_sigma"),
        replicate_subject_id=pid,                         # multiple noise realisations
        n_realisations=int(cfg.get("n_realisations", 10)),
        seed=int(cfg.get("seed", 2025)),
    )


def run_one_patient_multi_fwhm(cfg: dict) -> None:
    from pathlib import Path
    root = Path(cfg["out_root"])
    pid = int(cfg["patient_id"])
    fwhm_list = [float(x) for x in cfg["fwhm_list_mm"]]
    for fmm in fwhm_list:
        out_dir = root / f"fwhm_{fmm:.1f}mm"
        prepare_brainweb_pet_dataset(
            out_dir=out_dir,
            subject_ids=[pid],
            mr_modality=cfg.get("mr_modality", "T1"),
            fwhm_mm=fmm,
            noise_model=cfg.get("noise_model", "poisson"),
            poisson_scale=float(cfg.get("poisson_scale", 1e5)),
            gaussian_sigma=cfg.get("gaussian_sigma"),
            replicate_subject_id=None,
            n_realisations=1,
            seed=int(cfg.get("seed", 7)),
        )


def run_one_patient_multi_noise_levels(cfg: dict) -> None:
    from pathlib import Path
    root = Path(cfg["out_root"])
    pid = int(cfg["patient_id"])
    noise_model = cfg.get("noise_model", "poisson").lower()

    if noise_model == "poisson":
        scales = [float(x) for x in cfg["poisson_scale_list"]]
        for scale in scales:
            out_dir = root / f"poisson_scale_{int(scale):d}"
            prepare_brainweb_pet_dataset(
                out_dir=out_dir,
                subject_ids=[pid],
                mr_modality=cfg.get("mr_modality", "T1"),
                fwhm_mm=float(cfg["fwhm_mm"]),
                noise_model="poisson",
                poisson_scale=scale,
                replicate_subject_id=None,
                n_realisations=1,
                seed=int(cfg.get("seed", 99)),
            )
    elif noise_model == "gaussian":
        sigmas = [float(x) for x in cfg["gaussian_sigma_list"]]
        for sigma in sigmas:
            out_dir = root / f"gaussian_sigma_{sigma:.3f}"
            prepare_brainweb_pet_dataset(
                out_dir=out_dir,
                subject_ids=[pid],
                mr_modality=cfg.get("mr_modality", "T1"),
                fwhm_mm=float(cfg["fwhm_mm"]),
                noise_model="gaussian",
                gaussian_sigma=sigma,
                replicate_subject_id=None,
                n_realisations=1,
                seed=int(cfg.get("seed", 99)),
            )
    else:
        raise ValueError("noise_model must be 'poisson' or 'gaussian'.")


# --------------------------------- I/O utils ---------------------------------

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        with path.open("r") as f:
            return yaml.safe_load(f)
    with path.open("r") as f:
        return json.load(f)


# ----------------------------------- CLI -------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="BrainWeb PET batch runner (config-required). "
                    "Provide a YAML/JSON config containing a 'scenario' key with one of: "
                    "['cohort20_same_noise_fwhm', 'one_patient_multi_noise_realisations', "
                    "'one_patient_multi_fwhm', 'one_patient_multi_noise_levels']."
    )
    ap.add_argument("--config", type=str, required=True,
                    help="Path to YAML/JSON config file.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    scenario = cfg.get("scenario")
    runners = {
        "cohort20_same_noise_fwhm": run_cohort20_same_noise_fwhm,
        "one_patient_multi_noise_realisations": run_one_patient_multi_noise_realisations,
        "one_patient_multi_fwhm": run_one_patient_multi_fwhm,
        "one_patient_multi_noise_levels": run_one_patient_multi_noise_levels,
    }
    if scenario not in runners:
        raise ValueError(f"Unknown or missing scenario in config: {scenario}")

    runners[scenario](cfg)


if __name__ == "__main__":
    main()
