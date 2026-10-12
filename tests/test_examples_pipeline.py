"""Tests for the example sweep/pipeline code.

These tests exercise the crash-free paths of the example scripts and the
configuration defaults without running a real reconstruction (no data
downloads, no CIL display backend required).
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SWEEP_SCRIPT = REPO_ROOT / "examples" / "scripts" / "run_deconv_sweeps.py"

_SKIP_DIRS = {".git", "__pycache__", "build", "dist", ".pytest_cache", ".ruff_cache"}


def _run_sweep(args, cwd):
    return subprocess.run(
        [sys.executable, str(SWEEP_SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("run_deconv_sweeps_under_test", SWEEP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sweep_script_help_runs_without_env_flags(tmp_path):
    result = _run_sweep(["--help"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_sweep_script_dry_run_has_no_side_effects(tmp_path):
    result = _run_sweep(["--dry-run", "--max-runs", "1"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Dry-run complete" in result.stdout
    # No data was loaded (the referenced files do not exist), no output
    # directory or intermediate image was created.
    assert not (tmp_path / "results").exists()
    assert list(tmp_path.rglob("*.nii*")) == []


def test_sweep_non_dry_run_invokes_injected_runner(monkeypatch):
    module = _load_sweep_module()
    calls = []

    fake = types.ModuleType("examples.pipelines.run_deconv")
    fake.run_pipeline = lambda config, kernel: calls.append((config, kernel))
    monkeypatch.setitem(sys.modules, "examples.pipelines.run_deconv", fake)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_deconv_sweeps.py",
            "--datasets",
            "spheres",
            "--pipelines",
            "rl",
            "--max-runs",
            "1",
        ],
    )

    module.main()

    assert len(calls) == 1
    config, kernel = calls[0]
    assert config.emission_file == "phant_pet.nii"
    assert config.guidance_file == "phant_mri.nii"
    assert kernel.num_neighbours == 9


def test_default_config_uses_supported_nifti_paths():
    from examples.pipelines.config import KernelParameters, PipelineConfig, parse_common_args

    config, _, _ = parse_common_args(
        defaults=PipelineConfig(),
        kernel_defaults=KernelParameters(),
        description="test",
        argv=[],
    )

    assert config.emission_file == "OSEM_b1337_n5.nii.gz"
    assert config.guidance_file == "T1_b1337.nii.gz"
    assert str(config.emission_path()).endswith(".nii.gz")
    assert str(config.guidance_path()).endswith(".nii.gz")


def test_default_config_images_load_with_load_image(tmp_path):
    import nibabel as nib

    from examples.pipelines.config import KernelParameters, PipelineConfig, parse_common_args
    from krl.utils import load_image

    data_path = tmp_path / "data"
    data_path.mkdir()
    shape = (6, 7, 8)
    for name in ("OSEM_b1337_n5.nii.gz", "T1_b1337.nii.gz"):
        nib.save(
            nib.Nifti1Image(np.random.rand(*shape).astype(np.float32), np.eye(4)),
            str(data_path / name),
        )

    config, _, _ = parse_common_args(
        defaults=PipelineConfig(data_path=data_path),
        kernel_defaults=KernelParameters(),
        description="test",
        argv=[],
    )

    emission = load_image(config.emission_path())
    guidance = load_image(config.guidance_path())
    assert emission.as_array().shape == (shape[2], shape[1], shape[0])
    assert guidance.as_array().shape == (shape[2], shape[1], shape[0])


def test_psf_kernel_size_cli_option_is_rejected(capsys):
    from examples.pipelines.config import KernelParameters, PipelineConfig, parse_common_args

    with pytest.raises(SystemExit):
        parse_common_args(
            defaults=PipelineConfig(),
            kernel_defaults=KernelParameters(),
            description="test",
            argv=["--psf-kernel-size", "7"],
        )
    assert "psf-kernel-size" in capsys.readouterr().err


def test_invalid_sigma_raises_and_no_fallback():
    pytest.importorskip("matplotlib")
    from cil.framework import ImageGeometry

    from examples.pipelines import run_deconv
    from examples.pipelines.config import PipelineConfig

    geometry = ImageGeometry(voxel_num_x=7, voxel_num_y=7, voxel_num_z=7)
    image = geometry.allocate(1.0)

    with pytest.raises(ValueError):
        run_deconv.create_blurring_operator(PipelineConfig(fwhm=(0.0, 1.0, 1.0)), image)

    assert not hasattr(run_deconv, "psf")
    assert not hasattr(run_deconv, "geometry_voxel_sizes")


def test_pipeline_blur_matches_create_gaussian_blur():
    pytest.importorskip("matplotlib")
    from cil.framework import ImageGeometry

    from examples.pipelines import run_deconv
    from examples.pipelines.config import PipelineConfig
    from krl.operators.blurring import GaussianBlurringOperator, create_gaussian_blur

    geometry = ImageGeometry(
        voxel_num_x=9,
        voxel_num_y=8,
        voxel_num_z=7,
        voxel_size_x=2.0,
        voxel_size_y=2.5,
        voxel_size_z=3.5,
    )
    image = geometry.allocate(1.0)
    config = PipelineConfig(fwhm=(6.0, 6.0, 6.0))

    operator = run_deconv.create_blurring_operator(config, image)
    reference = create_gaussian_blur(run_deconv.fwhm_to_sigma(config.fwhm), geometry, backend="auto")

    assert isinstance(operator, GaussianBlurringOperator)
    assert operator.psf.shape == reference.psf.shape
    np.testing.assert_allclose(operator.psf, reference.psf)


def test_no_python_imports_reference_removed_modules():
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "cli_utils" or name.endswith(".cli_utils") or name.startswith("krl.cli"):
                    offenders.append((str(path.relative_to(REPO_ROOT)), name))

    assert offenders == []
    assert not (REPO_ROOT / "examples" / "pipelines" / "cli_utils.py").exists()
