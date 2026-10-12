"""Tests for the ``examples/rl_krl_dtv.py`` deconvolution example.

The example is loaded from its source path for every test so that each test
gets a fresh module object and monkeypatching one test cannot leak into another.
"""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from cil.framework import ImageGeometry
from cil.optimisation.functions import OperatorCompositionFunction, SmoothMixedL21Norm
from cil.optimisation.operators import CompositionOperator, GradientOperator

from krl import DirectionalOperator, create_gaussian_blur

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "rl_krl_dtv.py"


def load_example():
    spec = importlib.util.spec_from_file_location("rl_krl_dtv_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example():
    return load_example()


def make_phantom(geometry):
    """Rebuild the example's synthetic phantom independently from the spec."""
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


def test_main_tiny_end_to_end(tmp_path, monkeypatch, example):
    monkeypatch.chdir(tmp_path)

    results = example.main(["--shape", "6", "8", "8", "--iterations", "2"])

    assert set(results) == {"RL", "KRL", "RL-dTV", "L-BFGS-B"}
    for name, result in results.items():
        reconstruction = result["reconstruction"]
        array = reconstruction.as_array()
        assert array.shape == (6, 8, 8), name
        assert np.all(np.isfinite(array)), name
        assert np.all(array >= -1e-6), name

        objectives = result["objectives"]
        assert all(np.isfinite(value) for value in objectives), name
        assert np.isfinite(result["nrmse_percent"]), name

        iterations = result["iterations"]
        assert 0 <= iterations <= 2, name
        if name != "L-BFGS-B":
            assert len(objectives) == 3, name
            assert iterations == 2, name

    assert "final_step_size" in results["RL-dTV"]
    assert np.isfinite(results["RL-dTV"]["final_step_size"])

    # The example must not write any files into the working directory.
    assert list(tmp_path.iterdir()) == []


def test_krl_reconstruction_is_kernel_applied(monkeypatch, example):
    created = []
    original = example.RichardsonLucy

    def recording(*args, **kwargs):
        instance = original(*args, **kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr(example, "RichardsonLucy", recording)
    results = example.main(["--shape", "6", "8", "8", "--iterations", "2"])

    krl = next(instance for instance in created if instance.kernel_operator is not None)
    returned = results["KRL"]["reconstruction"]

    np.testing.assert_allclose(returned.as_array(), krl.get_output().as_array())
    # Returning the latent image ``.x`` instead of ``get_output()`` must fail.
    assert not np.allclose(returned.as_array(), krl.x.as_array())


def test_rl_dtv_fidelity_includes_blur(monkeypatch, example):
    captured = {}
    original = example.MAPRL

    def recording(*args, **kwargs):
        instance = original(*args, **kwargs)
        captured["algorithm"] = instance
        return instance

    monkeypatch.setattr(example, "MAPRL", recording)
    example.main(["--shape", "6", "8", "8", "--iterations", "2"])

    fidelity = captured["algorithm"].data_fidelity
    raw_kl = fidelity.function
    blur = fidelity.operator
    observed = raw_kl.b

    perturbed = observed.geometry.allocate(0.0)
    perturbed.fill(observed.as_array() + 0.1)

    expected_value = float(raw_kl(blur.direct(perturbed)))
    assert float(fidelity(perturbed)) == pytest.approx(expected_value, rel=1e-5)
    # An uncomposed fidelity would evaluate the KL on the unblurred image.
    assert abs(float(fidelity(perturbed)) - float(raw_kl(perturbed))) > 1e-3

    expected_gradient = blur.adjoint(raw_kl.gradient(blur.direct(perturbed))).as_array()
    np.testing.assert_allclose(
        fidelity.gradient(perturbed).as_array(), expected_gradient, rtol=1e-5, atol=1e-6
    )


def test_prior_matches_independent_dtv_composition(monkeypatch, example):
    captured = {}
    original = example.MAPRL

    def recording(*args, **kwargs):
        instance = original(*args, **kwargs)
        captured["algorithm"] = instance
        return instance

    monkeypatch.setattr(example, "MAPRL", recording)
    example.main(["--shape", "6", "8", "8", "--iterations", "2"])

    geometry = ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=6, dtype=np.float32)
    ground_truth, anatomical = make_phantom(geometry)
    blur = create_gaussian_blur(sigma=(1, 1, 1), geometry=geometry, backend="numba")
    observed = blur.direct(ground_truth)

    gradient = GradientOperator(geometry, method="forward", bnd_cond="Neumann")
    anatomical_gradient = gradient.direct(anatomical)
    directional = DirectionalOperator(anatomical_gradient, gamma=1.0, eta=0.05)
    directional_gradient = CompositionOperator(directional, gradient)
    expected_prior = 0.01 * OperatorCompositionFunction(
        SmoothMixedL21Norm(epsilon=0.01 * float(observed.max())), directional_gradient
    )

    test_image = geometry.allocate(0.0)
    test_image.fill(np.random.default_rng(0).random(geometry.shape).astype(np.float32))

    captured_prior = captured["algorithm"].prior
    assert float(captured_prior(test_image)) == pytest.approx(
        float(expected_prior(test_image)), rel=1e-5, abs=1e-6
    )
    np.testing.assert_allclose(
        captured_prior.gradient(test_image).as_array(),
        expected_prior.gradient(test_image).as_array(),
        rtol=1e-4,
        atol=1e-4,
    )


def test_objectives_decrease_from_initial_to_final(example):
    results = example.main(["--shape", "6", "8", "8", "--iterations", "2"])

    for name, result in results.items():
        objectives = result["objectives"]
        tolerance = 1e-6 * max(abs(objectives[0]), 1.0)
        assert objectives[-1] <= objectives[0] - tolerance, (
            f"{name} objective did not decrease: {objectives[0]} -> {objectives[-1]}"
        )


def test_rl_dtv_preconditioner_is_sensitivity_normalised_em(monkeypatch, example):
    captured = {}
    original = example.MAPRL

    def recording(*args, **kwargs):
        instance = original(*args, **kwargs)
        captured["algorithm"] = instance
        return instance

    monkeypatch.setattr(example, "MAPRL", recording)
    example.main(["--shape", "6", "8", "8", "--iterations", "3"])

    algorithm = captured["algorithm"]
    preconditioner = algorithm.preconditioner
    assert callable(preconditioner)
    assert algorithm.preconditioner_update_initial == 0
    assert algorithm.preconditioner_update_interval == 1
    assert algorithm.initial_step_size == 1.0
    assert algorithm.armijo_iterations == 3

    geometry = ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=6, dtype=np.float32)
    blur = create_gaussian_blur(sigma=(1, 1, 1), geometry=geometry, backend="numba")
    sensitivity = np.maximum(blur.adjoint(geometry.allocate(1.0)).as_array(), 1e-6)

    for seed in (0, 1):
        image = geometry.allocate(0.0)
        image.fill(np.random.default_rng(seed).uniform(0.0, 1.0, geometry.shape).astype(np.float32))
        expected = (image.as_array() + 1e-6) / sensitivity
        result = preconditioner(image).as_array()
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)


def test_maprl_and_lbfgsb_share_fidelity_and_prior(monkeypatch, example):
    maprl_instances = []
    lbfgsb_instances = []
    original_maprl = example.MAPRL
    original_lbfgsb = example.LBFGSBOptimizer

    def recording_maprl(*args, **kwargs):
        instance = original_maprl(*args, **kwargs)
        maprl_instances.append(instance)
        return instance

    def recording_lbfgsb(*args, **kwargs):
        instance = original_lbfgsb(*args, **kwargs)
        lbfgsb_instances.append(instance)
        return instance

    monkeypatch.setattr(example, "MAPRL", recording_maprl)
    monkeypatch.setattr(example, "LBFGSBOptimizer", recording_lbfgsb)
    example.main(["--shape", "6", "8", "8", "--iterations", "2"])

    maprl = maprl_instances[0]
    lbfgsb = lbfgsb_instances[0]

    assert maprl.data_fidelity is lbfgsb.data_fidelity
    assert maprl.prior is lbfgsb.prior
    np.testing.assert_allclose(
        maprl.initial_estimate.as_array(), lbfgsb.initial_estimate.as_array()
    )

    maprl_objective = float(maprl.data_fidelity(maprl.initial_estimate)) + float(
        maprl.prior(maprl.initial_estimate)
    )
    lbfgsb_objective = float(lbfgsb.data_fidelity(lbfgsb.initial_estimate)) + float(
        lbfgsb.prior(lbfgsb.initial_estimate)
    )
    assert maprl_objective == pytest.approx(lbfgsb_objective, rel=1e-9, abs=1e-9)

    assert lbfgsb.options.max_linesearch == 20
    assert lbfgsb.options.ftol == 1e-6
    assert lbfgsb.options.gtol == 1e-6
    assert lbfgsb.options.enforce_non_negativity is True


def test_lbfgsb_progress_and_diagnostics(monkeypatch, example):
    captured = {}
    original = example.LBFGSBOptimizer

    def recording(*args, **kwargs):
        instance = original(*args, **kwargs)
        captured["optimizer"] = instance
        return instance

    monkeypatch.setattr(example, "LBFGSBOptimizer", recording)
    results = example.main(["--shape", "6", "8", "8", "--iterations", "3"])

    entry = results["L-BFGS-B"]
    array = entry["reconstruction"].as_array()
    assert np.all(np.isfinite(array))
    assert np.all(array >= -1e-6)

    objectives = entry["objectives"]
    assert all(np.isfinite(value) for value in objectives)
    assert objectives[-1] < objectives[0]

    result = captured["optimizer"].result
    assert 0 <= entry["iterations"] <= 3
    assert entry["iterations"] == result.nit
    assert entry["scipy_nfev"] == result.nfev
    assert entry["scipy_njev"] == result.njev
    assert entry["success"] == result.success
    assert entry["status"] == result.status
    assert entry["message"] == result.message


def test_default_case_rl_dtv_beats_blurred_baseline(example):
    results = example.main([])

    geometry = ImageGeometry(voxel_num_x=16, voxel_num_y=16, voxel_num_z=8, dtype=np.float32)
    ground_truth, _ = make_phantom(geometry)
    blur = create_gaussian_blur(sigma=(1, 1, 1), geometry=geometry, backend="numba")
    observed = blur.direct(ground_truth)
    observed_nrmse = float(
        100.0
        * np.sqrt(np.mean((observed.as_array() - ground_truth.as_array()) ** 2))
        / ground_truth.as_array().max()
    )

    assert results["RL-dTV"]["nrmse_percent"] <= 0.90 * observed_nrmse


def test_import_is_headless_and_does_not_run():
    code = textwrap.dedent(
        f"""
        import contextlib, importlib.abc, importlib.util, io, sys

        BLOCKED = {{"matplotlib", "torch", "pandas", "yaml"}}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError("blocked import: " + name)
                return None

        sys.meta_path.insert(0, Blocker())

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            spec = importlib.util.spec_from_file_location("rl_krl_dtv", {str(EXAMPLE_PATH)!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        assert buffer.getvalue() == "", repr(buffer.getvalue())
        assert callable(module.main)

        results = module.main(["--shape", "6", "8", "8", "--iterations", "2"])
        assert set(results) == {{"RL", "KRL", "RL-dTV", "L-BFGS-B"}}

        for blocked in BLOCKED:
            assert blocked not in sys.modules, blocked

        print("BLOCKED_IMPORT_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert "BLOCKED_IMPORT_OK" in result.stdout


def test_help_lists_cli_options():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "--iterations" in result.stdout
    assert "--shape" in result.stdout


def test_invalid_iterations_rejected(example):
    with pytest.raises(SystemExit) as excinfo:
        example.main(["--iterations", "0"])
    assert excinfo.value.code == 2


def test_singleton_shape_rejected(example):
    with pytest.raises(SystemExit) as excinfo:
        example.main(["--shape", "8", "16", "1"])
    assert excinfo.value.code == 2
