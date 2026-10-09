"""End-to-end tests running the algorithms on real CIL data containers."""

import numpy as np
import pytest
from cil.framework import ImageGeometry
from cil.optimisation.functions import (
    L2NormSquared,
    OperatorCompositionFunction,
    SmoothMixedL21Norm,
)
from cil.optimisation.operators import CompositionOperator, GradientOperator

from krl.algorithms.lbfgsb import LBFGSBOptimizer, LBFGSBOptions
from krl.algorithms.maprl import MAPRL
from krl.algorithms.richardson_lucy import RichardsonLucy
from krl.callbacks import NRMSECallback, SaveIterationCallback
from krl.operators.blurring import create_gaussian_blur
from krl.operators.directional import DirectionalOperator
from krl.operators.kernel_operator import get_kernel_operator


@pytest.fixture
def geometry():
    return ImageGeometry(voxel_num_x=16, voxel_num_y=16, voxel_num_z=8)


@pytest.fixture
def phantom(geometry):
    """Smooth two-blob phantom."""
    x = np.zeros((8, 16, 16), dtype=np.float32)
    z, y, xx = np.indices(x.shape)
    x += 50 * np.exp(-((z - 2) ** 2 + (y - 5) ** 2 + (xx - 5) ** 2) / 4.0)
    x += 30 * np.exp(-((z - 5) ** 2 + (y - 10) ** 2 + (xx - 11) ** 2) / 6.0)
    img = geometry.allocate(0.0)
    img.fill(x)
    return img


@pytest.fixture
def blur(geometry):
    # Explicit backend: 'auto' would probe torch by importing it, which breaks
    # OpenMP when CIL's native libs are already loaded (see README notes).
    return create_gaussian_blur(sigma=(1.0, 1.0, 1.0), geometry=geometry, backend="numba")


@pytest.fixture
def observed(phantom, blur):
    return blur.direct(phantom)


def data_objective(operator, observed, image):
    """KL divergence between ``operator.direct(image)`` and the observed data."""
    sim = np.clip(operator.direct(image).as_array(), 1e-9, None)
    obs = observed.as_array()
    return float(np.sum(sim - obs - obs * np.log(sim / np.clip(obs, 1e-9, None))))


def make_kernel(geometry, anatomical, hybrid=False):
    kernel = get_kernel_operator(
        geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_emission=0.5,
        use_mask=True,
        mask_k=10,
        normalize_kernel=True,
        hybrid=hybrid,
    )
    kernel.set_anatomical_image(anatomical)
    return kernel


def make_dtv_prior(geometry, anatomical, observed, alpha=0.01):
    gradient = GradientOperator(geometry, method="forward", bnd_cond="Neumann")
    directional = CompositionOperator(DirectionalOperator(gradient.direct(anatomical)), gradient)
    return alpha * OperatorCompositionFunction(
        SmoothMixedL21Norm(epsilon=observed.max() * 1e-2), directional
    )


def test_richardson_lucy_reduces_kl(phantom, blur, observed):
    """RL deconvolution should decrease the KL divergence to the observed data."""
    rl = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
    )
    rl.run(iterations=5, verbose=0)

    result = rl.get_output()
    assert result is not None
    assert float(np.all(np.isfinite(result.as_array())))
    assert data_objective(blur, observed, result) < data_objective(blur, observed, observed)


def test_krl_end_to_end_reduces_data_objective(phantom, blur, observed):
    """Fixed-kernel KRL should decrease the data objective of the reconstruction."""
    kernel = make_kernel(phantom.geometry, phantom, hybrid=False)
    rl = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
        kernel_operator=kernel,
    )
    rl.run(iterations=5, verbose=0)

    result = rl.get_output()
    baseline = kernel.direct(observed)
    assert float(np.all(np.isfinite(result.as_array())))
    assert float(result.min()) >= -1e-6
    assert data_objective(blur, observed, result) < data_objective(blur, observed, baseline)


def test_hkrl_with_freezing_runs(phantom, blur, observed):
    """HKRL runs end-to-end and freezes its hybrid emission reference."""
    kernel = make_kernel(phantom.geometry, phantom, hybrid=True)
    rl = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
        kernel_operator=kernel,
        freeze_iteration=2,
    )
    rl.run(iterations=5, verbose=0)

    result = rl.get_output()
    assert rl._update_count == 5
    assert kernel.freeze_emission_kernel
    assert float(np.all(np.isfinite(result.as_array())))
    assert float(result.min()) >= -1e-6


def test_maprl_with_dtv_prior_runs(phantom, observed):
    """MAPRL runs end-to-end with a real DTV prior and a real CIL fidelity."""
    prior = make_dtv_prior(phantom.geometry, phantom, observed)
    maprl = MAPRL(
        initial_estimate=observed.clone(),
        data_fidelity=L2NormSquared(b=observed),
        prior=prior,
        step_size=1e-3,
        relaxation_eta=0.0,
        initial_line_search=False,
        armijo_iterations=0,
    )
    maprl.run(iterations=5, verbose=0)

    assert maprl._update_count == 5
    assert float(np.all(np.isfinite(maprl.x.as_array())))
    assert float(maprl.x.min()) >= -1e-6
    assert maprl.loss[-1] < maprl.loss[0]


def test_lbfgsb_reduces_objective(phantom, observed):
    """L-BFGS-B reduces a simple real-CIL objective."""
    prior = make_dtv_prior(phantom.geometry, phantom, observed)
    optimizer = LBFGSBOptimizer(
        initial_estimate=observed.clone(),
        data_fidelity=L2NormSquared(b=observed),
        prior=prior,
        options=LBFGSBOptions(ftol=1e-10, gtol=1e-10),
    )
    optimizer.run(iterations=15, verbose=0)

    assert float(np.all(np.isfinite(optimizer.solution.as_array())))
    assert optimizer.objective[-1] < optimizer.objective[0]


def test_callbacks_run_through_cil_algorithm(phantom, blur, observed, tmp_path):
    """NRMSE and SaveIteration callbacks run through a real CIL Algorithm.run()."""
    nrmse = NRMSECallback(
        phantom,
        tmp_path / "nrmse.csv",
        interval=1,
        verbose=False,
    )
    save = SaveIterationCallback(
        tmp_path / "iterations",
        interval=10,
        prefix="iter",
        save_first_n=2,
    )

    rl = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
    )
    rl.run(iterations=3, verbose=0, callbacks=[nrmse, save])

    assert len(nrmse.nrmse_values) == 4
    assert all(np.isfinite(value) for _, value in nrmse.nrmse_values)
    assert (tmp_path / "nrmse.csv").exists()

    saved = sorted((tmp_path / "iterations").glob("iter_*.nii.gz"))
    assert len(saved) == 3


def test_import_krl_does_not_pull_torch():
    """torch must stay optional: importing krl alone must not load it.

    Loading torch in the same process as CIL's native libraries breaks OpenMP
    on some platforms, so the core package must never import it eagerly.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "import krl;"
        "mods = list(sys.modules);"
        "assert 'torch' not in mods, 'krl imported torch';"
        "print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
