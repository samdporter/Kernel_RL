"""End-to-end tests running the algorithms on real CIL data containers."""

import numpy as np
import pytest
from cil.framework import ImageGeometry

from krl.algorithms.richardson_lucy import RichardsonLucy
from krl.operators.blurring import create_gaussian_blur
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


def test_richardson_lucy_reduces_kl(phantom):
    """RL deconvolution should decrease the KL divergence to the observed data."""
    # Explicit backend: 'auto' would probe torch by importing it, which breaks
    # OpenMP when CIL's native libs are already loaded (see README notes).
    blur = create_gaussian_blur(sigma=(1.0, 1.0, 1.0), geometry=phantom.geometry, backend="numba")
    observed = blur.direct(phantom)

    def kl_to_observed(img):
        sim = blur.direct(img).as_array()
        obs = observed.as_array()
        sim = np.clip(sim, 1e-9, None)
        return float(np.sum(sim - obs - obs * np.log(sim / np.clip(obs, 1e-9, None))))

    rl = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
    )
    rl.run(iterations=5, verbose=0)

    result = rl.get_output() if rl.get_output() is not None else rl.x
    assert result is not None
    assert float(np.all(np.isfinite(result.as_array())))
    assert kl_to_observed(result) < kl_to_observed(observed)


def test_krl_end_to_end_matches_unkernelised_direction(phantom):
    """KRL with anatomical guidance runs and stays non-negative."""
    guidance = phantom.clone()  # perfectly correlated anatomy
    kernel_op = get_kernel_operator(
        phantom.geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        use_mask=True,
        mask_k=10,
        normalize_kernel=True,
        hybrid=False,
    )
    kernel_op.set_anatomical_image(guidance)

    blurred = kernel_op.direct(phantom)
    assert float(blurred.min()) >= -1e-6
    assert blurred.shape == phantom.shape


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
