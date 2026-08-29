import numpy as np
import pytest
from conftest import write_test_nifti

from krl_studies.datasets.spheres import quick_sim
from krl_studies.methods.base import Iterate
from krl_studies.methods.richardson_lucy import HKRLMethod, KRLMethod, RLMethod


@pytest.fixture(scope="module")
def observed_pair(tmp_path_factory):
    """GT with a hot cube; blurred+noisy observation; MR-ish guidance."""
    from pathlib import Path

    d = tmp_path_factory.mktemp("imgs")
    gt = np.full((32, 40, 40), 1.0, dtype=np.float32)
    gt[12:20, 16:24, 16:24] = 6.0
    obs = quick_sim(gt, fwhm_mm=3.0, counts=5e6, realisation=0, voxel_mm=(1.0, 1.0, 1.0))
    guidance = np.where(gt > 3.0, 2.0, 0.4).astype(np.float32)
    write_test_nifti(d / "gt.nii", gt)
    write_test_nifti(d / "obs.nii", obs)
    write_test_nifti(d / "mr.nii", guidance)
    return Path(d)


def _load(d, name):
    import nibabel as nib

    return np.transpose(nib.load(str(d / name)).get_fdata().astype(np.float32), (2, 1, 0))


def _as_cil(arr):
    import tempfile

    from krl.utils import load_nifti_as_imagedata

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        write_test_nifti(f.name, arr)
        return load_nifti_as_imagedata(f.name)


def test_base_iterate_fields():
    it = Iterate(iteration=1, image=np.zeros((2, 2, 2)), objective=None)
    assert it.iteration == 1 and it.image.shape == (2, 2, 2)


def test_rl_yields_stream_of_iterates_improving_nrmse(observed_pair):
    from krl_studies.metrics.nrmse import nrmse

    d = observed_pair
    gt, obs = _load(d, "gt.nii"), _load(d, "obs.nii")
    method = RLMethod()
    iters = list(
        method.run(
            observed=_as_cil(obs),
            guidance=None,
            params={"fwhm_mm": 3.0, "backend": "numba"},
            n_iterations=8,
        )
    )
    assert [it.iteration for it in iters] == list(range(1, 9))
    n_first = nrmse(iters[0].image, gt)
    n_min = min(nrmse(it.image, gt) for it in iters)
    assert n_min < n_first


def test_rl_lazy_generator_not_consumed_until_needed(observed_pair):
    d = observed_pair
    method = RLMethod()
    gen = method.run(observed=_as_cil(_load(d, "obs.nii")), guidance=None, params={"fwhm_mm": 3.0}, n_iterations=3)
    assert hasattr(gen, "__next__")


def test_krl_guidance_changes_result_vs_rl(observed_pair):
    d = observed_pair
    obs, mr = _load(d, "obs.nii"), _load(d, "mr.nii")
    common = {"fwhm_mm": 3.0, "backend": "numba"}
    rl = list(RLMethod().run(_as_cil(obs), None, common, 5))
    krl = list(
        KRLMethod().run(
            _as_cil(obs),
            _as_cil(mr),
            {**common, "sigma_anat": 1.0, "num_neighbours": 5},
            5,
        )
    )
    assert not np.allclose(rl[-1].image, krl[-1].image)


def test_hkrl_freeze_runs(observed_pair):
    d = observed_pair
    obs, mr = _load(d, "obs.nii"), _load(d, "mr.nii")
    hkrl = list(
        HKRLMethod().run(
            _as_cil(obs),
            _as_cil(mr),
            {
                "fwhm_mm": 3.0,
                "sigma_anat": 1.0,
                "sigma_emission": 1.0,
                "freeze_iteration": 2,
                "num_neighbours": 5,
                "backend": "numba",
            },
            4,
        )
    )
    assert len(hkrl) == 4


def test_krl_rejects_unknown_params(observed_pair):
    d = observed_pair
    obs, mr = _load(d, "obs.nii"), _load(d, "mr.nii")
    with pytest.raises(ValueError, match="unknown parameter"):
        KRLMethod().run(
            _as_cil(obs),
            _as_cil(mr),
            {"fwhm_mm": 3.0, "sigma_anat": 1.0, "sigma_anatomy": 2.0},
            1,
        )


def test_krl_requires_guidance(observed_pair):
    d = observed_pair
    obs = _load(d, "obs.nii")
    with pytest.raises(ValueError, match="guidance"):
        KRLMethod().run(_as_cil(obs), None, {"fwhm_mm": 3.0, "sigma_anat": 1.0}, 1)


def test_hkrl_hybrid_activation_and_difference_from_krl(observed_pair):
    d = observed_pair
    obs, mr = _load(d, "obs.nii"), _load(d, "mr.nii")
    common = {"fwhm_mm": 3.0, "backend": "numba", "sigma_anat": 1.0, "num_neighbours": 5}
    krl_iters = list(
        KRLMethod().run(
            _as_cil(obs), _as_cil(mr), {**common, "sigma_emission": 1.0, "freeze_iteration": 0}, 5
        )
    )
    hkrl_iters = list(
        HKRLMethod().run(
            _as_cil(obs), _as_cil(mr), {**common, "sigma_emission": 1.0, "freeze_iteration": 0}, 5
        )
    )
    # HKRL must produce different iterates than KRL when hybrid emission weight is active
    assert not np.allclose(krl_iters[-1].image, hkrl_iters[-1].image, rtol=1e-4)
    # All iterates finite and non-negative
    for it in hkrl_iters:
        assert np.all(np.isfinite(it.image))
        assert np.all(it.image >= 0)
