import numpy as np
import pytest
from conftest import write_test_nifti

from krl_studies.methods.dtv import DTVMethod


def _as_cil(arr):
    import tempfile

    from krl.utils import load_nifti_as_imagedata

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        write_test_nifti(f.name, arr)
        return load_nifti_as_imagedata(f.name)


def _load(path):
    import nibabel as nib

    return np.transpose(nib.load(str(path)).get_fdata().astype(np.float32), (2, 1, 0))


def test_dtv_streams_iterates(tmp_path):
    obs = np.full((24, 32, 32), 1.0, dtype=np.float32)
    obs[8:16, 12:20, 12:20] = 4.0
    guidance = np.where(obs > 2.0, 2.0, 0.5).astype(np.float32)
    write_test_nifti(tmp_path / "o.nii", obs)
    write_test_nifti(tmp_path / "g.nii", guidance)

    iters = list(
        DTVMethod().run(
            observed=_as_cil(_load(tmp_path / "o.nii")),
            guidance=_as_cil(_load(tmp_path / "g.nii")),
            params={"alpha": 0.05, "fwhm_mm": 3.0, "lbfgs_max_linesearch": 5,
                    "lbfgs_ftol": 1e-6, "lbfgs_gtol": 1e-6},
            n_iterations=3,
        )
    )
    assert [it.iteration for it in iters] == [1, 2, 3]
    assert all(it.image.min() >= 0 for it in iters)
    assert all(it.objective is not None for it in iters)
    assert iters[-1].objective < iters[0].objective


def test_dtv_rejects_unknown_params(tmp_path):
    obs = np.full((8, 12, 12), 1.0, dtype=np.float32)
    write_test_nifti(tmp_path / "o.nii", obs)
    write_test_nifti(tmp_path / "g.nii", obs * 0.5)

    with pytest.raises(ValueError, match="unknown parameter"):
        DTVMethod().run(
            observed=_as_cil(_load(tmp_path / "o.nii")),
            guidance=_as_cil(_load(tmp_path / "g.nii")),
            params={"alpha": 0.05, "fwhm_mm": 3.0, "nonsense": 1},
            n_iterations=1,
        )


def test_dtv_requires_guidance(tmp_path):
    obs = np.full((8, 12, 12), 1.0, dtype=np.float32)
    write_test_nifti(tmp_path / "o.nii", obs)

    with pytest.raises(ValueError, match="guidance"):
        DTVMethod().run(
            observed=_as_cil(_load(tmp_path / "o.nii")),
            guidance=None,
            params={"alpha": 0.05, "fwhm_mm": 3.0},
            n_iterations=1,
        )
