import numpy as np
import pytest

from krl_studies.methods.baselines import PostSmoothingMethod
from krl_studies.methods.petpvc import GTMMethod, build_petpvc_cmd


def test_post_smoothing_blurs_and_single_iterate():
    img = np.zeros((16, 20, 20), dtype=np.float32)
    img[8, 10, 10] = 100.0
    iters = list(
        PostSmoothingMethod().run(
            observed=img, guidance=None,
            params={"sigma_mm": 2.0, "voxel_mm": (1.0, 1.0, 1.0)}, n_iterations=1,
        )
    )
    assert len(iters) == 1 and iters[0].iteration == 1
    assert iters[0].image.max() < img.max()


def test_post_smoothing_rejects_multi_iteration():
    with pytest.raises(ValueError, match="n_iterations=1"):
        list(PostSmoothingMethod().run(observed=np.zeros((4, 4, 4)), guidance=None,
                                       params={"sigma_mm": 2.0}, n_iterations=2))


def test_build_petpvc_cmd_shape():
    cmd = build_petpvc_cmd(
        petpvc_bin="petpvc",
        input_path="in.nii", output_path="out.nii",
        mode="GTM", pvc_fwhm=(5.0, 5.0, 5.0), mask_path="mask.nii",
        extra=["--reg", "rois.nii"],
    )
    assert cmd[0] == "petpvc"
    assert "-i" in cmd and "in.nii" in cmd
    assert "-o" in cmd and "out.nii" in cmd
    assert "-p" in cmd and "GTM" in cmd
    assert "-f" in cmd and "5p0" in ",".join(cmd) or "5.0" in ",".join(cmd)
    assert "--reg" in cmd


def test_gtm_missing_binary_raises(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="PETPVC"):
        list(GTMMethod().run(
            observed=np.zeros((4, 4, 4)), guidance=None,
            params={"petpvc_bin": "definitely-not-a-binary", "input_path": "a.nii",
                    "output_path": "b.nii"},
            n_iterations=1,
        ))
