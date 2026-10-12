"""Tests for the BrainWeb batch runner and the restored BrainWeb phantom helper.

No downloads and no network access: the external BrainWeb module is faked.

Note that the fake catalogue cannot verify parsing of the REAL upstream
catalogue format (get_files() return values / file naming) -- that remains
unverified without real BrainWeb access. Likewise the deterministic fake
cannot validate upstream RNG or cache behaviour.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = REPO_ROOT / "examples" / "scripts" / "brainweb_batch_runner.py"
PHANTOM_SOURCE = "examples/data/brainweb_phantoms.py"
CONFIG_DIR = REPO_ROOT / "examples" / "configs"

VOXEL_MM = (2.0, 2.5, 3.0)   # (z, y, x) mm

# Subject ids that exist in the real BrainWeb catalogue (non-consecutive).
REAL_CATALOGUE_IDS = [4, 5, 6, 18, 20, 38, *range(41, 55)]

FAKE_BRAINWEB_SOURCE = '''
import numpy as np

SHAPE = (5, 6, 7)  # (z, y, x) - distinct sizes per axis so axis order is observable
SUBJECT_IDS = [54, 55]
SEED_CALLS = []
LOADED = []


def get_files():
    return ["/cache/subject_%d.bin.gz" % sid for sid in SUBJECT_IDS]


def get_mmr_fromfile(path, **kwargs):
    sid = int(str(path).split("subject_")[1].split(".")[0])
    LOADED.append(sid)
    z, y, x = np.indices(SHAPE, dtype=np.float32)
    return {
        "PET": (1000.0 + 100.0 * sid + 4.0 * z + 2.0 * y + 1.0 * x).astype(np.float32),
        "T1": (10.0 * sid + 0.4 * z + 0.2 * y + 0.1 * x).astype(np.float32),
        "T2": (20.0 * sid + 40.0 * z + 20.0 * y + 10.0 * x).astype(np.float32),
        "uMap": (0.01 * (100.0 * z + 10.0 * y + x) + 0.5 * sid).astype(np.float32),
    }


def seed(value):
    SEED_CALLS.append(int(value))


def add_lesions(pet):
    out = np.array(pet, dtype=np.float32, copy=True)
    out[1, 2, 3] += 50.0
    return out
'''


def _load_runner():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import examples.scripts.brainweb_batch_runner as runner
    return runner


def _load_phantoms():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from examples.data.brainweb_phantoms import prepare_brainweb_pet_dataset
    return prepare_brainweb_pet_dataset


def _install_fake_brainweb(monkeypatch, tmp_path):
    path = tmp_path / "brainweb.py"
    path.write_text(FAKE_BRAINWEB_SOURCE)
    spec = importlib.util.spec_from_file_location("brainweb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "brainweb", module)
    return module


def _load_nifti(path):
    import nibabel as nib
    nii = nib.load(str(path))
    return np.asarray(nii.dataobj, dtype=np.float32), nii


def _written_subject_ids(out_dir, pattern="pet_gt_lesion_s*.nii.gz"):
    """Subject ids with written outputs, searching subdirectories too."""
    ids = []
    for p in out_dir.rglob(pattern):
        ids.append(int(p.name.rsplit("_s", 1)[1].split("_")[0].split(".")[0]))
    return sorted(ids)


def _dispatch_shipped_config(runner, config_path, monkeypatch):
    """Load a shipped YAML via the runner's own loader, then dispatch through main()."""
    cfg = runner.load_config(config_path)
    monkeypatch.setattr(sys, "argv", ["brainweb_batch_runner.py", "--config", str(config_path)])
    runner.main()
    return cfg


# ----------------------------------- CLI -------------------------------------

def test_cli_help_runs_without_brainweb(tmp_path):
    """--help must work when BrainWeb is unavailable and the cwd is not the repo."""
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "brainweb.py").write_text('raise RuntimeError("brainweb imported eagerly")\n')
    env = dict(os.environ)
    pypath = [str(blocker)]
    if env.get("PYTHONPATH"):
        pypath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pypath)

    proc = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--config" in proc.stdout
    assert "scenario" in proc.stdout


# ------------------------------ Scenario runners -----------------------------

@pytest.fixture
def stubbed_runner(monkeypatch):
    """Runner with the dataset helper replaced by a call recorder."""
    runner = _load_runner()
    calls = []

    def _record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(runner, "prepare_brainweb_pet_dataset", _record)
    return runner, calls


def test_cohort20_scenario_forwards_arguments(stubbed_runner, tmp_path):
    runner, calls = stubbed_runner
    cfg = {
        "out_root": str(tmp_path),
        "subject_ids": [4, 5],
        "fwhm_mm": 4,
        "mr_modality": "T2",
        "noise_model": "poisson",
        "poisson_scale": 1234,
        "seed": 42,
    }
    runner.run_cohort20_same_noise_fwhm(cfg)

    assert calls == [{
        "out_dir": tmp_path,
        "subject_ids": [4, 5],
        "mr_modality": "T2",
        "fwhm_mm": 4.0,
        "noise_model": "poisson",
        "poisson_scale": 1234.0,
        "gaussian_sigma": None,
        "replicate_subject_id": None,
        "n_realisations": 1,
        "seed": 42,
    }]


def test_multi_noise_realisations_scenario_forwards_arguments(stubbed_runner, tmp_path):
    runner, calls = stubbed_runner
    cfg = {
        "out_root": str(tmp_path),
        "patient_id": "54",
        "fwhm_mm": 5,
        "noise_model": "gaussian",
        "gaussian_sigma": 0.25,
        "n_realisations": 7,
        "seed": 2025,
    }
    runner.run_one_patient_multi_noise_realisations(cfg)

    assert calls == [{
        "out_dir": tmp_path,
        "subject_ids": [54],
        "mr_modality": "T1",
        "fwhm_mm": 5.0,
        "noise_model": "gaussian",
        "poisson_scale": 1e5,
        "gaussian_sigma": 0.25,
        "replicate_subject_id": 54,
        "n_realisations": 7,
        "seed": 2025,
    }]


def test_multi_fwhm_scenario_forwards_arguments(stubbed_runner, tmp_path):
    runner, calls = stubbed_runner
    cfg = {
        "out_root": str(tmp_path),
        "patient_id": 54,
        "fwhm_list_mm": [3.0, 6.5],
        "n_realisations": 4,   # this scenario always uses a single realisation
        "seed": 11,
    }
    runner.run_one_patient_multi_fwhm(cfg)

    assert [c["out_dir"] for c in calls] == [tmp_path / "fwhm_3.0mm", tmp_path / "fwhm_6.5mm"]
    assert [c["fwhm_mm"] for c in calls] == [3.0, 6.5]
    for kw in calls:
        assert kw["subject_ids"] == [54]
        assert kw["mr_modality"] == "T1"
        assert kw["noise_model"] == "poisson"
        assert kw["poisson_scale"] == 1e5
        assert kw["gaussian_sigma"] is None
        assert kw["replicate_subject_id"] is None
        assert kw["n_realisations"] == 1
        assert kw["seed"] == 11


def test_multi_noise_levels_scenario_forwards_arguments(stubbed_runner, tmp_path):
    runner, calls = stubbed_runner
    common = {
        "out_root": str(tmp_path),
        "patient_id": 54,
        "fwhm_mm": 6,
        "seed": 99,
    }

    runner.run_one_patient_multi_noise_levels(
        dict(common, noise_model="poisson", poisson_scale_list=[1000, 10000])
    )
    assert [c["out_dir"] for c in calls] == [
        tmp_path / "poisson_scale_1000", tmp_path / "poisson_scale_10000",
    ]
    assert [c["poisson_scale"] for c in calls] == [1000.0, 10000.0]
    for kw in calls:
        assert kw["noise_model"] == "poisson"
        assert kw["subject_ids"] == [54]
        assert kw["fwhm_mm"] == 6.0
        assert kw["replicate_subject_id"] is None
        assert kw["n_realisations"] == 1
        assert kw["seed"] == 99

    calls.clear()
    runner.run_one_patient_multi_noise_levels(
        dict(common, noise_model="gaussian", gaussian_sigma_list=[0.1, 0.25])
    )
    assert [c["out_dir"] for c in calls] == [
        tmp_path / "gaussian_sigma_0.100", tmp_path / "gaussian_sigma_0.250",
    ]
    assert [c["gaussian_sigma"] for c in calls] == [0.1, 0.25]
    for kw in calls:
        assert kw["noise_model"] == "gaussian"
        assert "poisson_scale" not in kw
        assert kw["replicate_subject_id"] is None
        assert kw["n_realisations"] == 1
        assert kw["seed"] == 99

    with pytest.raises(ValueError, match="noise_model"):
        runner.run_one_patient_multi_noise_levels(dict(common, noise_model="uniform"))


# --------------------------- Shipped YAML configs -----------------------------
#
# These tests load the four configs actually shipped in examples/configs and
# dispatch them end to end against a fake catalogue containing the documented
# real BrainWeb subject ids (the fake cannot verify the real catalogue format).

def test_shipped_cohort_config_selects_real_catalogue_subjects(monkeypatch, tmp_path):
    fake = _install_fake_brainweb(monkeypatch, tmp_path)
    fake.SUBJECT_IDS = list(REAL_CATALOGUE_IDS)
    runner = _load_runner()
    monkeypatch.chdir(tmp_path)

    cfg = _dispatch_shipped_config(runner, CONFIG_DIR / "cohort20_same_noise_fwhm.yaml", monkeypatch)
    assert cfg["scenario"] == "cohort20_same_noise_fwhm"
    assert cfg["subject_ids"] == REAL_CATALOGUE_IDS

    out = tmp_path / cfg["out_root"]
    assert _written_subject_ids(out) == sorted(REAL_CATALOGUE_IDS)
    assert sorted(set(fake.LOADED)) == sorted(REAL_CATALOGUE_IDS)
    # Single realisation per subject -> exactly four outputs each, no extras.
    assert len(list(out.glob("*.nii.gz"))) == 4 * len(REAL_CATALOGUE_IDS)


@pytest.mark.parametrize("scenario", [
    "one_patient_multi_fwhm",
    "one_patient_multi_noise_levels",
    "one_patient_multi_noise_realisations",
])
def test_shipped_one_patient_configs_select_subject_54(monkeypatch, tmp_path, scenario):
    fake = _install_fake_brainweb(monkeypatch, tmp_path)
    fake.SUBJECT_IDS = list(REAL_CATALOGUE_IDS)
    runner = _load_runner()
    monkeypatch.chdir(tmp_path)

    cfg = _dispatch_shipped_config(runner, CONFIG_DIR / f"{scenario}.yaml", monkeypatch)
    assert cfg["scenario"] == scenario
    assert cfg["patient_id"] == 54

    out = tmp_path / cfg["out_root"]
    # Some scenarios write one subject copy per subdirectory (fwhm / noise level).
    assert sorted(set(_written_subject_ids(out))) == [54]
    assert sorted(set(fake.LOADED)) == [54]


# --------------------------- Restored phantom helper -------------------------

def test_helper_added_noise_repeatability_for_fixed_upstream_volumes(monkeypatch, tmp_path):
    """Repeatability of the helper's ADDED noise given fixed upstream volumes.

    The fake BrainWeb source returns deterministic arrays, so this can only show
    that default_rng(seed) and the output writing are repeatable when the
    upstream volumes are identical. It says nothing about end-to-end BrainWeb
    reproducibility: upstream generation is not seeded by the helper and may be
    reused across caches, and the deterministic fake cannot exercise that.
    """
    fake = _install_fake_brainweb(monkeypatch, tmp_path)
    prepare = _load_phantoms()

    def run(out_name, seed):
        out = tmp_path / out_name
        prepare(
            out_dir=out,
            subject_ids=[54],
            fwhm_mm=5.0,
            noise_model="poisson",
            poisson_scale=1e5,
            replicate_subject_id=54,
            n_realisations=3,
            seed=seed,
            voxel_size_mm=VOXEL_MM,
        )
        return out

    out_a = run("run_a", 1337)
    out_b = run("run_b", 1337)
    out_c = run("run_c", 999)

    expected_names = [
        "ctlike_umap_s54.nii.gz",
        "mr_t1_s54.nii.gz",
        "pet_blur_noisy_poisson_s54_r00.nii.gz",
        "pet_blur_noisy_poisson_s54_r01.nii.gz",
        "pet_blur_noisy_poisson_s54_r02.nii.gz",
        "pet_gt_lesion_s54.nii.gz",
    ]
    assert sorted(p.name for p in out_a.iterdir()) == expected_names
    assert sorted(p.name for p in out_b.iterdir()) == expected_names
    assert sorted(p.name for p in out_c.iterdir()) == expected_names

    # Lesions were drawn by the real helper through the BrainWeb API.
    assert fake.SEED_CALLS == [1337, 1337, 999]
    vol = fake.get_mmr_fromfile("/cache/subject_54.bin.gz")
    pet_gt = fake.add_lesions(vol["PET"].astype(np.float32))

    # NIfTI stores (x, y, z); the helper receives (z, y, x).
    pet_gt_xyz, pet_gt_nii = _load_nifti(out_a / "pet_gt_lesion_s54.nii.gz")
    assert pet_gt_xyz.shape == (7, 6, 5)
    np.testing.assert_array_equal(pet_gt_xyz, np.transpose(pet_gt, (2, 1, 0)))
    mr_xyz, _ = _load_nifti(out_a / "mr_t1_s54.nii.gz")
    np.testing.assert_array_equal(mr_xyz, np.transpose(vol["T1"], (2, 1, 0)))
    umap_xyz, _ = _load_nifti(out_a / "ctlike_umap_s54.nii.gz")
    np.testing.assert_array_equal(umap_xyz, np.transpose(vol["uMap"], (2, 1, 0)))

    # Voxel spacing (z, y, x) -> NIfTI affine diagonal (x, y, z).
    np.testing.assert_allclose(pet_gt_nii.affine, np.diag([3.0, 2.5, 2.0, 1.0]))
    assert pet_gt_nii.header.get_zooms()[:3] == pytest.approx((3.0, 2.5, 2.0))

    for name in expected_names:
        same_seed = _load_nifti(out_a / name)[0]
        np.testing.assert_array_equal(same_seed, _load_nifti(out_b / name)[0])

    for name in expected_names:
        if not name.startswith("pet_blur_noisy_"):
            continue
        assert not np.array_equal(
            _load_nifti(out_a / name)[0], _load_nifti(out_c / name)[0]
        ), f"{name} should change with the noise seed"
    np.testing.assert_array_equal(
        _load_nifti(out_a / "pet_gt_lesion_s54.nii.gz")[0],
        _load_nifti(out_c / "pet_gt_lesion_s54.nii.gz")[0],
    )


def test_helper_blurs_with_scipy_and_counts_realisations(monkeypatch, tmp_path):
    from scipy.ndimage import gaussian_filter

    _install_fake_brainweb(monkeypatch, tmp_path)
    prepare = _load_phantoms()

    out = tmp_path / "out_blur"
    prepare(
        out_dir=out,
        subject_ids=[54, 55],
        fwhm_mm=5.0,
        noise_model="gaussian",
        gaussian_sigma=0.0,   # isolates the PSF blur: no random component added
        replicate_subject_id=55,
        n_realisations=2,
        seed=1337,
        voxel_size_mm=VOXEL_MM,
    )

    assert sorted(p.name for p in out.iterdir()) == [
        "ctlike_umap_s54.nii.gz",
        "ctlike_umap_s55.nii.gz",
        "mr_t1_s54.nii.gz",
        "mr_t1_s55.nii.gz",
        "pet_blur_noisy_gaussian_s54_r00.nii.gz",
        "pet_blur_noisy_gaussian_s55_r00.nii.gz",
        "pet_blur_noisy_gaussian_s55_r01.nii.gz",
        "pet_gt_lesion_s54.nii.gz",
        "pet_gt_lesion_s55.nii.gz",
    ]

    pet_gt_xyz = _load_nifti(out / "pet_gt_lesion_s54.nii.gz")[0]
    pet_gt_zyx = np.transpose(pet_gt_xyz, (2, 1, 0))
    const = 5.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma = (const / VOXEL_MM[0], const / VOXEL_MM[1], const / VOXEL_MM[2])
    expected = gaussian_filter(pet_gt_zyx, sigma=sigma, mode="constant", cval=0.0)

    noisy_xyz = _load_nifti(out / "pet_blur_noisy_gaussian_s54_r00.nii.gz")[0]
    np.testing.assert_allclose(noisy_xyz, np.transpose(expected, (2, 1, 0)), rtol=1e-6, atol=1e-6)
    assert not np.array_equal(noisy_xyz, pet_gt_xyz), "PSF blur should modify the volume"


def test_helper_missing_brainweb_dependency_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "brainweb", None)
    prepare = _load_phantoms()

    with pytest.raises(ImportError, match="pip install brainweb"):
        prepare(out_dir=tmp_path / "out", subject_ids=[54])


# ------------------------- Subject-selection validation -----------------------

def test_helper_partial_catalogue_match_lists_missing_id(monkeypatch, tmp_path):
    fake = _install_fake_brainweb(monkeypatch, tmp_path)
    fake.SUBJECT_IDS = [4]
    prepare = _load_phantoms()

    out = tmp_path / "out"
    with pytest.raises(ValueError) as excinfo:
        prepare(out_dir=out, subject_ids=[4, 7])

    assert "7" in str(excinfo.value)
    assert "4" not in str(excinfo.value)
    assert fake.LOADED == []
    assert not out.exists()


def test_helper_zero_catalogue_match_lists_missing_ids(monkeypatch, tmp_path):
    fake = _install_fake_brainweb(monkeypatch, tmp_path)
    fake.SUBJECT_IDS = [4]
    prepare = _load_phantoms()

    out = tmp_path / "out"
    with pytest.raises(ValueError) as excinfo:
        prepare(out_dir=out, subject_ids=[99, 100])

    message = str(excinfo.value)
    assert "99" in message and "100" in message
    assert fake.LOADED == []
    assert not out.exists()


def test_helper_empty_subject_ids_rejected(monkeypatch, tmp_path):
    _install_fake_brainweb(monkeypatch, tmp_path)
    prepare = _load_phantoms()

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="at least one"):
        prepare(out_dir=out, subject_ids=[])
    assert not out.exists()


def test_helper_none_subject_ids_selects_all_catalogue_subjects(monkeypatch, tmp_path):
    fake = _install_fake_brainweb(monkeypatch, tmp_path)
    fake.SUBJECT_IDS = [4, 54]
    prepare = _load_phantoms()

    out = tmp_path / "out"
    prepare(out_dir=out, subject_ids=None)

    assert _written_subject_ids(out) == [4, 54]
    assert sorted(set(fake.LOADED)) == [4, 54]


# ------------------------------------ Git ------------------------------------

def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_restored_phantom_source_visible_and_data_still_ignored():
    listed = _git("ls-files", "--cached", "--others", "--exclude-standard", "--", PHANTOM_SOURCE)
    assert listed.returncode == 0, listed.stderr
    assert listed.stdout.split() == [PHANTOM_SOURCE]

    assert _git("check-ignore", "-q", PHANTOM_SOURCE).returncode == 1
    assert _git("check-ignore", "-q", "examples/data/phantom_raw.nii.gz").returncode == 0
    assert _git("check-ignore", "-q", "data/phantom.bin").returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
