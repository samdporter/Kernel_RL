import json

import numpy as np
import pytest
import yaml
from conftest import write_test_nifti

from krl_studies.config import load_scenario_dict
from krl_studies.runner.cli import main
from krl_studies.runner.execute import execute_run
from krl_studies.runner.expand import expand_scenario


def _mini_scenario(tmp_path):
    # 64^3 so background_vois defaults fit: 8 VOIs of radius 5 need centres
    # >=20 voxels apart inside a 5-voxel margin.
    gt = np.full((64, 64, 64), 1.0, dtype=np.float32)
    gt[24:40, 24:40, 24:40] = 6.0
    d = tmp_path / "spheres"
    d.mkdir()
    write_test_nifti(d / "phant_orig.nii", gt)
    write_test_nifti(d / "phant_mri.nii", np.full_like(gt, 0.5))
    write_test_nifti(d / "phant_pet.nii", gt * 0.95)

    scenario = {
        "study": "spheres",
        "dataset": {"kind": "spheres", "root": str(d)},
        "inputs": [{"kind": "reference"}],
        "methods": [
            {"name": "post_smoothing", "params": {"sigma_mm": 2.0}},
        ],
        "sim": {"fwhm_mm": 3.0, "counts": 1e5},
        "output": str(tmp_path / "results"),
    }
    sp = tmp_path / "scen.yaml"
    sp.write_text(yaml.safe_dump(scenario))
    return sp


def test_execute_creates_manifest_metrics_marker(tmp_path):
    scenario = load_scenario_dict(yaml.safe_load(_mini_scenario(tmp_path).read_text()))
    run = expand_scenario(scenario)[0]
    assert run.sim == {"fwhm_mm": 3.0, "counts": 1e5}
    assert run.out_root == tmp_path / "results"

    out = execute_run(run)
    assert (out / ".done").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["method"] == "post_smoothing"
    assert manifest["status"] == "complete"
    csv_lines = (out / "metrics.csv").read_text().strip().splitlines()
    assert csv_lines[0].startswith("iteration")
    assert len(csv_lines) == 2  # header + single iterate
    assert (out / "final.nii.gz").exists()


def test_execute_skips_completed(tmp_path):
    scenario = load_scenario_dict(yaml.safe_load(_mini_scenario(tmp_path).read_text()))
    run = expand_scenario(scenario)[0]
    out1 = execute_run(run)
    marker1 = (out1 / ".done").read_text()
    out2 = execute_run(run, force=False)
    assert out1 == out2
    assert (out2 / ".done").read_text() == marker1


def test_cli_dry_run_lists_runs_without_executing(tmp_path, capsys):
    sp = _mini_scenario(tmp_path)
    rc = main(["--scenario", str(sp), "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "post_smoothing" in captured
    markers = list((tmp_path / "results").rglob(".done")) if (tmp_path / "results").exists() else []
    assert markers == []


def test_force_failure_removes_marker(tmp_path, monkeypatch):
    scenario = load_scenario_dict(yaml.safe_load(_mini_scenario(tmp_path).read_text()))
    run = expand_scenario(scenario)[0]
    out = execute_run(run)
    assert (out / ".done").exists()

    import krl_studies.runner.execute as ex

    def boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ex.METHOD_REGISTRY["post_smoothing"], "run", boom)
    with pytest.raises(RuntimeError):
        execute_run(run, force=True)
    assert not (out / ".done").exists()


def _write_brainweb_subject(root, subject_id=99, shape=(64, 64, 64)):
    from conftest import write_test_nifti

    from krl_studies.datasets.brainweb import (
        GUIDANCE_T1_INJECTION_RULE,
        build_guidance_t1_present,
    )
    from krl_studies.datasets.lesions import mask_union_hash

    subj = root / f"subject_{subject_id:02d}"
    subj.mkdir(parents=True, exist_ok=True)

    brain = np.zeros(shape, dtype=np.float32)
    brain[8:56, 8:56, 8:56] = 1.0
    labels = brain * 3.0
    pet = brain * 1.0
    masks = np.zeros((4, *shape), dtype=bool)
    for i, (z, y, x) in enumerate([(20, 20, 20), (20, 30, 30), (30, 20, 36), (30, 36, 24)]):
        masks[i, z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = True
        pet[masks[i]] = 4.0

    mask_list = [masks[i] for i in range(4)]
    write_test_nifti(subj / "pet_gt.nii.gz", pet)
    write_test_nifti(subj / "mr_t1_absent.nii.gz", brain)
    write_test_nifti(subj / "mr_t1_present.nii.gz", build_guidance_t1_present(brain, mask_list))
    write_test_nifti(subj / "mr_t2.nii.gz", brain * 0.75)
    write_test_nifti(subj / "labels.nii.gz", labels)
    write_test_nifti(subj / "mu_map.nii.gz", labels * 0.1)
    np.savez_compressed(subj / "lesion_masks.npz", masks=masks)
    (subj / "lesion_diameters_mm.json").write_text("[8, 12, 16, 24]")
    (subj / "lesion_layout.json").write_text('{"layout_hash": "fixture"}')
    (subj / "preparation.json").write_text(
        json.dumps(
            {
                "guidance": {
                    "injection_multiplier": 4.0,
                    "injection_rule": GUIDANCE_T1_INJECTION_RULE,
                    "mask_union_hash": mask_union_hash(mask_list),
                    "t1_variants": {
                        "absent": "mr_t1_absent.nii.gz",
                        "present": "mr_t1_present.nii.gz",
                    },
                }
            }
        )
    )
    return subj


def _brainweb_run(tmp_path, **overrides):
    from krl_studies.config import RunSpec

    params = {
        "run_id": "test_brainweb_lesion",
        "study": "brainweb",
        "dataset": {"kind": "brainweb", "root": str(tmp_path), "subject_id": 99},
        "input_kind": "quick_sim",
        "input_params": {"fwhm_mm": 3.0, "counts": 1e5, "realisation": 0},
        "method_name": "post_smoothing",
        "method_params": {"sigma_mm": 1.0},
        "sim": {"seed": 7},
        "out_root": str(tmp_path / "results"),
    }
    params.update(overrides)
    return RunSpec(**params)


def test_runner_brainweb_lesion_truth_fix(tmp_path):
    """Real smoke test: n persisted lesion masks must run without ambiguity errors."""
    _write_brainweb_subject(tmp_path)
    run = _brainweb_run(tmp_path)
    out_dir = execute_run(run, force=True)
    assert out_dir.exists()
    assert (out_dir / ".done").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["guidance_modality"] == "t1"
    header = (out_dir / "metrics.csv").read_text().splitlines()[0]
    assert "crc_mm8" in header
    assert "crc_mm24" in header


def test_brainweb_guidance_modality_selection(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset
    from krl_studies.datasets.transforms import apply_guidance_condition
    from krl_studies.runner.execute import _prepare_guidance

    _write_brainweb_subject(tmp_path)
    ds = BrainWebDataset(tmp_path, 99)
    attenuation_before = ds.mu_map.copy()

    # T1 lesion-state selection.
    absent_run = _brainweb_run(
        tmp_path,
        input_params={"fwhm_mm": 3.0, "counts": 1e5, "realisation": 0, "guidance_lesion_state": "absent"},
    )
    assert np.array_equal(
        _prepare_guidance(absent_run, ds, ds.voxel_mm, ds.guidance), ds.mr_t1_absent
    )
    present_run = _brainweb_run(
        tmp_path,
        input_params={"fwhm_mm": 3.0, "counts": 1e5, "realisation": 0, "guidance_lesion_state": "present"},
    )
    assert np.array_equal(
        _prepare_guidance(present_run, ds, ds.voxel_mm, ds.guidance), ds.mr_t1_present
    )
    assert not np.array_equal(ds.mr_t1_absent, ds.mr_t1_present)

    t2_run = _brainweb_run(
        tmp_path, input_params={"fwhm_mm": 3.0, "counts": 1e5, "realisation": 0, "guidance_condition": "t2"}
    )
    assert np.array_equal(_prepare_guidance(t2_run, ds, ds.voxel_mm, ds.guidance), ds.mr_t2)

    umap_run = _brainweb_run(
        tmp_path,
        input_params={
            "fwhm_mm": 3.0,
            "counts": 1e5,
            "realisation": 0,
            "guidance_modality": "umap",
            "guidance_condition": "shift_p2",
        },
    )
    shifted = _prepare_guidance(umap_run, ds, ds.voxel_mm, ds.guidance)
    expected = apply_guidance_condition(ds.mu_map, "shift_p2", ds.voxel_mm, order=1)
    assert np.array_equal(shifted, expected)
    # The attenuation array must be untouched by guidance conditioning.
    assert np.array_equal(ds.mu_map, attenuation_before)

    t2_shift_run = _brainweb_run(
        tmp_path,
        input_params={
            "fwhm_mm": 3.0,
            "counts": 1e5,
            "realisation": 0,
            "guidance_modality": "t2",
            "guidance_condition": "shift_m5",
        },
    )
    t2_shifted = _prepare_guidance(t2_shift_run, ds, ds.voxel_mm, ds.guidance)
    assert np.array_equal(t2_shifted, apply_guidance_condition(ds.mr_t2, "shift_m5", ds.voxel_mm, order=1))


def _make_spheres_run(tmp_path, *, input_kind, input_params, sim, method_name="post_smoothing", method_params=None):
    import uuid

    gt = np.full((64, 64, 64), 1.0, dtype=np.float32)
    gt[24:40, 24:40, 24:40] = 6.0
    # Unique sub-directory so a test can build multiple runs against one tmp_path.
    d = tmp_path / f"spheres_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    write_test_nifti(d / "phant_orig.nii", gt)
    write_test_nifti(d / "phant_mri.nii", np.full_like(gt, 0.5))
    write_test_nifti(d / "phant_pet.nii", gt * 0.95)
    from krl_studies.config import RunSpec

    return RunSpec(
        run_id="test_propagation",
        study="spheres",
        dataset={"kind": "spheres", "root": str(d)},
        input_kind=input_kind,
        input_params=input_params,
        method_name=method_name,
        method_params=method_params or {"sigma_mm": 2.0},
        sim=sim,
        out_root=str(tmp_path / "results"),
    ), d


def test_sim_values_propagate_to_simulate_inputs(tmp_path, monkeypatch):
    """scenario sim (scanner/seed/n_subits) must reach simulate_inputs via _build_observed."""
    from krl_studies.runner import execute as ex

    captured = {}

    def fake_simulate_inputs(gt_array, cfg_dict):
        captured["cfg"] = dict(cfg_dict)
        return np.zeros_like(gt_array), {"scanner": cfg_dict.get("scanner", "Siemens mMR")}

    # Patch in the simulate namespace because _build_observed lazy-imports it
    monkeypatch.setattr("krl_studies.simulation.simulate.simulate_inputs", fake_simulate_inputs)

    run, d = _make_spheres_run(
        tmp_path,
        input_kind="sirf_sim",
        input_params={
            "condition": "psf-matched",
            "beta": None,
            "counts": 1e5,
            "realisation": 0,
        },
        sim={"scanner": "Siemens VISION 600", "seed": 42, "n_subits": 7},
    )
    ex._build_observed(run, _spheres_ds(d), np.zeros((64, 64, 64), dtype=np.float32))
    assert captured["cfg"]["scanner"] == "Siemens VISION 600"
    assert captured["cfg"]["seed"] == 42
    assert captured["cfg"]["n_subits"] == 7


def test_input_params_override_scenario_sim_for_scanner_seed_subits(tmp_path, monkeypatch):
    """input_params overrides scenario sim for scanner, seed, and n_subits."""
    from krl_studies.runner import execute as ex

    captured = {}

    def fake_simulate_inputs(gt_array, cfg_dict):
        captured["cfg"] = dict(cfg_dict)
        return np.zeros_like(gt_array), {}

    monkeypatch.setattr("krl_studies.simulation.simulate.simulate_inputs", fake_simulate_inputs)

    run, d = _make_spheres_run(
        tmp_path,
        input_kind="sirf_sim",
        input_params={
            "condition": "psf-matched",
            "beta": None,
            "counts": 1e5,
            "realisation": 0,
            "scanner": "Siemens mMR",  # explicit override
            "seed": 999,  # explicit override
            "n_subits": 1,  # explicit override
        },
        sim={"scanner": "Siemens VISION 600", "seed": 42, "n_subits": 7},
    )
    ex._build_observed(run, _spheres_ds(d), np.zeros((64, 64, 64), dtype=np.float32))
    assert captured["cfg"]["scanner"] == "Siemens mMR"
    assert captured["cfg"]["seed"] == 999
    assert captured["cfg"]["n_subits"] == 1


def test_scenario_sim_n_subiterations_alias_propagates(tmp_path, monkeypatch):
    """n_subiterations alias in scenario sim must reach simulate_inputs."""
    from krl_studies.runner import execute as ex

    captured = {}

    def fake_simulate_inputs(gt_array, cfg_dict):
        captured["cfg"] = dict(cfg_dict)
        return np.zeros_like(gt_array), {}

    monkeypatch.setattr("krl_studies.simulation.simulate.simulate_inputs", fake_simulate_inputs)

    run, d = _make_spheres_run(
        tmp_path,
        input_kind="sirf_sim",
        input_params={
            "condition": "psf-matched",
            "beta": None,
            "counts": 1e5,
            "realisation": 0,
        },
        sim={"scanner": "Siemens mMR", "seed": 5, "n_subiterations": 3},
    )
    ex._build_observed(run, _spheres_ds(d), np.zeros((64, 64, 64), dtype=np.float32))
    # cfg may carry either n_subits or n_subiterations; the test must be tolerant.
    n_subits = captured["cfg"].get("n_subits", captured["cfg"].get("n_subiterations"))
    assert n_subits == 3
    assert captured["cfg"]["seed"] == 5


def test_quick_sim_receives_scenario_seed(tmp_path, monkeypatch):
    """quick_sim path must receive scenario sim seed (overridable by input_params)."""
    from krl_studies.runner import execute as ex

    captured = {}

    def fake_quick_sim(gt_array, fwhm_mm, counts, realisation, voxel_mm, seed=1337):
        captured["seed"] = seed
        return np.zeros_like(gt_array)

    # execute.py imports quick_sim at module load, so patch its bound name.
    monkeypatch.setattr(ex, "quick_sim", fake_quick_sim)

    run, d = _make_spheres_run(
        tmp_path,
        input_kind="quick_sim",
        input_params={"fwhm_mm": 3.0, "counts": 1e5, "realisation": 0},
        sim={"seed": 1234},
    )
    ex._build_observed(run, _spheres_ds(d), np.zeros((64, 64, 64), dtype=np.float32))
    assert captured["seed"] == 1234

    captured.clear()
    run2, d2 = _make_spheres_run(
        tmp_path,
        input_kind="quick_sim",
        input_params={"fwhm_mm": 3.0, "counts": 1e5, "realisation": 0, "seed": 77},
        sim={"seed": 1234},
    )
    ex._build_observed(run2, _spheres_ds(d2), np.zeros((64, 64, 64), dtype=np.float32))
    assert captured["seed"] == 77


def _spheres_ds(root):
    """Return a SphereDataset without touching the filesystem."""
    from krl_studies.datasets.spheres import SphereDataset

    return SphereDataset(root=root)


def test_get_acquisition_does_not_silently_fallback_to_mmr(monkeypatch):
    """Requested scanner failure must propagate, never silently swap to mMR."""
    from krl_studies.simulation import _api
    from krl_studies.simulation import simulate as sim_mod

    sim_mod._ACQ_CACHE.clear()
    try:
        def dispatch(name, **kwargs):
            if name == "Siemens mMR":
                return f"acq_{name}"
            raise RuntimeError(f"unsupported scanner: {name!r}")

        monkeypatch.setattr(_api, "acquisition_template", dispatch)

        # Vision request must fail even though mMR would build successfully.
        with pytest.raises(RuntimeError, match="unsupported scanner"):
            sim_mod._get_acquisition("Siemens VISION 600")
        # And no mMR entry was smuggled into the cache as a substitute.
        assert "Siemens mMR" not in sim_mod._ACQ_CACHE
    finally:
        sim_mod._ACQ_CACHE.clear()


def test_get_acquisition_cache_keyed_by_scanner(monkeypatch):
    """Cache is keyed by the exact scanner name; mMR and Vision are distinct entries."""
    from krl_studies.simulation import _api
    from krl_studies.simulation import simulate as sim_mod

    sim_mod._ACQ_CACHE.clear()
    try:
        seen = []

        def fake(name, **_kwargs):
            seen.append(name)
            return f"acq_{name}"

        monkeypatch.setattr(_api, "acquisition_template", fake)

        acq1, used1 = sim_mod._get_acquisition("Siemens mMR")
        acq2, used2 = sim_mod._get_acquisition("Siemens VISION 600")
        assert used1 == "Siemens mMR"
        assert used2 == "Siemens VISION 600"
        assert acq1 == "acq_Siemens mMR"
        assert acq2 == "acq_Siemens VISION 600"

        # second call uses the cache, not _api.acquisition_template
        seen.clear()
        sim_mod._get_acquisition("Siemens mMR")
        assert seen == []
    finally:
        sim_mod._ACQ_CACHE.clear()


# ---------------------------------------------------------------------------
# Task 3: input cache tests
# ---------------------------------------------------------------------------


def test_input_cache_reuses_identical_simulation(tmp_path, monkeypatch):
    """Two method runs sharing the same simulation identity must reuse one cached entry."""
    from krl_studies.config import RunSpec
    from krl_studies.runner import cache as cache_mod
    from krl_studies.runner import execute as ex

    run, d = _make_spheres_run(
        tmp_path,
        input_kind="sirf_sim",
        input_params={
            "condition": "psf-matched",
            "beta": None,
            "counts": 1e5,
            "realisation": 0,
        },
        sim={"seed": 1},
        method_name="post_smoothing",
        method_params={"sigma_mm": 2.0},
    )
    # Same simulation identity, different method. Reuse a unique out_root per test.
    out_root = tmp_path / "results"
    run_a = RunSpec(
        run_id="cache_a",
        study=run.study,
        dataset=run.dataset,
        input_kind=run.input_kind,
        input_params=run.input_params,
        method_name="post_smoothing",
        method_params={"sigma_mm": 2.0},
        sim=run.sim,
        out_root=out_root,
    )
    run_b = RunSpec(
        run_id="cache_b",
        study=run.study,
        dataset=run.dataset,
        input_kind=run.input_kind,
        input_params=run.input_params,
        method_name="post_smoothing",
        method_params={"sigma_mm": 4.0},  # different method params must not change cache key
        sim=run.sim,
        out_root=out_root,
    )

    call_count = {"n": 0}

    def fake_simulate_inputs(gt_array, cfg_dict):
        call_count["n"] += 1
        return np.zeros_like(gt_array), {
            "scanner": cfg_dict.get("scanner", "Siemens mMR"),
            "seed": cfg_dict.get("seed", 0),
        }

    monkeypatch.setattr("krl_studies.simulation.simulate.simulate_inputs", fake_simulate_inputs)

    gt = np.zeros((64, 64, 64), dtype=np.float32)
    obs_a, meta_a = ex._build_observed(run_a, _spheres_ds(d), gt)
    obs_b, meta_b = ex._build_observed(run_b, _spheres_ds(d), gt)

    assert call_count["n"] == 1, "second identical-simulation run must hit the cache"
    assert np.array_equal(obs_a, obs_b)
    assert meta_a == meta_b

    identity_a = cache_mod.build_input_identity(run_a)
    identity_b = cache_mod.build_input_identity(run_b)
    assert cache_mod.compute_input_id(identity_a) == cache_mod.compute_input_id(identity_b)

    expected_sha = cache_mod.compute_observed_array_sha256(obs_a)
    assert cache_mod.compute_observed_array_sha256(obs_b) == expected_sha

    cached = cache_mod.read_entry(out_root, cache_mod.compute_input_id(identity_a), identity_a)
    assert cached is not None
    cached_obs, cached_meta = cached
    assert np.array_equal(cached_obs, obs_a)
    assert cached_meta == meta_a


def test_input_cache_identity_excludes_method(tmp_path):
    """Method name and method params must not change the input identity."""
    from krl_studies.config import RunSpec
    from krl_studies.runner import cache as cache_mod

    base = _make_spheres_run(
        tmp_path,
        input_kind="sirf_sim",
        input_params={"condition": "psf-matched", "beta": None, "counts": 1e5, "realisation": 0},
        sim={"seed": 1},
        method_name="post_smoothing",
        method_params={"sigma_mm": 2.0},
    )[0]
    other = RunSpec(
        run_id=base.run_id,
        study=base.study,
        dataset=base.dataset,
        input_kind=base.input_kind,
        input_params=base.input_params,
        method_name="rl",
        method_params={"fwhm_mm": 5.0, "iterations": 3},
        sim=base.sim,
        out_root=base.out_root,
    )
    id_a = cache_mod.compute_input_id(cache_mod.build_input_identity(base))
    id_b = cache_mod.compute_input_id(cache_mod.build_input_identity(other))
    assert id_a == id_b

    differing_seed = RunSpec(
        run_id=base.run_id,
        study=base.study,
        dataset=base.dataset,
        input_kind=base.input_kind,
        input_params=base.input_params,
        method_name=base.method_name,
        method_params=base.method_params,
        sim={**base.sim, "seed": 999},
        out_root=base.out_root,
    )
    id_c = cache_mod.compute_input_id(cache_mod.build_input_identity(differing_seed))
    assert id_c != id_a


def test_input_cache_rejects_identity_mismatch(tmp_path):
    """Reading under a different identity must raise instead of returning stale data."""
    from krl_studies.runner import cache as cache_mod

    out_root = tmp_path / "results"
    arr = np.full((4, 4, 4), 0.5, dtype=np.float32)
    identity_a = {"study": "spheres", "input_kind": "sirf_sim", "seed": 1, "code_version": "v1"}
    identity_b = {"study": "spheres", "input_kind": "sirf_sim", "seed": 2, "code_version": "v1"}
    input_id = cache_mod.compute_input_id(identity_a)
    cache_mod.write_entry(out_root, input_id, arr, {"prompt_scale": 1.0}, identity=identity_a)

    with pytest.raises(RuntimeError, match="identity"):
        cache_mod.read_entry(out_root, input_id, identity_b)


def test_input_cache_rejects_corrupt_entry(tmp_path, monkeypatch):
    """A stored entry with a checksum mismatch must raise rather than be returned."""
    from krl_studies.runner import cache as cache_mod

    out_root = tmp_path / "results"
    arr = np.full((4, 4, 4), 0.5, dtype=np.float32)
    identity = {"study": "spheres", "input_kind": "sirf_sim", "seed": 1, "code_version": "v1"}
    input_id = cache_mod.compute_input_id(identity)
    cache_mod.write_entry(out_root, input_id, arr, {"prompt_scale": 1.0}, identity=identity)

    # Corrupt the stored NIfTI by overwriting its content with garbage.
    cached_path = cache_mod.cache_dir(out_root) / input_id / "observed.nii.gz"
    cached_path.write_bytes(b"corrupt garbage not a nifti")

    with pytest.raises(RuntimeError, match="checksum"):
        cache_mod.read_entry(out_root, input_id, identity)


def test_input_cache_atomic_write(tmp_path, monkeypatch):
    """If the write_image step raises, no partial cache entry must remain."""
    from krl_studies.runner import cache as cache_mod

    out_root = tmp_path / "results"
    arr = np.full((4, 4, 4), 0.5, dtype=np.float32)
    identity = {"study": "spheres", "input_kind": "sirf_sim", "seed": 1, "code_version": "v1"}
    input_id = cache_mod.compute_input_id(identity)

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cache_mod, "_write_nifti", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        cache_mod.write_entry(out_root, input_id, arr, {"prompt_scale": 1.0}, identity=identity)

    final_dir = cache_mod.cache_dir(out_root) / input_id
    assert not final_dir.exists(), f"partial entry left behind: {final_dir}"
    # And no leftover *.tmp.* siblings.
    leftovers = [p for p in cache_mod.cache_dir(out_root).glob(f"{input_id}.tmp.*")]
    assert leftovers == [], f"tmp siblings left behind: {leftovers}"
