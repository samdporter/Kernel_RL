"""BrainWeb dataset preparation tests (Task 4).

Requires ``brainweb`` pip package (lazy dep). Marked ``sirf`` per plan
so native macOS runs skip when the package (or network) is unavailable;
the ``brainweb`` marker is an alias. Pure unit test for
``regions_from_labels`` runs without brainweb.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

HAS_BRAINWEB = importlib.util.find_spec("brainweb") is not None

try:
    import nibabel  # noqa: F401

    HAS_NIB = True
except ImportError:
    HAS_NIB = False


# ------------------------------------------------------------------ voxel_mm test
def test_voxel_mm_comes_from_nifti_affine(tmp_path):
    import nibabel as nib

    from krl_studies.datasets.brainweb import BrainWebDataset

    # Create a synthetic subject with known voxel size
    custom_voxel = (2.1, 2.2, 2.3)  # z, y, x mm
    shape = (10, 20, 30)
    arr = np.ones(shape, dtype=np.float32)
    affine = np.diag([custom_voxel[2], custom_voxel[1], custom_voxel[0], 1.0]).astype(np.float32)
    subj_dir = tmp_path / "subject_99"
    subj_dir.mkdir(parents=True)
    for fname in [
        "pet_gt.nii.gz",
        "mr_t1_absent.nii.gz",
        "mr_t1_present.nii.gz",
        "mr_t2.nii.gz",
        "labels.nii.gz",
        "mu_map.nii.gz",
    ]:
        nii = nib.Nifti1Image(np.transpose(arr, (2, 1, 0)), affine)
        nib.save(nii, str(subj_dir / fname))
    # lesion files
    np.savez_compressed(subj_dir / "lesion_masks.npz", masks=np.zeros((0, *shape), dtype=bool))
    (subj_dir / "lesion_diameters_mm.json").write_text("[]")
    from krl_studies.datasets.brainweb import GUIDANCE_T1_INJECTION_RULE
    from krl_studies.datasets.lesions import mask_union_hash

    (subj_dir / "preparation.json").write_text(
        json.dumps(
            {
                "guidance": {
                    "injection_multiplier": 4.0,
                    "injection_rule": GUIDANCE_T1_INJECTION_RULE,
                    "mask_union_hash": mask_union_hash([]),
                }
            }
        )
    )

    ds = BrainWebDataset(tmp_path, 99)
    assert ds.voxel_mm == pytest.approx(custom_voxel, rel=1e-6), f"Expected {custom_voxel}, got {ds.voxel_mm}"

    # Round-trip save/load preserves affine
    reloaded = nib.load(str(subj_dir / "pet_gt.nii.gz"))
    reloaded_voxel = nib.affines.voxel_sizes(reloaded.affine)
    reloaded_voxel_zyx = (float(reloaded_voxel[2]), float(reloaded_voxel[1]), float(reloaded_voxel[0]))
    assert reloaded_voxel_zyx == pytest.approx(custom_voxel, rel=1e-6)


# ------------------------------------------------------------------ pure test
def test_regions_from_labels_partitions_brain():
    from krl_studies.datasets.brainweb import regions_from_labels

    # synthetic 10³ labels: BG=0, CSF=1, GM=2, WM=3
    labels = np.zeros((10, 10, 10), dtype=np.int16)
    labels[0:4, :, :] = 3  # WM
    labels[4:7, :, :] = 2  # GM
    labels[7:9, :, :] = 1  # CSF
    # last slice remains 0 BG

    masks = regions_from_labels(labels)
    assert len(masks) == 3
    wm, gm, rest = masks
    for m in masks:
        assert m.shape == labels.shape
        assert m.dtype == bool

    # disjoint
    assert not np.any(wm & gm)
    assert not np.any(wm & rest)
    assert not np.any(gm & rest)

    # cover brain (non-background) exactly when rest is CSF-only,
    # or cover whole volume when rest = CSF+BG - accept either
    brain = labels != 0
    union = wm | gm | rest
    # At minimum, brain must be covered
    assert np.all(union[brain])
    # Count sanity: WM+GM non-empty
    assert wm.sum() > 0
    assert gm.sum() > 0
    # rest covers at least CSF
    assert rest.sum() >= (labels == 1).sum()


def test_regions_from_labels_handles_random_labels():
    from krl_studies.datasets.brainweb import regions_from_labels

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=(8, 8, 8), dtype=np.int16)
    masks = regions_from_labels(labels)
    assert len(masks) == 3
    wm, gm, rest = masks
    assert not np.any(wm & gm)
    # union covers either brain or whole volume
    brain = labels != 0
    assert np.all((wm | gm | rest)[brain])


# ------------------------------------------------- synthetic subject fixtures
def _write_synthetic_subject(
    root,
    subject_id=99,
    *,
    shape=(16, 20, 24),
    voxel_mm=(2.0, 2.0, 2.0),
    n_lesions=4,
    bad_affine=None,
    bad_shape=None,
    nonfinite=None,
):
    import nibabel as nib

    from krl_studies.datasets.brainweb import (
        GUIDANCE_T1_INJECTION_RULE,
        build_guidance_t1_present,
    )
    from krl_studies.datasets.lesions import mask_union_hash

    subj = pathlib.Path(root) / f"subject_{subject_id:02d}"
    subj.mkdir(parents=True, exist_ok=True)

    brain = np.zeros(shape, dtype=np.float32)
    brain[3:13, 4:16, 5:19] = 1.0
    labels = brain * 3.0
    pet = brain * 2.0
    mu = labels * 0.1

    masks = np.zeros((n_lesions, *shape), dtype=bool)
    centres = [(5, 6, 8), (6, 8, 10), (8, 10, 12), (9, 12, 14)][:n_lesions]
    for i, (z, y, x) in enumerate(centres):
        masks[i, z, y, x] = True
        pet[z, y, x] = 8.0

    mask_list = [masks[i] for i in range(n_lesions)]
    t1_present = build_guidance_t1_present(brain, mask_list)

    assets = {
        "pet_gt.nii.gz": pet,
        "mr_t1_absent.nii.gz": brain,
        "mr_t1_present.nii.gz": t1_present,
        "mr_t2.nii.gz": brain * 0.8,
        "labels.nii.gz": labels,
        "mu_map.nii.gz": mu,
    }
    for fname, arr in assets.items():
        affine = np.diag([voxel_mm[2], voxel_mm[1], voxel_mm[0], 1.0]).astype(float)
        data = np.transpose(arr, (2, 1, 0))
        if bad_shape == fname:
            data = data[..., :-1]
        if bad_affine == fname:
            affine = np.diag([voxel_mm[2] * 2.0, voxel_mm[1], voxel_mm[0], 1.0]).astype(float)
        if nonfinite == fname:
            data = data.copy()
            data[0, 0, 0] = np.nan
        nib.save(nib.Nifti1Image(data, affine), str(subj / fname))

    np.savez_compressed(subj / "lesion_masks.npz", masks=masks)
    (subj / "lesion_diameters_mm.json").write_text(json.dumps([8, 12, 16, 24][:n_lesions]))
    (subj / "lesion_layout.json").write_text(
        json.dumps({"subject_id": str(subject_id), "layout_hash": "deadbeef"})
    )
    (subj / "preparation.json").write_text(
        json.dumps(
            {
                "guidance": {
                    "lesion_states": ["absent", "present"],
                    "injection_multiplier": 4.0,
                    "injection_rule": GUIDANCE_T1_INJECTION_RULE,
                    "mask_union_hash": mask_union_hash(mask_list),
                    "t1_variants": {"absent": "mr_t1_absent.nii.gz", "present": "mr_t1_present.nii.gz"},
                }
            }
        )
    )
    return subj


def test_dataset_loads_n_boolean_masks_and_layout(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    _write_synthetic_subject(tmp_path, 99, n_lesions=4)
    ds = BrainWebDataset(tmp_path, 99)
    assert isinstance(ds.lesion_masks, list)
    assert len(ds.lesion_masks) == 4
    for mask in ds.lesion_masks:
        assert mask.ndim == 3
        assert mask.dtype == bool
        assert mask.shape == ds.pet_gt.shape
    assert len(ds.lesion_diameters_mm) == 4
    assert ds.lesion_layout["layout_hash"] == "deadbeef"


def test_dataset_rejects_mismatched_affine(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    _write_synthetic_subject(tmp_path, 99, bad_affine="mr_t2.nii.gz")
    with pytest.raises(ValueError, match="affine"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_rejects_mismatched_shape(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    _write_synthetic_subject(tmp_path, 99, bad_shape="mu_map.nii.gz")
    with pytest.raises(ValueError, match="shape"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_rejects_nonfinite(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    _write_synthetic_subject(tmp_path, 99, nonfinite="labels.nii.gz")
    with pytest.raises(ValueError, match="non-finite"):
        BrainWebDataset(tmp_path, 99)


def _overwrite_lesion_files(subj, masks, diameters):
    np.savez_compressed(subj / "lesion_masks.npz", masks=np.asarray(masks))
    (subj / "lesion_diameters_mm.json").write_text(json.dumps(diameters))


def test_dataset_rejects_nan_masks(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=0)
    masks = np.zeros((1, 16, 20, 24), dtype=float)
    masks[0, 6, 8, 10] = np.nan
    _overwrite_lesion_files(subj, masks, [8])
    with pytest.raises(ValueError, match="non-finite"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_rejects_empty_masks(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=0)
    masks = np.zeros((1, 16, 20, 24), dtype=bool)
    _overwrite_lesion_files(subj, masks, [8])
    with pytest.raises(ValueError, match="empty"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_rejects_out_of_brain_masks(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=0)
    masks = np.zeros((1, 16, 20, 24), dtype=bool)
    masks[0, 0, 0, 0] = True
    _overwrite_lesion_files(subj, masks, [8])
    with pytest.raises(ValueError, match="contained"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_rejects_overlapping_masks(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=0)
    masks = np.zeros((2, 16, 20, 24), dtype=bool)
    masks[0, 6, 8, 10] = True
    masks[1, 6, 8, 10] = True
    _overwrite_lesion_files(subj, masks, [8, 8])
    with pytest.raises(ValueError, match="overlap"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_rejects_insufficient_separation(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=0)
    masks = np.zeros((2, 16, 20, 24), dtype=bool)
    masks[0, 6, 8, 10] = True
    masks[1, 6, 8, 11] = True
    _overwrite_lesion_files(subj, masks, [8, 8])
    with pytest.raises(ValueError, match="mm from another lesion"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_umap_guidance_is_a_separate_copy(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    _write_synthetic_subject(tmp_path, 99)
    ds = BrainWebDataset(tmp_path, 99)
    attenuation = ds.mu_map.copy()
    guidance = ds.guidance_for("umap")
    assert guidance is not ds.mu_map
    assert np.array_equal(guidance, attenuation)
    guidance[:] = 99.0
    assert np.array_equal(ds.mu_map, attenuation)


def test_dataset_guidance_for_modalities(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    _write_synthetic_subject(tmp_path, 99)
    ds = BrainWebDataset(tmp_path, 99)
    assert np.array_equal(ds.guidance_for("t1"), ds.mr_t1_absent)
    assert np.array_equal(ds.guidance_for("t1", "absent"), ds.mr_t1_absent)
    assert np.array_equal(ds.guidance_for("t1", "present"), ds.mr_t1_present)
    assert np.array_equal(ds.guidance_for("t2"), ds.mr_t2)
    with pytest.raises(ValueError):
        ds.guidance_for("bogus")
    with pytest.raises(ValueError):
        ds.guidance_for("t1", "bogus")


def test_dataset_t1_present_matches_union_rule(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=4)
    ds = BrainWebDataset(tmp_path, 99)
    union = np.zeros(ds.pet_gt.shape, dtype=bool)
    for mask in ds.lesion_masks:
        union |= mask
    assert np.allclose(ds.mr_t1_present[union], 4.0 * ds.mr_t1_absent[union])
    assert np.array_equal(ds.mr_t1_present[~union], ds.mr_t1_absent[~union])
    # exactly two T1 variant assets on disk
    variants = sorted(p.name for p in pathlib.Path(subj).glob("mr_t1_*.nii.gz"))
    assert variants == ["mr_t1_absent.nii.gz", "mr_t1_present.nii.gz"]


def test_build_guidance_t1_present_leaves_input_unchanged():
    from krl_studies.datasets.brainweb import build_guidance_t1_present

    t1 = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4) + 1.0
    mask_a = np.zeros((4, 4, 4), dtype=bool)
    mask_a[0, 0, 0] = True
    mask_b = np.zeros((4, 4, 4), dtype=bool)
    mask_b[3, 3, 3] = True
    snapshot = t1.copy()
    present = build_guidance_t1_present(t1, [mask_a, mask_b])
    assert np.array_equal(t1, snapshot)
    assert present[0, 0, 0] == 4.0 * t1[0, 0, 0]
    assert present[3, 3, 3] == 4.0 * t1[3, 3, 3]
    outside = ~(mask_a | mask_b)
    assert np.array_equal(present[outside], t1[outside])


def _patch_guidance(subj, **overrides):
    preparation = json.loads((subj / "preparation.json").read_text())
    preparation["guidance"].update(overrides)
    (subj / "preparation.json").write_text(json.dumps(preparation))


def test_dataset_rejects_wrong_injection_multiplier(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=4)
    _patch_guidance(subj, injection_multiplier=2.0)
    with pytest.raises(ValueError, match="injection_multiplier"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_rejects_wrong_injection_rule(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=4)
    _patch_guidance(subj, injection_rule="something else")
    with pytest.raises(ValueError, match="injection_rule"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_rejects_union_hash_mismatch(tmp_path):
    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=4)
    _patch_guidance(subj, mask_union_hash="0" * 64)
    with pytest.raises(ValueError, match="mask_union_hash"):
        BrainWebDataset(tmp_path, 99)


def test_dataset_rejects_present_variant_not_following_union_rule(tmp_path):
    import nibabel as nib

    from krl_studies.datasets.brainweb import BrainWebDataset

    subj = _write_synthetic_subject(tmp_path, 99, n_lesions=4)
    present = np.transpose(
        nib.load(str(subj / "mr_t1_present.nii.gz")).get_fdata(), (2, 1, 0)
    ).astype(np.float32)
    present += 1.0
    affine = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(np.transpose(present, (2, 1, 0)), affine), str(subj / "mr_t1_present.nii.gz"))
    with pytest.raises(ValueError, match="mr_t1_present"):
        BrainWebDataset(tmp_path, 99)


def test_texture_namespace_tracks_parameters():
    from krl_studies.datasets.brainweb import DEFAULT_TEXTURE, texture_namespace

    base = texture_namespace(DEFAULT_TEXTURE)
    assert texture_namespace(dict(DEFAULT_TEXTURE)) == base
    assert texture_namespace({**DEFAULT_TEXTURE, "petNoise": 0.5}) != base
    assert texture_namespace(DEFAULT_TEXTURE, seed=1) != base


def test_prepare_subjects_reports_incomplete_and_round_trips_inventory(tmp_path, monkeypatch):
    from krl_studies.datasets import brainweb as bw

    inventory = ("subject_04.bin.gz", "subject_05.bin.gz")
    monkeypatch.setattr(bw, "subject_inventory", lambda: inventory)

    def fake_prepare(subject_id, out_dir, tumour=True, *, seed=bw.DEFAULT_PREP_SEED, texture=None):
        if "05" in bw.normalize_subject_id(subject_id):
            raise RuntimeError("download failed")
        return {"pet_gt": pathlib.Path(out_dir) / "pet_gt.nii.gz"}, None

    monkeypatch.setattr(bw, "prepare_subject", fake_prepare)

    # subject_inventory() filenames round-trip straight into prepare_subjects.
    report = bw.prepare_subjects(list(bw.subject_inventory()), tmp_path)
    assert set(report["prepared"]) == {"subject_04.bin.gz"}
    assert report["incomplete"] == {"subject_05.bin.gz": "RuntimeError: download failed"}
    assert report["inventory"] == list(inventory)


def test_prepare_cli_writes_inventory_and_report(tmp_path, monkeypatch):
    from krl_studies import prepare as prep
    from krl_studies.datasets import brainweb as bw

    inventory = ("subject_04.bin.gz", "subject_05.bin.gz")
    monkeypatch.setattr(bw, "subject_inventory", lambda: inventory)
    captured = {}

    def fake_prepare_subjects(subject_ids, out_root, tumour=True, *, seed=bw.DEFAULT_PREP_SEED, texture=None):
        captured["tumour"] = tumour
        captured["subject_ids"] = list(subject_ids)
        return {
            "inventory": list(inventory),
            "prepared": {"subject_04.bin.gz": {}},
            "incomplete": {"subject_05.bin.gz": "RuntimeError: download failed"},
        }

    monkeypatch.setattr(bw, "prepare_subjects", fake_prepare_subjects)

    rc = prep.main(["--out-root", str(tmp_path)])
    assert rc == 1
    # The campaign CLI always prepares tumour-injected subjects.
    assert captured["tumour"] is True
    assert captured["subject_ids"] == list(inventory)
    inventory_doc = json.loads((tmp_path / "inventory.json").read_text())
    assert inventory_doc["inventory"] == list(inventory)
    assert inventory_doc["count"] == 2
    report = json.loads((tmp_path / "prep_report.json").read_text())
    assert report["incomplete"] == {"subject_05.bin.gz": "RuntimeError: download failed"}


def test_prepare_cli_rejects_tumour_free_option(tmp_path):
    from krl_studies import prepare as prep

    with pytest.raises(SystemExit):
        prep.main(["--out-root", str(tmp_path), "--no-tumour"])


@pytest.mark.brainweb
@pytest.mark.skipif(not HAS_BRAINWEB, reason="brainweb not installed")
def test_subject_inventory_matches_links():
    import brainweb

    from krl_studies.datasets.brainweb import subject_inventory

    assert subject_inventory() == tuple(sorted(brainweb.LINKS))


# ------------------------------------------------------------ download tests
@pytest.mark.sirf
@pytest.mark.brainweb
@pytest.mark.skipif(not HAS_BRAINWEB, reason="brainweb not installed")
@pytest.mark.skipif(not HAS_NIB, reason="nibabel not available")
def test_prepare_subject_writes_files(tmp_path):
    from krl_studies.datasets.brainweb import prepare_subject

    out = tmp_path / "subj04"
    try:
        paths, labels = prepare_subject(subject_id=4, out_dir=out, tumour=False)
    except Exception as exc:  # noqa: BLE001 - network / brainweb download failures
        msg = str(exc).lower()
        if any(k in msg for k in ("network", "download", "connection", "timeout", "url", "http", "get_file", "links")):
            pytest.skip(f"brainweb download unavailable: {exc}")
        # Also handle requests exceptions which may not contain those keywords
        if "brainweb" in msg or "requests" in msg:
            pytest.skip(f"brainweb unavailable: {exc}")
        raise

    # files exist
    assert pathlib.Path(paths["pet_gt"]).exists()
    assert pathlib.Path(paths["mr_t1_absent"]).exists()
    assert pathlib.Path(paths["mr_t1_present"]).exists()
    assert pathlib.Path(paths["labels"]).exists()
    assert pathlib.Path(paths["mu_map"]).exists()
    assert (out / "pet_gt.nii.gz").exists()
    assert (out / "pet_tumour_free.nii.gz").exists()
    assert (out / "mr_t1_absent.nii.gz").exists()
    assert (out / "mr_t1_present.nii.gz").exists()
    assert (out / "labels.nii.gz").exists()
    assert (out / "mu_map.nii.gz").exists()

    preparation = json.loads((out / "preparation.json").read_text())
    assert preparation["texture_namespace"]
    assert preparation["inventory_count"] >= 1
    assert len(preparation["inventory"]) == preparation["inventory_count"]
    assert preparation["guidance"]["injection_multiplier"] == 4.0
    assert preparation["guidance"]["mask_union_hash"]
    layout = json.loads((out / "lesion_layout.json").read_text())
    assert layout["layout_hash"]
    assert layout["centres_zyx"] == []

    # shapes & dtypes
    import nibabel as nib

    pet = np.transpose(nib.load(str(paths["pet_gt"])).get_fdata(), (2, 1, 0))
    t1 = np.transpose(nib.load(str(paths["mr_t1_absent"])).get_fdata(), (2, 1, 0))
    t1_present = np.transpose(nib.load(str(paths["mr_t1_present"])).get_fdata(), (2, 1, 0))
    lab = np.transpose(nib.load(str(paths["labels"])).get_fdata(), (2, 1, 0))
    mu = np.transpose(nib.load(str(paths["mu_map"])).get_fdata(), (2, 1, 0))
    assert pet.shape == t1.shape == t1_present.shape == lab.shape == labels.shape
    assert pet.ndim == 3
    # uMap shares the PET grid; values are mu in 1/cm (bone ~0.13, tissue ~0.096)
    assert mu.shape == pet.shape
    assert 0.0 < float(mu.max()) <= 0.15
    assert set(np.unique(labels).tolist()).issubset({0, 1, 2, 3})
    assert pet.max() > 0
    assert t1.max() > 0
    # tumour=False: no lesion masks, so the present variant equals the absent one.
    assert np.array_equal(t1, t1_present)


@pytest.mark.sirf
@pytest.mark.brainweb
@pytest.mark.skipif(not HAS_BRAINWEB, reason="brainweb not installed")
@pytest.mark.skipif(not HAS_NIB, reason="nibabel not available")
def test_prepare_subject_tumour_placement_respects_labels(tmp_path):
    from krl_studies.datasets.brainweb import prepare_subject, regions_from_labels

    out_tum = tmp_path / "tum" / "subject_04"
    out_tum2 = tmp_path / "tum2" / "subject_04"
    out_base = tmp_path / "base" / "subject_04"
    try:
        paths_tum, labels = prepare_subject(subject_id=4, out_dir=out_tum, tumour=True)
        paths_tum2, _ = prepare_subject(subject_id=4, out_dir=out_tum2, tumour=True)
        paths_base, _ = prepare_subject(subject_id=4, out_dir=out_base, tumour=False)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if any(k in msg for k in ("network", "download", "connection", "timeout", "url", "http", "get_file", "links")):
            pytest.skip(f"brainweb download unavailable: {exc}")
        if "brainweb" in msg or "requests" in msg:
            pytest.skip(f"brainweb unavailable: {exc}")
        raise

    masks = regions_from_labels(labels)
    assert len(masks) == 3
    wm, gm, _ = masks
    brain_tissue = wm | gm
    assert brain_tissue.sum() > 0

    lesion_array = np.load(out_tum / "lesion_masks.npz")["masks"]
    assert lesion_array.shape == (4, *labels.shape)
    assert all(lesion_array[i].any() for i in range(4))
    layout = json.loads((out_tum / "lesion_layout.json").read_text())
    assert len(layout["centres_zyx"]) == 4
    assert layout["layout_hash"]
    assert all(v > 0 for v in layout["realised_volumes_mm3"])
    assert (out_base / "pet_tumour_free.nii.gz").exists()

    # Exactly two T1 assets per subject; repeated preparation is bit-identical.
    import hashlib

    def _sha(path):
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

    variants = sorted(p.name for p in out_tum.glob("mr_t1_*.nii.gz"))
    assert variants == ["mr_t1_absent.nii.gz", "mr_t1_present.nii.gz"]
    assert _sha(paths_tum["mr_t1_absent"]) == _sha(paths_tum2["mr_t1_absent"])
    assert _sha(paths_tum["mr_t1_present"]) == _sha(paths_tum2["mr_t1_present"])
    prep_tum = json.loads((out_tum / "preparation.json").read_text())
    prep_tum2 = json.loads((out_tum2 / "preparation.json").read_text())
    assert prep_tum["guidance"]["mask_union_hash"] == prep_tum2["guidance"]["mask_union_hash"]
    assert prep_tum["guidance"]["injection_multiplier"] == 4.0

    # Loading the prepared subject re-validates masks and the injected-T1 rule.
    from krl_studies.datasets.brainweb import BrainWebDataset

    loaded = BrainWebDataset(tmp_path / "tum", 4)
    assert len(loaded.lesion_masks) == 4
    union = np.zeros(loaded.pet_gt.shape, dtype=bool)
    for mask in loaded.lesion_masks:
        union |= mask
    assert union.any()
    assert np.allclose(loaded.mr_t1_present[union], 4.0 * loaded.mr_t1_absent[union])
    assert np.array_equal(loaded.mr_t1_present[~union], loaded.mr_t1_absent[~union])
    assert loaded.guidance_mask_union_hash == prep_tum["guidance"]["mask_union_hash"]

    import nibabel as nib
    from scipy.ndimage import label as nd_label

    # Repeated preparation at the same seed/texture must be bit-identical.
    free_tum = np.transpose(nib.load(str(paths_tum["pet_tumour_free"])).get_fdata(), (2, 1, 0))
    free_base = np.transpose(nib.load(str(paths_base["pet_tumour_free"])).get_fdata(), (2, 1, 0))
    assert np.array_equal(free_tum, free_base)
    prep_base = json.loads((out_base / "preparation.json").read_text())
    assert prep_tum["texture_namespace"] == prep_base["texture_namespace"]

    pet_tum = np.transpose(nib.load(str(paths_tum["pet_gt"])).get_fdata(), (2, 1, 0))
    pet_base = np.transpose(nib.load(str(paths_base["pet_gt"])).get_fdata(), (2, 1, 0))
    assert pet_tum.shape == pet_base.shape == labels.shape
    # tumours increase PET by contrast factor 4; find lesion voxels where ratio >1.5
    # guard against zeros in pet_base
    ratio = np.divide(pet_tum, np.maximum(pet_base, 1e-6), out=np.zeros_like(pet_tum), where=pet_base > 1e-6)
    lesion_mask = ratio > 1.5
    assert lesion_mask.sum() > 0, "tumour placement did not increase PET"
    # lesions should be inside GM/WM tissue (allow small CSF spill due to interpolation)
    overlap = np.logical_and(lesion_mask, brain_tissue).sum()
    assert overlap / lesion_mask.sum() > 0.5, (
        f"only {overlap}/{lesion_mask.sum()} lesion voxels overlap GM/WM"
    )
    # check that major lesion components are near GM/WM (ignore tiny fragments)
    lab_arr, nlab = nd_label(lesion_mask)
    assert nlab >= 3, f"expected >=3 lesion components, got {nlab}"
    sizes = [(lab_arr == i).sum() for i in range(1, nlab + 1)]
    # consider only substantial lesions (>30 voxels)
    large_ids = [i for i, s in enumerate(sizes, start=1) if s > 30]
    assert len(large_ids) >= 2, f"expected >=2 substantial lesions, got {large_ids} sizes {sizes}"
    # at least 60% of large lesions should be near GM/WM
    near_brain = 0
    for i in large_ids:
        comp = lab_arr == i
        coords = np.argwhere(comp)
        cz, cy, cx = coords.mean(axis=0)
        iz, iy, ix = int(round(cz)), int(round(cy)), int(round(cx))
        iz = int(np.clip(iz, 0, labels.shape[0] - 1))
        iy = int(np.clip(iy, 0, labels.shape[1] - 1))
        ix = int(np.clip(ix, 0, labels.shape[2] - 1))
        z0, z1 = max(0, iz - 1), min(labels.shape[0], iz + 2)
        y0, y1 = max(0, iy - 1), min(labels.shape[1], iy + 2)
        x0, x1 = max(0, ix - 1), min(labels.shape[2], ix + 2)
        neigh = labels[z0:z1, y0:y1, x0:x1]
        if np.any((neigh == 2) | (neigh == 3)):
            near_brain += 1
        else:
            overlap_c = np.logical_and(comp, brain_tissue).sum()
            if overlap_c / comp.sum() > 0.2:
                near_brain += 1
    assert near_brain >= max(2, len(large_ids) * 0.6), (
        f"only {near_brain}/{len(large_ids)} large lesions near GM/WM"
    )


@pytest.mark.brainweb
@pytest.mark.skipif(not HAS_BRAINWEB, reason="brainweb not installed")
@pytest.mark.skipif(not HAS_NIB, reason="nibabel not available")
def test_prepare_cli_produces_injected_subject(tmp_path):
    import nibabel as nib

    from krl_studies import prepare as prep

    root = tmp_path / "data"
    rc = prep.main(["--out-root", str(root), "--subjects", "4"])
    report = json.loads((root / "prep_report.json").read_text())
    if report["incomplete"]:
        pytest.skip(f"brainweb unavailable: {report['incomplete']}")
    assert rc == 0
    assert set(report["prepared"]) == {"subject_04.bin.gz"}

    masks = np.load(root / "subject_04" / "lesion_masks.npz")["masks"]
    assert masks.shape[0] == 4
    assert all(masks[i].any() for i in range(4))
    assert (root / "subject_04" / "mr_t1_absent.nii.gz").exists()
    assert (root / "subject_04" / "mr_t1_present.nii.gz").exists()

    pet = np.transpose(nib.load(str(root / "subject_04" / "pet_gt.nii.gz")).get_fdata(), (2, 1, 0))
    base = np.transpose(
        nib.load(str(root / "subject_04" / "pet_tumour_free.nii.gz")).get_fdata(), (2, 1, 0)
    )
    assert np.any(pet > base), "campaign CLI must produce injected PET"
