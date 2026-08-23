import zlib

import numpy as np
import pytest

from krl_studies.datasets.patients import PatientDataset, discover_subjects


def _mk(root, sid, with_roi=False):
    from conftest import write_test_nifti

    d = root / sid
    d.mkdir(parents=True)
    arr = np.random.default_rng(abs(zlib.crc32(sid.encode())) % 2**31).random((24, 28, 28)).astype("float32")
    write_test_nifti(d / "PET.nii.gz", arr)
    write_test_nifti(d / "T1.nii.gz", arr * 2)
    if with_roi:
        write_test_nifti(d / "ROIs.nii.gz", (arr > 0.5).astype("float32"))


def test_discover_finds_only_complete_subjects(tmp_path):
    _mk(tmp_path, "A")
    _mk(tmp_path, "B", with_roi=True)
    (tmp_path / "incomplete").mkdir()
    found = discover_subjects(tmp_path)
    assert set(found) == {"A", "B"}
    assert set(found["B"].keys()) == {"PET", "T1", "ROIs"}


def test_patient_dataset_loads_optional_roi(tmp_path):
    _mk(tmp_path, "MK-H001", with_roi=True)
    ds = PatientDataset(subject_id="MK-H001", root=tmp_path)
    assert ds.pet.shape == (24, 28, 28)
    assert ds.guidance.shape == (24, 28, 28)
    assert ds.rois is not None
    assert ds.ground_truth is None


def test_patient_dataset_without_roi(tmp_path):
    _mk(tmp_path, "C")
    ds = PatientDataset(subject_id="C", root=tmp_path)
    assert ds.rois is None
    assert ds.ground_truth is None


def test_missing_subject_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nobody"):
        PatientDataset(subject_id="nobody", root=tmp_path)
