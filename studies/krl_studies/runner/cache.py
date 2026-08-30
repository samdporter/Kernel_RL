"""Cache each simulated input once per canonical identity.

The cache lives under ``{out_root}/input_cache/{input_id}/`` and stores:
- ``observed.nii.gz`` — the simulated observed image (CIL-compatible NIfTI)
- ``identity.json`` — the canonical identity used to build the key
- ``meta.json`` — the simulation metadata returned by ``simulate_inputs`` / ``quick_sim``
- ``data.sha256`` — sha256 of the on-disk NIfTI bytes

Writes are atomic: data and metadata are staged in a sibling temp directory and
renamed only after both files plus the checksum are written and verified.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "runner_cache_v1"


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _code_version() -> str:
    return f"cil-krl={_pkg_version('cil-krl')};krl-studies={_pkg_version('krl-studies')};schema={SCHEMA_VERSION}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_input_id(identity: dict) -> str:
    """Return the sha256 hex digest of the canonical JSON form of ``identity``.

    Raises ``TypeError`` if any value in ``identity`` is not JSON-serializable.
    """
    canonical = _canonical_json(identity)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_observed_sha256(arr: np.ndarray) -> str:
    """Return sha256 of ``arr.tobytes()`` together with the array's dtype."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(f"dtype={arr.dtype.str};shape={tuple(arr.shape)};".encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _write_nifti(arr: np.ndarray, path: Path) -> None:
    """Write a (z, y, x) float32 array as a NIfTI file with 1 mm voxels.

    Uses nibabel directly to avoid pulling in CIL/SIRF at import time so the
    rest of the cache module is usable in environments without those packages.
    """
    import nibabel as nib

    arr = np.asarray(arr, dtype=np.float32)
    data_xyz = np.transpose(arr, (2, 1, 0)) if arr.ndim == 3 else arr
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data_xyz, affine), str(path))


def cache_dir(out_root: Path | str) -> Path:
    return Path(out_root) / "input_cache"


def _source_checksums(run) -> dict[str, str]:
    """Hash dataset source files whose contents drive simulation output."""
    files: list[Path] = []
    if run.study == "spheres":
        root = Path(run.dataset.get("root", ""))
        if run.input_kind == "reference":
            candidate = root / "phant_pet.nii"
        else:
            candidate = root / "phant_orig.nii"
        if candidate.exists():
            files.append(candidate)
    elif run.study == "brainweb":
        root = Path(run.dataset.get("root", ""))
        subject = run.dataset.get("subject_id", run.dataset.get("subject"))
        if subject is not None:
            slug = str(subject)
            if not slug.startswith("subject_"):
                slug = f"subject_{slug}"
            candidate = root / slug / "pet_gt.nii.gz"
            if candidate.exists():
                files.append(candidate)
    out: dict[str, str] = {}
    for p in files:
        try:
            out[str(p)] = _sha256_file(p)
        except OSError:
            out[str(p)] = "unreadable"
    return out


def build_input_identity(run) -> dict:
    """Build the canonical input identity dict for ``run``.

    Excludes ``method_name`` and ``method_params`` on purpose so all methods
    sharing the same simulated acquisition share one cache entry.
    """
    identity = {
        "study": run.study,
        "dataset": dict(run.dataset),
        "input_kind": run.input_kind,
        "input_params": dict(run.input_params),
        "sim": dict(run.sim),
        "source_checksums": _source_checksums(run),
        "code_version": _code_version(),
    }
    return identity


def _read_nifti(path: Path) -> np.ndarray:
    import nibabel as nib

    nii = nib.load(str(path))
    data = nii.get_fdata().astype(np.float32)
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
    return data


def write_entry(
    out_root: Path | str,
    input_id: str,
    observed_array: np.ndarray,
    sim_meta: dict,
    *,
    identity: dict,
) -> None:
    """Atomically write a cache entry. Raises if any step fails; no partial entry.

    ``identity`` is the canonical identity dict used to build ``input_id``; it
    is stored alongside the entry so future reads can verify the entry matches
    the identity of a subsequent request.
    """
    out_root = Path(out_root)
    cd = cache_dir(out_root)
    cd.mkdir(parents=True, exist_ok=True)
    final_dir = cd / input_id
    if final_dir.exists():
        # Treat pre-existing entry as success — caller is re-using a known cache.
        return

    tmp_dir = cd / f"{input_id}.tmp.{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        nifti_path = tmp_dir / "observed.nii.gz"
        _write_nifti(observed_array, nifti_path)
        observed_bytes = nifti_path.read_bytes()
        data_sha = _sha256_bytes(observed_bytes)
        (tmp_dir / "data.sha256").write_text(data_sha + "\n")
        (tmp_dir / "meta.json").write_text(json.dumps(sim_meta, indent=2, default=str))
        (tmp_dir / "identity.json").write_text(_canonical_json(identity) + "\n")
        os_replace(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def os_replace(src: Path, dst: Path) -> None:
    """Atomic rename on the same filesystem. Public for monkeypatching in tests."""
    src.replace(dst)


def read_entry(
    out_root: Path | str,
    input_id: str,
    expected_identity: dict,
) -> tuple[np.ndarray, dict] | None:
    """Read and validate a cached entry.

    Returns ``None`` when the entry does not exist. Raises ``RuntimeError`` if
    the stored identity does not match ``expected_identity`` or if the stored
    checksum does not match the on-disk NIfTI bytes.
    """
    out_root = Path(out_root)
    entry_dir = cache_dir(out_root) / input_id
    nifti_path = entry_dir / "observed.nii.gz"
    sha_path = entry_dir / "data.sha256"
    meta_path = entry_dir / "meta.json"
    identity_path = entry_dir / "identity.json"
    if not (nifti_path.exists() and sha_path.exists() and meta_path.exists() and identity_path.exists()):
        return None

    stored_identity = json.loads(identity_path.read_text())
    expected_canonical = _canonical_json(expected_identity)
    stored_canonical = _canonical_json(stored_identity)
    if expected_canonical != stored_canonical:
        raise RuntimeError(
            f"input cache identity mismatch for {input_id}: "
            f"stored identity does not match the requested identity "
            f"(stored key={stored_canonical[:32]}…, requested key={expected_canonical[:32]}…)"
        )

    stored_sha = sha_path.read_text().strip()
    actual_sha = _sha256_bytes(nifti_path.read_bytes())
    if stored_sha != actual_sha:
        raise RuntimeError(
            f"input cache checksum mismatch for {input_id}: "
            f"stored sha256={stored_sha[:16]}…, actual sha256={actual_sha[:16]}… "
            "(observed.nii.gz is corrupt or has been modified)"
        )

    observed = _read_nifti(nifti_path)
    meta = json.loads(meta_path.read_text())
    return observed, meta
