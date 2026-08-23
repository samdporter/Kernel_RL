"""ROI construction: lesion ROIs from ground truth, background VOIs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def derive_lesion_rois(
    ground_truth: np.ndarray,
    threshold_fraction: float = 0.5,
    min_volume_vox: int = 20,
) -> list[np.ndarray]:
    """Connected components above a fraction of GT max, largest first."""
    gt_max = float(np.max(ground_truth))
    mask = ground_truth > threshold_fraction * gt_max
    labels, n = label(mask)
    rois = []
    for i in range(1, n + 1):
        comp = labels == i
        if int(comp.sum()) >= min_volume_vox:
            rois.append(comp)
    rois.sort(key=lambda m: int(m.sum()), reverse=True)
    return rois


def background_vois(
    shape: tuple[int, int, int],
    exclude_mask: np.ndarray,
    n_vois: int = 8,
    radius_vox: float = 5.0,
    margin_vox: int = 5,
    n_candidates: int = 4000,
    seed: int = 2026,
) -> list[np.ndarray]:
    """Pick n well-separated spherical background VOIs avoiding exclude_mask.

    Candidates are sampled uniformly (seeded), filtered by exclusion + margin,
    then greedily selected to maximise pairwise separation.
    """
    from krl_studies.datasets.lesions import sphere_mask

    rng = np.random.default_rng(seed)
    lo = np.array([margin_vox] * 3, dtype=float)
    hi = np.array(shape, dtype=float) - margin_vox
    candidates = rng.uniform(lo, hi, size=(n_candidates, 3))

    allowed = np.ones(shape, dtype=bool)
    allowed[:margin_vox, :, :] = False
    allowed[-margin_vox:, :, :] = False
    allowed[:, :margin_vox, :] = False
    allowed[:, -margin_vox:, :] = False
    allowed[:, :, :margin_vox] = False
    allowed[:, :, -margin_vox:] = False
    allowed &= ~exclude_mask

    chosen: list[tuple[np.ndarray, np.ndarray]] = []  # (centre, mask)
    min_sep = 4.0 * radius_vox
    for c in candidates:
        if len(chosen) >= n_vois:
            break
        if any(float(np.linalg.norm(c - c0)) < min_sep for c0, _ in chosen):
            continue
        m = sphere_mask(shape, tuple(c), radius_vox)
        if allowed[m].all():
            chosen.append((c, m))
    if len(chosen) < n_vois:
        raise ValueError(
            f"could only place {len(chosen)}/{n_vois} background VOIs; "
            "reduce n_vois or radius_vox"
        )
    return [m for _, m in chosen]
