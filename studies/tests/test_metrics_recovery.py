import numpy as np

from krl_studies.datasets.lesions import sphere_mask
from krl_studies.metrics.recovery import background_variability, crc_percent


def _world():
    gt = np.full((40, 40, 40), 1.0, dtype=np.float32)
    lesion = sphere_mask((40, 40, 40), (20.0, 20.0, 20.0), 4.0)
    gt[lesion] = 5.0
    return gt, lesion


def test_crc_perfect_for_ground_truth():
    gt, lesion = _world()
    vois = [sphere_mask((40, 40, 40), (c, 20.0, 20.0), 3.0) for c in (5.0, 35.0)]
    assert crc_percent(lesion, gt, gt, vois) == 100.0


def test_crc_zero_when_no_recovery():
    gt, lesion = _world()
    flat = np.full_like(gt, 2.0)  # constant image: measured ratio = 1
    vois = [sphere_mask((40, 40, 40), (c, 20.0, 20.0), 3.0) for c in (5.0, 35.0)]
    assert crc_percent(lesion, flat, gt, vois) == 0.0


def test_background_variability_zero_for_constant():
    const = np.full((20, 20, 20), 3.0, dtype=np.float32)
    vois = [sphere_mask((20, 20, 20), (c, 10.0, 10.0), 2.0) for c in (5.0, 10.0, 15.0)]
    assert background_variability(const, vois) == 0.0


def test_background_variability_positive_for_spread(rng):
    img = rng.normal(10.0, 1.0, size=(30, 30, 30)).astype(np.float32)
    vois = [sphere_mask((30, 30, 30), (c, 15.0, 15.0), 3.0) for c in (8.0, 15.0, 22.0)]
    bv = background_variability(img, vois)
    assert bv > 0.0


def test_curves_dataframe_roundtrip(tmp_path):
    import pandas as pd

    from krl_studies.metrics.curves import metrics_to_dataframe, write_metrics_csv

    rows = [
        {"iteration": 1, "nrmse": 0.5, "crc_mm8": 10.0},
        {"iteration": 2, "nrmse": 0.4, "crc_mm8": 20.0},
    ]
    df = metrics_to_dataframe(rows)
    assert list(df.columns) == ["iteration", "nrmse", "crc_mm8"]
    out = tmp_path / "m.csv"
    write_metrics_csv(rows, out)
    assert pd.read_csv(out).shape == (2, 3)
