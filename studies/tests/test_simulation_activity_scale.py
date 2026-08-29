# studies/tests/test_simulation_activity_scale.py
import numpy as np
import pytest

try:
    import sirf.STIR  # noqa: F401
    HAS_SIRF = True
except ImportError:
    HAS_SIRF = False

pytestmark = [
    pytest.mark.sirf,
    pytest.mark.skipif(not HAS_SIRF, reason="SIRF not available"),
]

def test_simulate_inputs_activity_scale_contract():
    """High-count limit: sum(recon) * scale should be invariant to counts."""
    from krl_studies.simulation import simulate_inputs

    # Uniform GT on 1 mm grid
    gt = np.full((32, 32, 32), 5.0, dtype=np.float32)

    # Two count levels
    cfg_base = {
        "condition": "psf-matched",
        "beta": None,
        "realisation": 0,
        "seed": 1337,
        "n_subits": 4,
    }

    scales = []
    recon_sums = []
    metas = []
    for counts in (1e6, 1e7, 1e8):
        cfg = dict(cfg_base)
        cfg["counts"] = counts
        recon, meta = simulate_inputs(gt, cfg)
        scales.append(meta.get("scale", 1.0))
        recon_sums.append(float(recon.sum()))
        metas.append(meta)

    # scale * sum(recon) should be roughly constant (invariant to Poisson noise)
    scaled_sums = [s * r for s, r in zip(scales, recon_sums)]
    rel_std = np.std(scaled_sums) / np.mean(scaled_sums)
    assert rel_std < 0.1, f"Activity scale contract violated: scaled sums {scaled_sums}, rel_std {rel_std:.3f}"

    # Also check that scale is recorded
    for meta in metas:
        assert "scale" in meta, "meta must contain 'scale' = counts / sum(prompts)"
