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
    """High-count limit: sum(recon) should be invariant to counts (GT units).

    The reconstruction is rescaled back to ground-truth units by dividing by
    the prompt scaling factor, so the sum of the returned image should be
    approximately invariant to the noise level chosen for the prompts.
    """
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

    recon_sums = []
    metas = []
    for counts in (1e6, 1e7, 1e8):
        cfg = dict(cfg_base)
        cfg["counts"] = counts
        recon, meta = simulate_inputs(gt, cfg)
        recon_sums.append(float(recon.sum()))
        metas.append(meta)

    # sum(recon) should be roughly constant across counts (recon is in GT units)
    rel_std = np.std(recon_sums) / np.mean(recon_sums)
    assert rel_std < 0.1, (
        f"Activity scale contract violated: recon sums {recon_sums}, "
        f"rel_std {rel_std:.3f}"
    )

    # Metadata contract: reconstructions are in GT units and prompt_scale is recorded
    for meta in metas:
        assert meta["activity_units"] == "ground_truth", (
            f"meta must declare activity_units='ground_truth', got {meta.get('activity_units')!r}"
        )
        assert "prompt_scale" in meta, "meta must contain 'prompt_scale'"
        assert meta["prompt_scale"] > 0
