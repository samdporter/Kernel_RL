#!/usr/bin/env python3
"""Manual smoke test for kernel operators using SIRF data structures."""

from __future__ import annotations

import time

import numpy as np

try:
    import sirf.STIR as pet
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("This script requires SIRF to be installed.") from exc

from src.my_kem import get_kernel_operator

np.seterr(over="raise", invalid="raise")


PARAMETERS = {
    "num_neighbours": 5,
    "sigma_anat": 0.1,
    "sigma_dist": 0.1,
    "normalize_features": False,
    "normalize_kernel": False,
    "use_mask": True,
    "mask_k": 15,
    "recalc_mask": False,
    "distance_weighting": False,
    "hybrid": False,
}


def create_fake_image(dimensions=(64, 64, 64)):
    img = pet.ImageData()
    img.initialise(dimensions, (1.0, 1.0, 1.0))
    img.fill(np.random.rand(*dimensions))
    return img


def run_kernel_check(backend, image, reps=5):
    op = get_kernel_operator(image, backend=backend)
    op.set_parameters(PARAMETERS)
    op.set_anatomical_image(image)

    fwd = op.direct(image)
    adj = op.adjoint(image)
    fwd_arr = fwd.as_array().copy()
    adj_arr = adj.as_array().copy()

    t0 = time.time()
    for _ in range(reps):
        _ = op.direct(image)
    elapsed = (time.time() - t0) / reps

    print(f"[{backend:6s}] avg forward time: {elapsed:.4f}s")
    return fwd_arr, adj_arr, elapsed


def main():
    dims = (96, 21, 108)
    base_image = create_fake_image(dims)

    backends = ["python", "numba"]
    results = {}
    for b in backends:
        print(f"Testing '{b}' backend…")
        try:
            fwd, adj, timing = run_kernel_check(b, base_image)
            results[b] = {"fwd": fwd, "adj": adj, "time": timing}
        except Exception as exc:  # pragma: no cover - manual smoke test
            print(f"  '{b}' failed: {exc}")

    python_result = results.get("python")
    if python_result:
        ref = python_result["fwd"]
        for b, data in results.items():
            if b == "python":
                continue
            ok_f = np.allclose(data["fwd"], ref, rtol=1e-6, atol=1e-8)
            ok_a = np.allclose(data["adj"], ref, rtol=1e-6, atol=1e-8)
            print(f"{b:6s} forward match: {ok_f}")
            print(f"{b:6s} adjoint match: {ok_a}")

    print("\n--- Timings (s) ---")
    for b, data in results.items():
        print(f"{b:6s}: {data['time']:.4f}")


if __name__ == "__main__":
    main()
