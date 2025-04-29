import numpy as np
np.seterr(over='raise', invalid='raise')
import time
import sirf.STIR as pet
from my_kem import get_kernel_operator  # your module with KernelOperator implementations

# fix RNG for reproducibility
np.random.seed(0)

# shared parameters for all backends
PARAMETERS = {
    'num_neighbours':    5,
    'sigma_anat':        0.1,
    'sigma_dist':        0.1,
    'normalize_features': False,
    'normalize_kernel':   False,
    'use_mask':           True,    # turn on similarity mask
    'mask_k':             15,      # keep top-15 most similar voxels
    'recalc_mask':        False,   # cache the mask after first build
}

def create_fake_image(dimensions=(64,64,64)):
    """Generate one random SIRF ImageData."""
    img = pet.ImageData()
    img.initialise(dimensions, (1.,1.,1.))
    img.fill(np.random.rand(*dimensions))
    return img

def test_kernel_operator(backend, image, reps=5):
    """
    Instantiate the operator for `backend`, set params/anatomy, then
    run .direct() and .adjoint() once (for correctness) and time `reps`
    forwards.
    Returns (fwd_array, adj_array, avg_forward_time).
    """
    op = get_kernel_operator(image, backend=backend)
    op.set_parameters(PARAMETERS)
    op.set_anatomical_image(image)

    # reference apply/adjoin
    fwd = op.direct(image)
    adj = op.adjoint(image)
    fwd_arr = fwd.as_array().copy()
    adj_arr = adj.as_array().copy()

    # timing forward only
    t0 = time.time()
    for _ in range(reps):
        _ = op.direct(image)
    elapsed = (time.time() - t0) / reps

    print(f"[{backend:6s}] avg forward time: {elapsed:.4f}s")
    return fwd_arr, adj_arr, elapsed

if __name__ == "__main__":
    dims = (64,64,64)
    base_image = create_fake_image(dims)

    backends = ['python', 'numba']
    results = {}

    # test each backend
    for b in backends:
        print(f"Testing '{b}' backend…")
        try:
            fwd, adj, t = test_kernel_operator(b, base_image)
            results[b] = {'fwd': fwd, 'adj': adj, 'time': t}
        except Exception as e:
            print(f"  '{b}' failed: {e}")

    # consistency check against Python reference
    if 'python' in results:
        ref = results['python']['fwd']
        print("\n--- Consistency vs Python ---")
        for b, data in results.items():
            if b == 'python':
                continue
            ok_f = np.allclose(data['fwd'], ref, rtol=1e-6, atol=1e-8)
            print(f"{b:6s} forward match: {ok_f}")
            ok_a = np.allclose(data['adj'], ref, rtol=1e-6, atol=1e-8)
            print(f"{b:6s} adjoint match: {ok_a}")
            if not ok_f or not ok_a:
                print(f"  {b} failed to match Python reference")

        # locate maximal discrepancy for NumPy vs Numba
        if 'numba' in results:
            py_fwd = ref
            nb_fwd = results['numba']['fwd']
            diff = nb_fwd - py_fwd
            idx_max = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
            val_py = py_fwd[idx_max]
            val_nb = nb_fwd[idx_max]
            delta = diff[idx_max]
            print("\nMax difference between Python and Numba at voxel", idx_max)
            print(f"  python = {val_py:.6e}")
            print(f"  numba  = {val_nb:.6e}")
            print(f"  Δ      = {delta:.6e}")

            # optionally list all mismatches above tolerance
            tol = 1e-8
            bad = np.where(np.abs(diff) > tol)
            count = bad[0].size
            print(f"\nVoxels differing by more than {tol:e}: {count}")
            if count > 0:
                coords = list(zip(bad[0], bad[1], bad[2]))[:10]
                print("First few mismatches:", coords)

    # timing summary
    print("\n--- Timings (s) ---")
    for b, d in results.items():
        print(f"{b:6s}: {d['time']:.4f}")
