import numpy as np
import time
import sirf.STIR as pet
from my_kem import get_kernel_operator

# fix the NumPy RNG so that .rand() is reproducible
np.random.seed(0)

def create_fake_image(dimensions=(64,64,64)):
    """Generate one random image; seed is already global."""
    img = pet.ImageData()
    img.initialise(dimensions, (1.,1.,1.))
    img.fill(np.random.rand(*dimensions))
    return img

def test_kernel_operator(backend, image, reps=5):
    op = get_kernel_operator(image, backend=backend)
    op.set_anatomical_image(image)

    # warm-up + reference forward/adjoint
    fwd = op.direct(image); adj = op.adjoint(image)
    fwd_arr = fwd.as_array().copy()
    adj_arr = adj.as_array().copy()

    # timing only the forward
    t0 = time.time()
    for _ in range(reps):
        _ = op.direct(image)
    elapsed = (time.time() - t0)/reps

    print(f"[{backend:6s}] avg forward time: {elapsed:.4f}s")
    return fwd_arr, adj_arr, elapsed

if __name__ == "__main__":
    dimensions = (64,64,64)
    # generate *one* image for all backends
    base_image = create_fake_image(dimensions)

    backends = ['python','numba','torch']
    results = {}
    for b in backends:
        print(f"Testing {b!r} backend…")
        try:
            fwd, adj, t = test_kernel_operator(b, base_image)
            results[b] = {'fwd':fwd, 'adj':adj, 'time':t}
        except Exception as e:
            print(f"  {b!r} failed: {e}")

    # consistency vs Python
    ref = results.get('python')
    if ref:
        print("\n--- Consistency vs Python ---")
        for b, data in results.items():
            if b=='python': continue
            ok_f = np.allclose(data['fwd'], ref['fwd'], rtol=1e-6, atol=1e-8)
            ok_a = np.allclose(data['adj'], ref['adj'], rtol=1e-6, atol=1e-8)
            print(f"{b:6s} forward match: {ok_f}, adjoint match: {ok_a}")

    # timing summary
    print("\n--- Timings (s) ---")
    for b, d in results.items():
        print(f"{b:6s}: {d['time']:.4f}")
