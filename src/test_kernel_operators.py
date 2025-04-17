import numpy as np
import time
import sirf.STIR as pet
from my_kem import get_kernel_operator

# Create a fake image using sirf.STIR
def create_fake_image(dimensions=(64, 64, 64)):
    image = pet.ImageData()
    image.initialise(dimensions, (1.0, 1.0, 1.0))
    image.fill(np.random.rand(*dimensions))
    return image

# Test each backend implementation
def test_kernel_operator(backend, dimensions=(181, 217, 181), reps=10):
    # Prepare image
    image = create_fake_image(dimensions)
    # Select backend flags
    flags = {'use_torch': False, 'use_cupy': False, 'use_numba': False}
    if backend == 'torch':
        flags['use_torch'] = True
    elif backend == 'cupy':
        flags['use_cupy'] = True
    elif backend == 'numba':
        flags['use_numba'] = True
    # Create operator
    op = get_kernel_operator(image, **flags)
    op.set_anatomical_image(image)
    # Warm-up
    _ = op.direct(image)
    # Time the direct (apply) operation
    start = time.time()
    for _ in range(reps):
        _ = op.direct(image)
    elapsed = (time.time() - start) / reps
    print(f"[{backend}] avg time: {elapsed:.6f}s")
    return elapsed

if __name__ == '__main__':
    backends = ['torch', 'cupy', 'numba', 'python']
    results = {}
    for b in backends:
        print(f"Testing {b} backend...")
        try:
            results[b] = test_kernel_operator(b)
        except Exception as e:
            print(f"{b} failed: {e}")

    print("\n--- Summary ---")
    for b, t in results.items():
        print(f"{b}: {t:.6f} s")
