import numpy as np
import time
import sirf.STIR as pet
import torch
import cupy as cp
from scipy.ndimage import convolve
import numba
from gaussian_blurring import GaussianBlurringOperator  # Assuming the class is in 'gaussian_blurring.py'


# Create a fake image using sirf.STIR
def create_fake_image(dimensions=(64, 64, 64)):
    image = pet.ImageData()
    image.initialise(dimensions, (1.0, 1.0, 1.0))  # Using voxel size of (1.0, 1.0, 1.0)
    image.fill(np.random.rand(*dimensions))  # Filling with random values
    return image


# Test the backend with different configurations
def test_blurring_operator(backend='numba',
                           dimensions=(181, 217, 181),
                           repetitions=10):
    # Create fake image
    image = create_fake_image(dimensions)

    # Create the Gaussian Blurring Operator
    sigma = (1.0, 1.0, 1.0)  # Example Gaussian standard deviation
    operator = GaussianBlurringOperator(sigma, image, backend=backend)

    # Timing the direct (convolution) method
    start_time = time.time()
    for _ in range(repetitions):
        blurred_image = operator.direct(image)
    end_time = time.time()

    print(f"[{backend}] Convolution Average Time: {(end_time - start_time)/repetitions:.6f} seconds")

    return end_time - start_time


if __name__ == '__main__':
    # Test different backends
    backends = ['numba', 'cupy', 'torch', 'scipy']

    times = {}
    for backend in backends:
        print(f"Testing {backend} backend:")
        try:
            times[backend] = test_blurring_operator(backend)
        except Exception as e:
            print(f"Error with {backend}: {e}")
    # Print out the times for comparison
    print("\n--- Results ---")
    for backend, time_taken in times.items():
        print(f"{backend}: {time_taken:.6f} seconds")
