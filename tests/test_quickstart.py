"""Runs the README quickstart against real CIL.

Keep this snippet in sync with the Quickstart section of ``README.md``: the
geometry, synthetic images, observed data, kernel operator, callback and run
sequence below are exactly what the README documents.
"""

import numpy as np
from cil.framework import ImageGeometry
from cil.optimisation.utilities.callbacks import Callback

from krl import RichardsonLucy, create_gaussian_blur, get_kernel_operator


def test_readme_quickstart_reconstruction_is_finite_and_non_negative():
    # 1. Geometry and two aligned synthetic images
    geometry = ImageGeometry(voxel_num_x=32, voxel_num_y=32, voxel_num_z=16)

    z, y, x = np.indices((16, 32, 32), dtype=np.float32)
    emission = geometry.allocate(0.0)
    emission.fill(50.0 * np.exp(-((z - 8) ** 2 + (y - 12) ** 2 + (x - 14) ** 2) / 12.0))
    mr_image = geometry.allocate(0.0)
    mr_image.fill(np.exp(-((z - 6) ** 2 + (y - 20) ** 2 + (x - 18) ** 2) / 8.0))

    # 2. Observed data: the emission image blurred by the PSF
    blur_op = create_gaussian_blur(sigma=(1.5, 1.5, 1.5), geometry=geometry, backend="numba")
    observed = blur_op.direct(emission)

    # 3. Anatomical guidance operator
    kernel_op = get_kernel_operator(
        geometry, backend="numba", num_neighbours=3, sigma_anat=0.5
    )
    kernel_op.set_anatomical_image(mr_image)

    # 4. Callback: records and prints the objective after each iteration
    class ObjectiveCallback(Callback):
        def __init__(self):
            super().__init__()
            self.values = []

        def __call__(self, algorithm):
            self.values.append(float(algorithm.loss[-1]))
            print(f"iteration {algorithm.iteration}: objective {self.values[-1]:.4f}")

    # 5. Reconstruct (omit kernel_operator for standard RL)
    callback = ObjectiveCallback()
    algo = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur_op,
        observed_data=observed,
        kernel_operator=kernel_op,
    )
    algo.run(iterations=8, callbacks=[callback])

    reconstruction = algo.get_output()
    assert np.isfinite(reconstruction.as_array()).all()
    assert (reconstruction.as_array() >= 0).all()

    # One callback call before the first update, then one per update.
    assert len(callback.values) == 9
    assert all(np.isfinite(value) for value in callback.values)
