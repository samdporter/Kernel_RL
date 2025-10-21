"""Callback utilities for KRL reconstruction algorithms."""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from cil.optimisation.utilities.callbacks import Callback
    from cil.optimisation.algorithms import Algorithm
except ImportError:
    # Fallback for when CIL is not available
    class Callback:
        def __call__(self, algorithm):
            pass

    class Algorithm:
        pass

from krl.utils import get_array


class NRMSECallback(Callback):
    """
    Callback to compute and save Normalized Root Mean Square Error (NRMSE) per iteration.

    NRMSE is computed as: ||reconstruction - ground_truth|| / max(ground_truth)

    This is useful for quantitative evaluation when a ground truth is available,
    such as with phantom data.

    Parameters
    ----------
    ground_truth : ImageData
        Ground truth image for comparison
    output_file : str or Path
        Path to save NRMSE values (CSV format)
    interval : int, optional
        Compute NRMSE every N iterations (default: 1, i.e., every iteration)
    kernel_operator : LinearOperator, optional
        If provided, apply this operator to the solution before computing NRMSE.
        This is useful for KRL/HKRL where the algorithm operates on a latent image.
    verbose : bool, optional
        If True, print NRMSE values to console (default: True)

    Examples
    --------
    >>> ground_truth = load_image("data/spheres/phant_orig.nii")
    >>> callback = NRMSECallback(
    ...     ground_truth=ground_truth,
    ...     output_file="results/nrmse.csv"
    ... )
    >>> algorithm.run(iterations=100, callbacks=[callback])
    """

    def __init__(
        self,
        ground_truth,
        output_file: Path,
        interval: int = 1,
        kernel_operator=None,
        verbose: bool = True,
    ):
        super().__init__()
        self.ground_truth = ground_truth
        self.ground_truth_array = get_array(ground_truth)
        self.gt_max = np.max(self.ground_truth_array)
        self.output_file = Path(output_file)
        self.interval = interval
        self.kernel_operator = kernel_operator
        self.verbose = verbose
        self.nrmse_values = []

        # Create output file with header
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            f.write("iteration,nrmse\n")

    def __call__(self, algorithm: Algorithm) -> None:
        """
        Called by the algorithm at each iteration.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm instance (provides iteration number and solution)
        """
        if algorithm.iteration % self.interval != 0:
            return

        # Get current solution
        current_solution = algorithm.solution

        # Apply kernel operator if provided (for KRL/HKRL)
        if self.kernel_operator is not None:
            current_solution = self.kernel_operator.direct(current_solution)

        # Compute NRMSE
        current_array = get_array(current_solution)
        mse = np.mean((current_array - self.ground_truth_array) ** 2)
        nrmse = np.sqrt(mse) / self.gt_max

        self.nrmse_values.append((algorithm.iteration, nrmse))

        # Save to file (append mode)
        with open(self.output_file, 'a') as f:
            f.write(f"{algorithm.iteration},{nrmse:.8e}\n")

        # Print if verbose
        if self.verbose:
            print(f"  Iteration {algorithm.iteration}: NRMSE = {nrmse:.6f}")


class SaveIterationCallback(Callback):
    """
    Callback to save reconstruction at specific iterations.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save iteration files
    interval : int
        Save every N iterations
    prefix : str, optional
        Prefix for saved filenames (default: "iter")
    kernel_operator : LinearOperator, optional
        If provided, apply this operator to the solution before saving
    save_first_n : int, optional
        Save the first N iterations (default: 5)
    """

    def __init__(
        self,
        output_dir: Path,
        interval: int = 10,
        prefix: str = "iter",
        kernel_operator=None,
        save_first_n: int = 5,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.prefix = prefix
        self.kernel_operator = kernel_operator
        self.save_first_n = save_first_n

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, algorithm: Algorithm) -> None:
        """Save current solution at specified intervals."""
        # Save first N iterations (1, 2, 3, 4, 5)
        if algorithm.iteration <= self.save_first_n:
            should_save = True
        # Then save at regular intervals (10, 20, 30...)
        elif algorithm.iteration % self.interval == 0:
            should_save = True
        else:
            should_save = False

        if not should_save:
            return

        from krl.utils import save_image

        # Get current solution
        current_solution = algorithm.solution

        # Apply kernel operator if provided
        if self.kernel_operator is not None:
            current_solution = self.kernel_operator.direct(current_solution)

        # Clamp to non-negative values
        with np.errstate(invalid="ignore"):
            current_solution.maximum(0, out=current_solution)

        # Save
        output_path = self.output_dir / f"{self.prefix}_{algorithm.iteration:04d}.nii.gz"
        save_image(current_solution, output_path)
