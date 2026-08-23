#!/usr/bin/env python3
"""
Example script demonstrating KRL reconstruction on spheres phantom data.

This script runs three reconstruction methods:
1. Standard Richardson-Lucy (RL)
2. Kernelised RL (KRL) with anatomical guidance
3. Hybrid KRL (HKRL) mixing emission and anatomical features
4. MAP-RL with Directional Total Variation (DTV)

Make sure you have the required data in data/spheres/:
- PET emission image (e.g., OSEM reconstruction)
- Anatomical guidance image (e.g., T1 MRI)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import KRL package
from krl import (
    get_kernel_operator,
    create_gaussian_blur,
    DirectionalOperator,
    load_image,
    save_image,
)
from krl.algorithms.lbfgsb import LBFGSBOptimizer, LBFGSBOptions
from krl.utils import get_array

# Import CIL components
try:
    import cil.optimisation.functions as fn
    import cil.optimisation.operators as op
    from cil.optimisation.operators import GradientOperator
    from cil.optimisation.utilities.callbacks import Callback
    from cil.utilities.display import show2D
except ImportError:
    raise ImportError(
        "CIL is required. Install with:\n"
        "  conda install -c conda-forge -c ccpi cil"
    )
    
ALGORITHMS_TO_RUN=["RL", "KRL", "HKRL", "DTV"]

DATA_TYPE = "brain"  # Options: "spheres", "brain"

def fwhm_to_sigma(fwhm):
    """Convert FWHM to sigma for Gaussian."""
    return tuple(f / (2.0 * np.sqrt(2.0 * np.log(2.0))) for f in fwhm)


def richardson_lucy(
    observed, blur_op, 
    iterations, *, 
    freeze_iteration=np.inf,
    epsilon=1e-10, 
    kernel_operator=None):
    """
    Richardson-Lucy deconvolution with optional kernel operator.

    Parameters
    ----------
    observed : ImageData
        Observed image (blurred emission data)
    blur_op : LinearOperator
        Blurring operator (PSF)
    iterations : int
        Number of RL iterations
    epsilon : float
        Small value to avoid division by zero
    kernel_operator : LinearOperator, optional
        Kernel operator for anatomical guidance (KRL/HKRL)

    Returns
    -------
    result : ImageData
        Reconstructed image
    objectives : list
        Objective function values per iteration
    """
    print(f"Running Richardson-Lucy for {iterations} iterations...")

    # Compute sensitivity (normalization)
    geometry = observed.geometry
    sensitivity = geometry.allocate(value=1)
    current = observed.clone()

    if kernel_operator is not None:
        # KRL/HKRL mode: compose blur and kernel operators
        effective_blur = op.CompositionOperator(blur_op, kernel_operator)
        est_blur = effective_blur.direct(current)
        sensitivity = effective_blur.adjoint(geometry.allocate(value=1))
    else:
        # Standard RL mode
        effective_blur = blur_op
        est_blur = effective_blur.direct(current)
        
    objective_values = []

    # RL iterations
    for idx in range(iterations):
        # RL update: x *= (A^T (y / Ax)) / sensitivity
        current *= effective_blur.adjoint(observed / (est_blur + epsilon))
        current /= (sensitivity + epsilon)

        # Re-estimate
        est_blur = effective_blur.direct(current)

        # Compute objective (KL divergence)
        obj = (est_blur - observed * (est_blur + epsilon).log()).sum()
        objective_values.append(obj)

        if (idx + 1) % 20 == 0:
            print(f"  Iteration {idx+1}/{iterations}, objective: {obj:.2f}")
            
        if idx + 1 == freeze_iteration and kernel_operator is not None:
            print(f"  Freezing kernel operator at iteration {idx+1}")
            kernel_operator.freeze_emission_kernel = True

    return current, objective_values


def main():
    """Run the example reconstruction pipeline."""

    # ========== Configuration ==========

    # Data paths
    if DATA_TYPE == "brain":
        data_dir = Path("data/MK-H001")
        emission_file = data_dir / "MK-H001_PET_MNI.nii"  # Or .nii.gz
        anatomy_file = data_dir / "MK-H001_T1_MNI.nii"     # Or .nii.gz
        output_dir = Path("results/brain_example")
    else:  # spheres phantom
        data_dir = Path("data/spheres")
        emission_file = data_dir / "phant_pet.nii"  # Or .nii.gz
        anatomy_file = data_dir / "phant_mri.nii"        # Or .nii.gz
        output_dir = Path("results/spheres_example")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # PSF parameters (FWHM in mm)
    fwhm = (6.0, 6.0, 6.0)

    # RL parameters
    rl_iterations = 100

    # KRL parameters
    krl_params = {
        "num_neighbours": 7,          # 9x9x9 neighbourhood
        "sigma_anat": 1.0,            # Anatomical similarity weight
        "sigma_dist": 3.0,            # Spatial distance weight
        "sigma_emission": 1.0,        # Emission similarity (for hybrid)
        "normalize_kernel": True,     # Normalize kernel weights
        "use_mask": True,             # Use k-NN masking
        "mask_k": 48,                 # Keep top 48 neighbours
        "hybrid": False,              # Standard KRL (not hybrid)
    }

    # HKRL parameters (hybrid mode)
    hkrl_params = krl_params.copy()
    hkrl_params["hybrid"] = True    # Enable hybrid mode

    # DTV parameters
    dtv_iterations = 100
    alpha = 0.1                       # Regularization strength
    lbfgs_max_linesearch = 20
    lbfgs_ftol = 1e-6
    lbfgs_gtol = 1e-6

    print("="*60)
    print("KRL Spheres Phantom Reconstruction Example")
    print("="*60)

    # ========== Load Data ==========

    print("\n1. Loading images...")
    try:
        emission = load_image(emission_file)
        anatomy = load_image(anatomy_file)
        print(f"   Emission shape: {emission.shape}")
        print(f"   Anatomy shape: {anatomy.shape}")
    except FileNotFoundError as e:
        print(f"\nError: Could not find data files in {data_dir}")
        print(f"Please ensure you have:")
        print(f"  - {emission_file}")
        print(f"  - {anatomy_file}")
        print(f"\nYou may need to run setup_demo.py first or adjust the file paths.")
        return

    # ========== Create Operators ==========

    print("\n2. Creating operators...")

    # PSF blurring operator
    blur_op = create_gaussian_blur(
        fwhm_to_sigma(fwhm),
        emission.geometry,
        backend="auto"  # Auto-select best available backend
    )
    print(f"   Blur operator backend: {blur_op.backend}")
    
    rl_result = None
    krl_result = None
    hkrl_result = None
    dtv_result = None
    rl_obj = None
    krl_obj = None
    hkrl_obj = None
    dtv_obj = None

    if 'RL' in ALGORITHMS_TO_RUN:
        # ========== Standard RL ==========

        print("\n3. Running standard Richardson-Lucy...")
        rl_result, rl_obj = richardson_lucy(
            emission,
            blur_op,
            iterations=rl_iterations
        )

        # Save result
        save_image(rl_result, output_dir / "rl_reconstruction.nii.gz")
        print(f"   Saved: {output_dir / 'rl_reconstruction.nii.gz'}")

    if 'KRL' in ALGORITHMS_TO_RUN:
        # ========== Kernelised RL (KRL) ==========
        print("\n4. Running Kernelised RL (KRL)...")

        # Create kernel operator
        kernel_op = get_kernel_operator(emission, backend="auto")
        kernel_op.set_parameters(krl_params)
        kernel_op.set_anatomical_image(anatomy)
        print(f"   Kernel operator backend: {kernel_op.backend}")

        krl_latent, krl_obj = richardson_lucy(
            emission,
            blur_op,
            iterations=rl_iterations,
            kernel_operator=kernel_op
        )

        # Apply kernel to get final reconstruction
        krl_result = kernel_op.direct(krl_latent)
        save_image(krl_result, output_dir / "krl_reconstruction.nii.gz")
        print(f"   Saved: {output_dir / 'krl_reconstruction.nii.gz'}")

    if 'HKRL' in ALGORITHMS_TO_RUN:
        # ========== Hybrid KRL (HKRL) ==========

        print("\n5. Running Hybrid KRL (HKRL)...")

        # Create hybrid kernel operator
        hkernel_op = get_kernel_operator(emission, backend="auto")
        hkernel_op.set_parameters(hkrl_params)
        hkernel_op.set_anatomical_image(anatomy)

        hkrl_latent, hkrl_obj = richardson_lucy(
            emission,
            blur_op,
            iterations=rl_iterations,
            freeze_iteration=5,
            kernel_operator=hkernel_op
        )

        hkrl_result = hkernel_op.direct(hkrl_latent)
        save_image(hkrl_result, output_dir / "hkrl_reconstruction.nii.gz")
        print(f"   Saved: {output_dir / 'hkrl_reconstruction.nii.gz'}")

    if 'DTV' in ALGORITHMS_TO_RUN:
        # ========== L-BFGS-B with Directional TV (DTV) ==========

        print("\n6. Running L-BFGS-B with Directional TV...")

        # Data fidelity term (KL divergence)
        f = fn.KullbackLeibler(
            b=emission,
            eta=emission.geometry.allocate(value=1e-6)
        )
        df = fn.OperatorCompositionFunction(f, blur_op)

        # Regularization prior (Directional TV)
        grad = GradientOperator(emission.geometry, method="forward", bnd_cond="Neumann")
        grad_anatomy = grad.direct(anatomy)
        d_op = op.CompositionOperator(DirectionalOperator(grad_anatomy), grad)
        prior = alpha * fn.OperatorCompositionFunction(
            fn.SmoothMixedL21Norm(epsilon=1e-4), d_op
        )

        lbfgs = LBFGSBOptimizer(
            initial_estimate=emission,
            data_fidelity=df,
            prior=prior,
            options=LBFGSBOptions(
                max_linesearch=lbfgs_max_linesearch,
                ftol=lbfgs_ftol,
                gtol=lbfgs_gtol,
                enforce_non_negativity=True,
            ),
        )

        lbfgs.run(verbose=1, iterations=dtv_iterations)
        dtv_result = lbfgs.solution
        dtv_obj = lbfgs.objective

        save_image(dtv_result, output_dir / "dtv_reconstruction.nii.gz")
        print(f"   Saved: {output_dir / 'dtv_reconstruction.nii.gz'}")

    # ========== Visualization ==========

    print("\n7. Creating comparison plots...")

    titles = [
        "Richardson-Lucy (RL)",
        "Kernelised RL (KRL)",
        "Hybrid KRL (HKRL)",
        "L-BFGS-B with DTV"
    ]

    # Plot objectives
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i, obj in enumerate([rl_obj, krl_obj, hkrl_obj, dtv_obj]):
        if obj is not None:
            ax.plot(obj, label=titles[i], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Function Value")
    ax.set_title("Convergence Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "objectives_comparison.png", dpi=150)
    print(f"   Saved: {output_dir / 'objectives_comparison.png'}")

    recon_results = [rl_result, krl_result, hkrl_result, dtv_result]
    recon_pairs = [
        (titles[idx], res) for idx, res in enumerate(recon_results) if res is not None
    ]

    # Plot central slice comparison
    mid_slice = emission.shape[0] // 2

    cols = max(3, len(recon_pairs))  # Keep space for inputs even if few results
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 10))
    axes = np.asarray(axes)


    em_max = np.percentile(get_array(emission), 99.9)
    # Input images
    axes[0, 0].imshow(get_array(emission)[mid_slice], cmap="gray", vmin=0, vmax=em_max)
    axes[0, 0].set_title("Input: OSEM")
    axes[0, 0].axis("off")

    if cols > 1:
        axes[0, 1].imshow(get_array(anatomy)[mid_slice], cmap="gray")
        axes[0, 1].set_title("Anatomical Guidance (T1)")
        axes[0, 1].axis("off")

    for idx in range(2, cols):
        axes[0, idx].axis("off")

    # Reconstructions
    for idx, (label, result) in enumerate(recon_pairs):
        axes[1, idx].imshow(get_array(result)[mid_slice], cmap="hot", vmin=0, vmax=em_max)
        axes[1, idx].set_title(label)
        axes[1, idx].axis("off")

    for idx in range(len(recon_pairs), cols):
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "reconstructions_comparison.png", dpi=150)
    print(f"   Saved: {output_dir / 'reconstructions_comparison.png'}")

    # Profile plot
    profile_y = emission.shape[1] // 2
    profile_x = emission.shape[2] // 2

    fig, ax = plt.subplots(figsize=(12, 6))
    profile_pairs = [("Input: OSEM", emission)] + recon_pairs
    for label, data in profile_pairs:
        ax.plot(
            get_array(data)[mid_slice, :, profile_x],
            label=label,
            linewidth=2
        )
    ax.set_xlabel("Position (voxels)")
    ax.set_ylabel("Intensity")
    ax.set_title("Central Profile Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "profile_comparison.png", dpi=150)
    print(f"   Saved: {output_dir / 'profile_comparison.png'}")

    print("\n" + "="*60)
    print("Reconstruction complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
