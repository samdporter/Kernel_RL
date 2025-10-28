#!/usr/bin/env python3
"""
Poster figure generation script for KRL deconvolution methods.

This script generates:
1. Dataset overview figures (spheres and brain)
2. Best reconstruction comparisons across methods (RL, KRL, HKRL, RL-dTV)
3. NRMSE convergence plots
4. Objective function convergence plots
5. Profile plots through images
6. Summary table of best results

CONFIGURATION:
--------------
To manually select methods and parameters:
1. Set USE_BEST_RESULTS = False (line ~47)
2. Edit the parameter variables (lines ~63-86):
   - RL_ITERATION_SPHERES/BRAIN
   - KRL_SIGMA_ANAT, KRL_ITERATION_SPHERES/BRAIN
   - HKRL_FREEZE_ITERATION (spheres), HKRL_FREEZE_ITERATION_BRAIN (brain), HKRL_SIGMA_ANAT, HKRL_SIGMA_EMISSION, HKRL_ITERATION_SPHERES/BRAIN
   - DTV_ALPHA, DTV_ITERATION_SPHERES/BRAIN

   IMPORTANT: Choose parameter values that match existing experiments:
   - Spheres: sigma_anat in {0.1, 0.2, 0.5, 1, 2, 5}
             DTV alpha in {0.1, 0.2, 0.5, 1, 2, 5}
             HKRL sigma_emission in {0.1, 0.2, 0.5, 1, 2, 5}
   - Brain:  sigma_anat in {0.01, 0.05, 0.1, 0.2, 0.5, 1, 2}
             DTV alpha in {0.1, 0.2, 0.5, 1, 2, 5}
             HKRL sigma_emission in {0.01, 0.05, 0.1, 0.5, 1, 5}

To use automatic best result finding (spheres only):
1. Set USE_BEST_RESULTS = True (default)
   Note: Brain ALWAYS uses manual selection (no ground truth available)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional
import re
import cmasher

# Configuration
DATA_DIR = Path("/home/sam/working/others/Kjell/KRL/data")
RESULTS_DIR = Path("/home/sam/working/others/Kjell/KRL/results")
OUTPUT_DIR = Path("/home/sam/working/others/Kjell/KRL/poster_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Method Selection for Figures
# ============================================================================
# Configuration for method parameters and iterations
# Set USE_BEST_RESULTS = False to use manual selection below
USE_BEST_RESULTS = True  # Set to True to automatically find best NRMSE results for spheres (default)

# Manual parameter selection (used when USE_BEST_RESULTS = False for spheres)
# BRAIN always uses manual selection (no ground truth for auto-finding)
#
# IMPORTANT: Manually specify parameters below to select which experiments to plot
#
# For SPHERES: Available sigma_anat values: 0.1, 0.2, 0.5, 1, 2, 5
#              Available DTV alpha values: 0.1, 0.2, 0.5, 1, 2, 5
#              Available HKRL sigma_emission values: 0.1, 0.2, 0.5, 1, 2, 5
#
# For BRAIN (mk-h001): Available sigma_anat values: 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2
#                      Available DTV alpha values: 0.1, 0.2, 0.5, 1, 2, 5
#                      Available HKRL sigma_emission values: 0.01, 0.05, 0.1, 0.5, 1, 5
#                      Note: Some KRL/HKRL experiments use sd_1_0, others use sd_5000

# Method-specific parameters
RL_ITERATION_SPHERES = 0
RL_ITERATION_BRAIN = 10

# KRL parameters
KRL_SIGMA_ANAT = 0.1
KRL_ITERATION_SPHERES = 2
KRL_ITERATION_BRAIN = 50

# HKRL parameters
HKRL_FREEZE_ITERATION = 10  # For spheres - freeze at iteration 10
HKRL_FREEZE_ITERATION_BRAIN = 5  # For brain - different from spheres!
HKRL_SIGMA_ANAT = 0.1  # Changed to 0.1 to match actual results
HKRL_SIGMA_EMISSION = 1  # Changed to 1 to match actual results
HKRL_ITERATION_SPHERES = 50  # Spheres HKRL final iteration
HKRL_ITERATION_BRAIN = 50

# RL-dTV parameters
DTV_ALPHA = 0.02  # Changed to 0.01 to match actual brain results
DTV_ALPHA_SPHERES = 0.1  # Spheres uses alpha=0.1 for DTV
DTV_ITERATION_SPHERES = 250  # Changed to 250 - the actual max iteration for spheres DTV
DTV_ITERATION_BRAIN = 250 # Changed to 250 - the actual max iteration for brain DTV

# Helper function to format parameter values for directory names
def format_param(value: float) -> str:
    """Format parameter for directory name (e.g., 1.0 -> '1', 0.5 -> '0p5')"""
    if value == int(value):
        return str(int(value))
    else:
        return str(value).replace(".", "p")

def format_param_underscore(value: float) -> str:
    """Format parameter for suffix (e.g., 1.0 -> '1_0', 0.5 -> '0_5')"""
    if value == int(value):
        return f"{int(value)}_0"
    else:
        return str(value).replace(".", "_")

# Auto-generated method dictionaries (don't edit these directly, edit params above)
SPHERES_METHODS = {
    'spheres_rl_base_RL_G5_5mm': RL_ITERATION_SPHERES,
    f'spheres_krl_sigma_anat-{format_param(KRL_SIGMA_ANAT)}_KRL_s{format_param_underscore(KRL_SIGMA_ANAT)}_sd_1_0_k9_G5_5mm_init-uniform': KRL_ITERATION_SPHERES,
    f'spheres_hkrl_freeze_iteration-{HKRL_FREEZE_ITERATION}_sigma_anat-{format_param(HKRL_SIGMA_ANAT)}_sigma_emission-{format_param(HKRL_SIGMA_EMISSION)}_KRL_s{format_param_underscore(HKRL_SIGMA_ANAT)}_sd_1_0_k9_G5_5mm_init-uniform': HKRL_ITERATION_SPHERES,
    f'spheres_dtv_alpha-{format_param(DTV_ALPHA_SPHERES)}_use_uniform_initial-off_DTV_a{format_param_underscore(DTV_ALPHA_SPHERES)}_ls20_ftol1e-06_G5_5mm_init-blurred': DTV_ITERATION_SPHERES,
}

# BRAIN_METHODS: Manually specify exact experiment directories and iterations
# Brain datasets should ALWAYS be manually specified (no automatic best-finding)
# NOTE: Brain experiments have _init-uniform or _init-blurred suffixes
BRAIN_METHODS = {
    'mk-h001_rl_use_uniform_initial-on_RL_G6_0mm_init-uniform': RL_ITERATION_BRAIN,
    f'mk-h001_krl_sigma_anat-{format_param(KRL_SIGMA_ANAT)}_KRL_s{format_param_underscore(KRL_SIGMA_ANAT)}_sd_1_0_k9_G6_0mm_init-uniform': KRL_ITERATION_BRAIN,
    f'mk-h001_hkrl_freeze_iteration-{HKRL_FREEZE_ITERATION_BRAIN}_sigma_anat-{format_param(HKRL_SIGMA_ANAT)}_sigma_emission-{format_param(HKRL_SIGMA_EMISSION)}_KRL_s{format_param_underscore(HKRL_SIGMA_ANAT)}_sd_1_0_k9_G6_0mm_init-uniform': HKRL_ITERATION_BRAIN,
    f'mk-h001_dtv_alpha-{format_param(DTV_ALPHA)}_use_uniform_initial-off_DTV_a{format_param_underscore(DTV_ALPHA)}_ls20_ftol1e-08_G6_0mm_init-blurred': DTV_ITERATION_BRAIN,
}

# ============================================================================
# Helper Functions
# ============================================================================

def load_nifti_data(filepath: Path) -> np.ndarray:
    """Load NIfTI file and return data array."""
    img = nib.load(filepath)
    return np.array(img.dataobj)


def get_central_slice(data: np.ndarray, axis: int = 1) -> np.ndarray:
    """Get central slice along specified axis."""
    mid_idx = data.shape[axis] // 2
    if axis == 0:
        return data[mid_idx, :, :]
    elif axis == 1:
        return data[:, mid_idx, :]
    elif axis == 2:
        return data[:, :, mid_idx]
    else:
        raise ValueError(f"Invalid axis: {axis}")


def load_spheres_data() -> Dict[str, np.ndarray]:
    """Load spheres phantom data (ground truth, blurred, MRI)."""
    spheres_dir = DATA_DIR / "spheres"
    return {
        'ground_truth': load_nifti_data(spheres_dir / "phant_orig.nii"),
        'blurred_pet': load_nifti_data(spheres_dir / "phant_pet.nii"),
        'mri_guidance': load_nifti_data(spheres_dir / "phant_mri.nii")
    }


def load_brain_data() -> Dict[str, np.ndarray]:
    """Load brain (MK-H001) data."""
    brain_dir = DATA_DIR / "MK-H001"
    return {
        'blurred_pet': load_nifti_data(brain_dir / "MK-H001_PET_MNI.nii"),
        'mri_guidance': load_nifti_data(brain_dir / "MK-H001_T1_MNI.nii")
    }


def find_reconstruction_file(result_dir: Path, iteration: int) -> Optional[Path]:
    """Find reconstruction file for a specific iteration or nearest available."""
    # Try different naming patterns for exact iteration
    patterns = [
        f"kernel_deconv_iter_{iteration:04d}.nii.gz",
        f"deconv_iter_{iteration:04d}.nii.gz",
        f"recon_iter_{iteration:04d}.nii.gz",
        f"dtv_iter_{iteration}.nii.gz",  # DTV naming
        f"dtv_iter_{iteration + 1}.nii.gz",  # DTV sometimes saves iteration+1
        f"rl_deconv_iter_{iteration:04d}.nii.gz",  # RL naming
        f"rl_iter_{iteration:04d}.nii.gz",
    ]

    for pattern in patterns:
        filepath = result_dir / pattern
        if filepath.exists():
            return filepath

    # If exact iteration not found, find nearest available iteration
    print(f"  Exact iteration {iteration} not found, searching for nearest...")

    # Search for all reconstruction files
    all_patterns = [
        "kernel_deconv_iter_*.nii.gz",
        "deconv_iter_*.nii.gz",
        "recon_iter_*.nii.gz",
        "dtv_iter_*.nii.gz",
        "rl_deconv_iter_*.nii.gz",
        "rl_iter_*.nii.gz",
    ]

    available_files = []
    for pattern in all_patterns:
        available_files.extend(result_dir.glob(pattern))

    if available_files:
        # Extract iteration numbers from filenames
        import re
        iter_files = []
        for f in available_files:
            # Try to extract iteration number from filename
            match = re.search(r'iter_(\d+)', f.stem)
            if match:
                iter_num = int(match.group(1))
                iter_files.append((iter_num, f))

        if iter_files:
            # Sort by iteration number
            iter_files.sort(key=lambda x: x[0])

            # Find nearest iteration
            nearest = min(iter_files, key=lambda x: abs(x[0] - iteration))
            print(f"  Using nearest iteration {nearest[0]} (requested: {iteration})")
            return nearest[1]

    # Fallback: check for generic deconv output files (e.g., deconv_dtv.nii.gz)
    # These are typically the final reconstruction
    generic_patterns = [
        "deconv_dtv.nii.gz",
        "deconv_kernel.nii.gz",
        "deconv_rl.nii.gz",
    ]

    for pattern in generic_patterns:
        filepath = result_dir / pattern
        if filepath.exists():
            print(f"  Using generic reconstruction file: {pattern}")
            return filepath

    return None


def parse_method_type(experiment_name: str) -> str:
    """Parse method type from experiment directory name."""
    if '_dtv_' in experiment_name:
        return 'RL-dTV'
    elif '_hkrl_' in experiment_name:
        return 'HKRL'
    elif '_krl_' in experiment_name:
        return 'KRL'
    elif '_rl_' in experiment_name:
        return 'RL'
    else:
        return 'Unknown'


def extract_parameters(experiment_name: str) -> Dict[str, str]:
    """Extract parameters from experiment directory name."""
    params = {}

    # Extract alpha for DTV
    alpha_match = re.search(r'alpha-([0-9p]+)', experiment_name)
    if alpha_match:
        params['alpha'] = alpha_match.group(1).replace('p', '.')

    # Extract sigma values
    sigma_anat_match = re.search(r'sigma_anat-([0-9p]+)', experiment_name)
    if sigma_anat_match:
        params['sigma_anat'] = sigma_anat_match.group(1).replace('p', '.')

    sigma_emission_match = re.search(r'sigma_emission-([0-9p]+)', experiment_name)
    if sigma_emission_match:
        params['sigma_emission'] = sigma_emission_match.group(1).replace('p', '.')

    # Extract freeze iteration
    freeze_match = re.search(r'freeze_iteration-([0-9]+)', experiment_name)
    if freeze_match:
        params['freeze_iter'] = freeze_match.group(1)

    return params


def get_methods_to_plot(dataset: str = 'spheres'):
    """
    Get methods to plot based on configuration.

    Returns list of dicts with 'experiment', 'iteration', 'method', 'result_dir'
    """
    # Brain always uses manual selection (no ground truth)
    if USE_BEST_RESULTS and dataset == 'spheres':
        # Use automatic best result finding for spheres
        return find_best_results(dataset)
    else:
        # Use manual selection (always for brain, or when USE_BEST_RESULTS=False for spheres)
        methods_dict = SPHERES_METHODS if dataset == 'spheres' else BRAIN_METHODS

        results = []
        for exp_name, iteration in methods_dict.items():
            result_dir = RESULTS_DIR / exp_name
            if not result_dir.exists():
                print(f"WARNING: Directory not found: {exp_name}")
                continue

            method = parse_method_type(exp_name)
            params = extract_parameters(exp_name)

            results.append({
                'method': method,
                'experiment': exp_name,
                'iteration': iteration,
                'params': params,
                'result_dir': result_dir,
                'min_nrmse': None  # Not available for manual selection
            })

        return pd.DataFrame(results)


def find_best_results(dataset: str = 'spheres') -> pd.DataFrame:
    """
    Find best results for each method type.

    Args:
        dataset: 'spheres' or 'mk-h001'

    Returns:
        DataFrame with columns: method, experiment, min_nrmse, iteration, params
    """
    # Find all nrmse CSV files for the dataset
    nrmse_files = list(RESULTS_DIR.glob(f"{dataset}*/*nrmse.csv"))

    results = []
    for csv_file in nrmse_files:
        try:
            df = pd.read_csv(csv_file)

            # Find minimum NRMSE
            min_idx = df['nrmse'].idxmin()
            min_nrmse = df.loc[min_idx, 'nrmse']
            min_iter = int(df.loc[min_idx, 'iteration'])

            experiment_name = csv_file.parent.name
            method = parse_method_type(experiment_name)
            params = extract_parameters(experiment_name)

            results.append({
                'method': method,
                'experiment': experiment_name,
                'min_nrmse': min_nrmse,
                'iteration': min_iter,
                'params': params,
                'result_dir': csv_file.parent
            })
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Return empty DataFrame if no results
    if len(df_results) == 0:
        return pd.DataFrame()

    # For each method, find the best result
    best_results = []
    for method in ['RL', 'KRL', 'HKRL', 'RL-dTV']:
        method_df = df_results[df_results['method'] == method]
        if len(method_df) > 0:
            best_idx = method_df['min_nrmse'].idxmin()
            best_results.append(method_df.loc[best_idx])

    return pd.DataFrame(best_results)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_dataset_overview(data_dict: Dict[str, np.ndarray],
                          title: str,
                          output_filename: str,
                          axis: int = 1,
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None):
    """
    Plot dataset overview with multiple images side by side.

    Args:
        data_dict: Dictionary with image names and 3D arrays
        title: Overall figure title
        output_filename: Output filename
        axis: Axis along which to take central slice
        vmin, vmax: Color scale limits
    """
    n_images = len(data_dict)
    fig, axes = plt.subplots(1, n_images, figsize=(3*n_images, 3))

    if n_images == 1:
        axes = [axes]

    for idx, (ax, (name, data)) in enumerate(zip(axes, data_dict.items())):
        slice_data = get_central_slice(data, axis=axis)
        # if mri or t1 in name.lower(), use different colormap
        if 'mri' in name.lower() or 't1' in name.lower():
            im = ax.imshow(slice_data.T, cmap='gray', origin='lower',
                           vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(slice_data.T, cmap=cmasher.fall, origin='lower',
                        vmin=vmin, vmax=vmax)
        ax.set_title(name.replace('_', ' ').title(), fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add subplot label (a), (b), (c), etc.
        label = chr(ord('a') + idx)  # Convert index to letter
        ax.text(0.02, 0.98, f'({label})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left',
                color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_method_comparison(reconstructions: Dict[str, np.ndarray],
                          ground_truth: Optional[np.ndarray],
                          title: str,
                          output_filename: str,
                          axis: int = 1,
                          vmax_div: Optional[float] = None,
                          blurred_pet: Optional[np.ndarray] = None):
    """
    Plot comparison of different reconstruction methods.

    Args:
        reconstructions: Dict mapping method name to reconstruction array
        ground_truth: Ground truth array (optional, for spheres)
        title: Figure title
        output_filename: Output filename
        axis: Axis for slicing
        vmax_div: Optional divider for vmax (for manual adjustment)
        blurred_pet: Optional blurred PET to use for color scale
    """
    methods = list(reconstructions.keys())
    n_methods = len(methods)

    n_cols = n_methods

    fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 3))

    if n_cols == 1:
        axes = [axes]

    # Determine common color scale
    vmin = 0

    # Use blurred PET's color scale if provided, otherwise use reconstructions
    if blurred_pet is not None:
        blurred_slice = get_central_slice(blurred_pet, axis=axis)
        vmax = blurred_slice.max()
    else:
        all_data = list(reconstructions.values())
        vmax = min(get_central_slice(d, axis).max() for d in all_data)

    # Apply vmax_div if provided
    if vmax_div is not None:
        vmax = vmax / vmax_div

    # Plot ground truth if available
    col_idx = 0

    # Plot reconstructions
    for method_name in methods:
        recon_data = reconstructions[method_name]
        slice_data = get_central_slice(recon_data, axis=axis)

        im = axes[col_idx].imshow(slice_data.T, cmap=cmasher.fall, origin='lower',
                                 vmin=vmin, vmax=vmax)
        axes[col_idx].set_title(method_name, fontsize=14, fontweight='bold')
        axes[col_idx].axis('off')
        plt.colorbar(im, ax=axes[col_idx], fraction=0.046, pad=0.04)

        # Add subplot label (a), (b), (c), etc.
        label = chr(ord('a') + col_idx)  # Convert index to letter
        axes[col_idx].text(0.02, 0.98, f'({label})', transform=axes[col_idx].transAxes,
                           fontsize=16, fontweight='bold', va='top', ha='left',
                           color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        col_idx += 1

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_nrmse_curves(nrmse_data: Dict[str, pd.DataFrame],
                     title: str,
                     output_filename: str,
                     mark_best_iterations: Optional[Dict[str, int]] = None):
    """
    Plot NRMSE convergence curves for different methods.

    Args:
        nrmse_data: Dict mapping method name to DataFrame with 'iteration' and 'nrmse'
        title: Figure title
        output_filename: Output filename
        mark_best_iterations: Dict mapping method name to iteration number to mark
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = {'RL': 'blue', 'KRL': 'green', 'HKRL': 'orange', 'RL-dTV': 'red'}

    for method_name, df in nrmse_data.items():
        color = colors.get(method_name, 'black')
        # Add 1 to iteration to allow log scale (iteration 0 -> 1, etc.)
        # But only if the method starts from iteration 0
        min_iter = df['iteration'].min()
        offset = 1 if min_iter == 0 else 0
        ax.plot(df['iteration'] + offset, df['nrmse'], label=method_name,
               color=color, linewidth=2)

        # Mark best iteration if provided
        if mark_best_iterations and method_name in mark_best_iterations:
            best_iter = mark_best_iterations[method_name]
            best_nrmse = df[df['iteration'] == best_iter]['nrmse'].values[0]
            ax.plot(best_iter + offset, best_nrmse, 'o', color=color,
                   markersize=10, markeredgecolor='black', markeredgewidth=2,
                   label=f'{method_name} (best)')

    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('NRMSE', fontsize=14)
    ax.set_xscale('log')
    #ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')  # Show grid for both major and minor ticks

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_objective_curves(objective_data: Dict[str, pd.DataFrame],
                         title: str,
                         output_filename: str,
                         mark_best_iterations: Optional[Dict[str, int]] = None):
    """
    Plot objective function convergence curves for different methods.

    Args:
        objective_data: Dict mapping method name to DataFrame with 'iteration' and 'objective'
        title: Figure title
        output_filename: Output filename
        mark_best_iterations: Dict mapping method name to iteration number to mark
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = {'RL': 'blue', 'KRL': 'green', 'HKRL': 'orange', 'RL-dTV': 'red'}

    for method_name, df in objective_data.items():
        color = colors.get(method_name, 'black')
        # Add 1 to iteration to allow log scale (iteration 0 -> 1, etc.)
        # But only if the method starts from iteration 0
        min_iter = df['iteration'].min()
        offset = 1 if min_iter == 0 else 0
        ax.plot(df['iteration'] + offset, df['objective'], label=method_name,
               color=color, linewidth=2)

        # Mark best iteration if provided (best = iteration with lowest NRMSE)
        if mark_best_iterations and method_name in mark_best_iterations:
            best_iter = mark_best_iterations[method_name]
            best_objective = df[df['iteration'] == best_iter]['objective'].values[0]
            ax.plot(best_iter + offset, best_objective, 'o', color=color,
                   markersize=10, markeredgecolor='black', markeredgewidth=2,
                   label=f'{method_name} (best)')

    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Objective', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')  # Show grid for both major and minor ticks

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_profiles(reconstructions: Dict[str, np.ndarray],
                 ground_truth: Optional[np.ndarray],
                 blurred_pet: np.ndarray,
                 title: str,
                 output_filename: str,
                 y_index: int,
                 axis: int = 0):
    """
    Plot horizontal profiles through the images at a specific y-index.

    Args:
        reconstructions: Dict mapping method name to reconstruction array
        ground_truth: Ground truth array (optional)
        blurred_pet: Blurred PET image for reference
        title: Figure title
        output_filename: Output filename
        y_index: Y-axis index to extract profile from (row index in the 2D slice)
        axis: Axis for slicing (0 for z-slice, 1 for y-slice)
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Get the slice for all images
    if axis == 0:
        # Central z-slice profiles
        mid_z = blurred_pet.shape[0] // 2
        if ground_truth is not None:
            gt_slice = ground_truth[mid_z, :, :]
        blurred_slice = blurred_pet[mid_z, :, :]
        recon_slices = {name: data[mid_z, :, :] for name, data in reconstructions.items()}
    elif axis == 1:
        # Central y-slice profiles
        mid_y = blurred_pet.shape[1] // 2
        if ground_truth is not None:
            gt_slice = ground_truth[:, mid_y, :]
        blurred_slice = blurred_pet[:, mid_y, :]
        recon_slices = {name: data[:, mid_y, :] for name, data in reconstructions.items()}
    elif axis == 2:
        # Central slice along axis 2 (for spheres)
        mid_idx = blurred_pet.shape[2] // 2
        if ground_truth is not None:
            gt_slice = ground_truth[:, :, mid_idx]
        blurred_slice = blurred_pet[:, :, mid_idx]
        recon_slices = {name: data[:, :, mid_idx] for name, data in reconstructions.items()}
    else:
        raise ValueError(f"Axis {axis} not supported")

    # Extract profiles at y_index (vertical profile through column y_index in transposed display)
    # Since images are displayed as .T, we need to extract slice[:, y_index] to match axhline
    x_coords = np.arange(blurred_slice.shape[0])  # X-axis coordinates

    # Plot ground truth if available
    if ground_truth is not None:
        gt_profile = gt_slice[:, y_index]  # Vertical profile (matches axhline after transpose)
        ax.plot(x_coords, gt_profile, 'k-', linewidth=2.5, label='Ground Truth', zorder=10)

    # Plot blurred PET
    blurred_profile = blurred_slice[:, y_index]  # Vertical profile
    ax.plot(x_coords, blurred_profile, '--', color='gray', linewidth=2,
            label='Blurred PET', alpha=0.7)

    # Plot reconstructions
    colors = {'RL': 'blue', 'KRL': 'green', 'HKRL': 'orange', 'RL-dTV': 'red'}
    for method_name, recon_slice in recon_slices.items():
        # Extract method name without iteration info
        base_method = method_name.split('\n')[0]
        color = colors.get(base_method, 'black')
        profile = recon_slice[:, y_index]  # Vertical profile
        ax.plot(x_coords, profile, '-', color=color, linewidth=2, label=method_name)

    ax.set_xlabel('X Position (voxels)', fontsize=14)
    ax.set_ylabel('Intensity', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(best_results: pd.DataFrame,
                        output_filename: str):
    """
    Create a formatted summary table of best results.

    Args:
        best_results: DataFrame with best results for each method
        output_filename: Output CSV filename
    """
    # Create a clean table
    table_data = []

    for _, row in best_results.iterrows():
        entry = {
            'Method': row['method'],
            'Min NRMSE': f"{row['min_nrmse']:.6f}",
            'Iteration': row['iteration'],
        }

        # Add parameter columns
        params = row['params']
        if 'alpha' in params:
            entry['Alpha'] = params['alpha']
        if 'sigma_anat' in params:
            entry['Sigma Anat'] = params['sigma_anat']
        if 'sigma_emission' in params:
            entry['Sigma Emission'] = params['sigma_emission']
        if 'freeze_iter' in params:
            entry['Freeze Iter'] = params['freeze_iter']

        table_data.append(entry)

    df_table = pd.DataFrame(table_data)

    # Save to CSV
    output_path = OUTPUT_DIR / output_filename
    df_table.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Also print to console
    print("\n" + "="*100)
    print(df_table.to_string(index=False))
    print("="*100 + "\n")

    return df_table


# ============================================================================
# Main execution functions
# ============================================================================

def generate_figure1_spheres_overview():
    """Generate Figure 1: Spheres dataset overview."""
    print("\n=== Generating Figure 1: Spheres Dataset Overview ===")
    data = load_spheres_data()
    plot_dataset_overview(
        data,
        title="Spheres Phantom Dataset - Central Slice (Axis 0)",
        output_filename="fig1_spheres_overview.png",
        axis=2
    )


def generate_figure2_brain_overview():
    """Generate Figure 2: Brain dataset overview."""
    print("\n=== Generating Figure 2: Brain Dataset Overview ===")
    data = load_brain_data()
    plot_dataset_overview(
        data,
        title="Brain (MK-H001) Dataset - Central Slice (Axis 1)",
        output_filename="fig2_brain_overview.png",
        axis=1
    )


def generate_figure3_spheres_comparison():
    """Generate Figure 3: Best reconstruction comparison for spheres."""
    print("\n=== Generating Figure 3: Spheres Best Reconstructions ===")

    # Get methods to plot (either best or manual)
    best_results = get_methods_to_plot('spheres')
    print("\nBest results found:")
    print(best_results[['method', 'min_nrmse', 'iteration']])

    # Load ground truth and blurred PET
    data = load_spheres_data()
    ground_truth = data['ground_truth']
    blurred_pet = data['blurred_pet']

    # Load best reconstructions
    reconstructions = {}
    for _, row in best_results.iterrows():
        method = row['method']
        result_dir = row['result_dir']
        iteration = row['iteration']

        recon_file = find_reconstruction_file(result_dir, iteration)
        if recon_file:
            reconstructions[f"{method}\n(iter {iteration})"] = load_nifti_data(recon_file)
            print(f"Loaded {method} reconstruction from {recon_file}")
        else:
            print(f"WARNING: Could not find reconstruction file for {method} at iteration {iteration}")

    # Plot comparison
    plot_method_comparison(
        reconstructions,
        ground_truth,
        title="Spheres: Best Reconstructions Comparison",
        output_filename="fig3_spheres_best_comparison.png",
        axis=2,
        blurred_pet=blurred_pet
    )


def generate_figure4_spheres_nrmse():
    """Generate Figure 4: NRMSE curves for spheres."""
    print("\n=== Generating Figure 4: Spheres NRMSE Curves ===")

    # Get methods to plot (either best or manual)
    best_results = get_methods_to_plot('spheres')

    # Load NRMSE data
    nrmse_data = {}
    best_iterations = {}

    for _, row in best_results.iterrows():
        method = row['method']
        result_dir = row['result_dir']

        # Find NRMSE CSV file
        nrmse_files = list(result_dir.glob("*nrmse.csv"))
        if nrmse_files:
            df = pd.read_csv(nrmse_files[0])
            nrmse_data[method] = df
            best_iterations[method] = row['iteration']
            print(f"Loaded NRMSE data for {method}")

    # Plot curves
    plot_nrmse_curves(
        nrmse_data,
        title="Spheres: NRMSE Convergence",
        output_filename="fig4_spheres_nrmse_curves.png",
        mark_best_iterations=best_iterations
    )


def generate_figure4b_spheres_objective():
    """Generate Figure 4b: Objective function curves for spheres."""
    print("\n=== Generating Figure 4b: Spheres Objective Curves ===")

    # Get methods to plot (either best or manual)
    best_results = get_methods_to_plot('spheres')

    # Load objective data
    objective_data = {}
    best_iterations = {}

    for _, row in best_results.iterrows():
        method = row['method']
        result_dir = row['result_dir']

        # Find objective CSV file - different naming for different methods
        objective_files = list(result_dir.glob("*objective.csv"))
        if objective_files:
            df = pd.read_csv(objective_files[0])
            objective_data[method] = df
            best_iterations[method] = row['iteration']
            print(f"Loaded objective data for {method} from {objective_files[0].name}")
        else:
            print(f"WARNING: No objective file found for {method}")

    # Plot curves
    if objective_data:
        plot_objective_curves(
            objective_data,
            title="Spheres: Objective Function Convergence",
            output_filename="fig4b_spheres_objective_curves.png",
            mark_best_iterations=best_iterations
        )
    else:
        print("WARNING: No objective data found for any method.")


def generate_figure5_brain_reconstructions():
    """Generate Figure 5: Brain reconstructions."""
    print("\n=== Generating Figure 5: Brain Reconstructions ===")
    print("Note: Brain (mk-h001) has no ground truth, so showing example reconstructions.")

    # Load brain data
    data = load_brain_data()
    blurred_pet = data['blurred_pet']

    # Brain ALWAYS uses manual BRAIN_METHODS configuration
    brain_methods = get_methods_to_plot('mk-h001')
    brain_reconstructions = {}

    for _, row in brain_methods.iterrows():
        method = row['method']
        result_dir = row['result_dir']
        iteration = row['iteration']

        recon_file = find_reconstruction_file(result_dir, iteration)

        if recon_file:
            brain_reconstructions[f"{method}\n(iter {iteration})"] = load_nifti_data(recon_file)
            print(f"Loaded {method} reconstruction from {recon_file}")
        else:
            print(f"WARNING: Could not find reconstruction file for {method} at iteration {iteration}")

    # Only plot if we have reconstructions
    if len(brain_reconstructions) > 0:
        plot_method_comparison(
            brain_reconstructions,
            None,  # No ground truth for brain
            title="Brain (MK-H001): Example Reconstructions",
            output_filename="fig5_brain_example_comparison.png",
            axis=1,
            blurred_pet=blurred_pet
        )
    else:
        print("WARNING: No reconstruction files found for brain results.")


def plot_image_with_profile_line(image_slice: np.ndarray,
                                 y_index: int,
                                 title: str,
                                 output_filename: str):
    """
    Plot an image slice with a line indicating where the profile is taken.

    Args:
        image_slice: 2D image slice
        y_index: Y-coordinate where profile line should be drawn
        title: Figure title
        output_filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display image
    im = ax.imshow(image_slice.T, cmap='gray', origin='lower')

    # Draw horizontal line at y_index
    ax.axhline(y=y_index, color='red', linewidth=2, linestyle='--', label=f'Profile at y={y_index}')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (voxels)', fontsize=12)
    ax.set_ylabel('Y Position (voxels)', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_figure6_spheres_profiles():
    """Generate Figure 6: Spheres profiles at central and peak locations."""
    print("\n=== Generating Figure 6: Spheres Profiles ===")

    # Get methods to plot (either best or manual)
    best_results = get_methods_to_plot('spheres')

    # Load ground truth and blurred PET
    data = load_spheres_data()
    ground_truth = data['ground_truth']
    blurred_pet = data['blurred_pet']

    # Load best reconstructions
    reconstructions = {}
    for _, row in best_results.iterrows():
        method = row['method']
        result_dir = row['result_dir']
        iteration = row['iteration']

        recon_file = find_reconstruction_file(result_dir, iteration)
        if recon_file:
            reconstructions[f"{method}\n(iter {iteration})"] = load_nifti_data(recon_file)
            print(f"Loaded {method} reconstruction")

    # Get central slice along axis 2 (like fig3) to show spheres properly
    mid_slice_idx = blurred_pet.shape[2] // 2
    blurred_slice = blurred_pet[:, :, mid_slice_idx]
    gt_slice = ground_truth[:, :, mid_slice_idx]

    # Profile 1: Central y location
    central_y = blurred_slice.shape[0] // 2  # Central row

    # Create annotated image showing profile location
    plot_image_with_profile_line(
        gt_slice,
        central_y,
        title="",
        output_filename="fig6a_spheres_profile_central_location.png"
    )

    # Create profile plot
    plot_profiles(
        reconstructions,
        ground_truth,
        blurred_pet,
        title="",
        output_filename="fig6a_spheres_profile_central.png",
        y_index=central_y,
        axis=2
    )

    # Profile 2: Location of highest pixel value in the blurred slice
    # Find the location of the maximum pixel value
    max_location = np.unravel_index(np.argmax(blurred_slice), blurred_slice.shape)
    peak_y = max_location[0]  # Y-coordinate of the maximum pixel

    print(f"  Peak pixel location: y={peak_y}, x={max_location[1]} (value={blurred_slice[peak_y, max_location[1]]:.2f})")

    # Create annotated image showing profile location
    plot_image_with_profile_line(
        blurred_slice,
        peak_y,
        title="",
        output_filename="fig6b_spheres_profile_peak_location.png"
    )

    # Create profile plot
    plot_profiles(
        reconstructions,
        ground_truth,
        blurred_pet,
        title="",
        output_filename="fig6b_spheres_profile_peak.png",
        y_index=peak_y,
        axis=2
    )


def plot_image_with_multiple_profile_lines(image_slice: np.ndarray,
                                           y_indices: list,
                                           labels: list,
                                           title: str,
                                           output_filename: str):
    """
    Plot an image slice with multiple lines indicating profile locations.

    Args:
        image_slice: 2D image slice
        y_indices: List of y-coordinates where profile lines should be drawn
        labels: List of labels for each profile line
        title: Figure title
        output_filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display image
    im = ax.imshow(image_slice.T, cmap='gray', origin='lower')

    # Draw horizontal lines at each y_index
    colors = ['red', 'blue', 'green', 'orange']
    for i, (y_idx, label) in enumerate(zip(y_indices, labels)):
        color = colors[i % len(colors)]
        ax.axhline(y=y_idx, color=color, linewidth=2, linestyle='--', label=label)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (voxels)', fontsize=12)
    ax.set_ylabel('Y Position (voxels)', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_figure7_brain_profiles():
    """Generate Figure 7: Brain profiles at central +/- 15% locations."""
    print("\n=== Generating Figure 7: Brain Profiles ===")
    print("Note: Brain has no ground truth, showing profiles at central +/- 15%")

    # Load brain data
    data = load_brain_data()
    blurred_pet = data['blurred_pet']

    # Brain ALWAYS uses manual BRAIN_METHODS configuration
    brain_methods = get_methods_to_plot('mk-h001')

    # Load brain reconstructions
    brain_reconstructions = {}
    for _, row in brain_methods.iterrows():
        method = row['method']
        result_dir = row['result_dir']
        iteration = row['iteration']

        recon_file = find_reconstruction_file(result_dir, iteration)

        if recon_file:
            brain_reconstructions[f"{method}\n(iter {iteration})"] = load_nifti_data(recon_file)
            print(f"Loaded {method} reconstruction")

    # Get central y-slice for visualization
    mid_y_slice = blurred_pet.shape[1] // 2
    brain_slice = blurred_pet[:, mid_y_slice, :]

    # Get y positions for profiles
    mid_y = blurred_pet.shape[1] // 2
    offset = int(blurred_pet.shape[1] * 0.15)  # 15% offset

    y1 = mid_y - offset
    y2 = mid_y + offset

    # Create annotated image showing both profile locations
    plot_image_with_multiple_profile_lines(
        brain_slice,
        [y1, y2],
        [f'Profile at y={y1} (central - 15%)', f'Profile at y={y2} (central + 15%)'],
        title="",
        output_filename="fig7_brain_profile_locations.png"
    )

    # Profile 1: Central - 15%
    plot_profiles(
        brain_reconstructions,
        None,  # No ground truth
        blurred_pet,
        title="",
        output_filename="fig7a_brain_profile_minus15.png",
        y_index=y1,
        axis=1
    )

    # Profile 2: Central + 15%
    plot_profiles(
        brain_reconstructions,
        None,  # No ground truth
        blurred_pet,
        title="",
        output_filename="fig7b_brain_profile_plus15.png",
        y_index=y2,
        axis=1
    )


def generate_figure5b_brain_reconstructions_axis2():
    """Generate Figure 5b: Brain reconstructions along axis 2 (same as spheres)."""
    print("\n=== Generating Figure 5b: Brain Reconstructions (Axis 2) ===")
    print("Note: Brain (mk-h001) has no ground truth, so showing example reconstructions.")

    # Load brain data
    data = load_brain_data()
    blurred_pet = data['blurred_pet']

    # Brain ALWAYS uses manual BRAIN_METHODS configuration
    brain_methods = get_methods_to_plot('mk-h001')
    brain_reconstructions = {}

    for _, row in brain_methods.iterrows():
        method = row['method']
        result_dir = row['result_dir']
        iteration = row['iteration']

        recon_file = find_reconstruction_file(result_dir, iteration)

        if recon_file:
            brain_reconstructions[f"{method}\n(iter {iteration})"] = load_nifti_data(recon_file)
            print(f"Loaded {method} reconstruction from {recon_file}")
        else:
            print(f"WARNING: Could not find reconstruction file for {method} at iteration {iteration}")

    # Only plot if we have reconstructions
    if len(brain_reconstructions) > 0:
        plot_method_comparison(
            brain_reconstructions,
            None,  # No ground truth for brain
            title="Brain (MK-H001): Example Reconstructions (Axis 2)",
            output_filename="fig5b_brain_example_comparison_axis2.png",
            axis=2,
            blurred_pet=blurred_pet
        )
    else:
        print("WARNING: No reconstruction files found for brain results.")


def generate_table1_summary():
    """Generate Table 1: Summary of best results."""
    print("\n=== Generating Table 1: Summary Tables ===")

    # Spheres summary
    spheres_best = get_methods_to_plot('spheres')
    create_summary_table(spheres_best, "table1_spheres_summary.csv")

    # Brain summary - always use manual methods (no ground truth)
    # Note: Brain table won't have NRMSE values
    print("\nNote: Brain has no ground truth, so no NRMSE table generated.")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("POSTER FIGURE GENERATION")
    print("="*100)

    # Generate all figures and tables
    generate_figure1_spheres_overview()
    generate_figure2_brain_overview()
    generate_figure3_spheres_comparison()
    generate_figure4_spheres_nrmse()
    generate_figure4b_spheres_objective()
    generate_figure5_brain_reconstructions()
    generate_figure5b_brain_reconstructions_axis2()
    generate_figure6_spheres_profiles()
    generate_figure7_brain_profiles()
    generate_table1_summary()

    print("\n" + "="*100)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("="*100)

    # To generate individual figures, uncomment the desired function:
    # generate_figure1_spheres_overview()
    # generate_figure2_brain_overview()
    # generate_figure3_spheres_comparison()
    # generate_figure4_spheres_nrmse()
    # generate_figure4b_spheres_objective()
    # generate_figure5_brain_reconstructions()
    # generate_figure6_spheres_profiles()
    # generate_figure7_brain_profiles()
    # generate_table1_summary()
