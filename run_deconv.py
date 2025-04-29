# %% [markdown]
# # PET Deconvolution Exercise with Global Parameters (including Kernel Parameters)
#
# In this notebook, key parameters are defined and displayed at the top.
# These global constants (e.g., iteration numbers, seed values, kernel parameters)
# are then used throughout the analysis.

# %%
# Global Parameters
NOISE_SEED = 5
BW_SEED = 1337
RL_ITERATIONS_KERNEL = 20     # Iterations for kernel-guided RL deconvolution
RL_ITERATIONS_STANDARD = 20     # Iterations for standard RL deconvolution
DTV_ITERATIONS = 20          # Iterations for directional TV deconvolution (MAPRL)
ALPHA = 0.1                     # Regularization parameter for TV prior
STEP_SIZE = 0.1                  # Step size for the MAPRL algorithm
RELAXATION_ETA = 0.01            # Relaxation parameter for MAPRL
UPDATE_OBJ_INTERVAL = 1          # Objective function update interval
PSF_KERNEL_SIZE = 5              # Size of the PSF kernel along each dimension
FWHM_VALUES = [4.0, 4.0, 4.0]   # FWHM values for the PSF kernel in mm
EMISSION_PATH = "/home/sam/working/others/Kjell/KRL/data/MK-H001/MK-H001_PET_MNI.nii"
GUIDANCE_PATH = "/home/sam/working/others/Kjell/KRL/data/MK-H001/MK-H001_T1_MNI.nii"
BACKEND='numba' # backend for kernel operator. Won't fit on GPU so either 'numba' or 'python'
SHOW_PLOTS = False
FLIP_GUIDANCE = True # flip the guidance image along axis 1
DO_RL = False # do RL deconvolution
DO_KRL = True # do kernel guided RL deconvolution
DO_DRL = False # do directional RL deconvolution


# Kernel Global Parameters
KERNEL_NUM_NEIGHBOURS = 5
KERNEL_SIGMA_ANAT = 0.5
KERNEL_SIGMA_DIST = 3.0
KERNEL_TYPE = 'neighbourhood'
KERNEL_NORMALIZE_FEATURES = True
KERNEL_NORMALIZE_KERNEL = True
USE_MASK = True
MASK_K = 75
RECALC_MASK = False

# Display global parameters for reference.
for name, val in [
    ("NOISE_SEED", NOISE_SEED),
    ("BW_SEED", BW_SEED),
    ("RL_ITERATIONS_KERNEL", RL_ITERATIONS_KERNEL),
    ("RL_ITERATIONS_STANDARD", RL_ITERATIONS_STANDARD),
    ("DTV_ITERATIONS", DTV_ITERATIONS),
    ("ALPHA", ALPHA),
    ("STEP_SIZE", STEP_SIZE),
    ("RELAXATION_ETA", RELAXATION_ETA),
    ("UPDATE_OBJ_INTERVAL", UPDATE_OBJ_INTERVAL),
    ("PSF_KERNEL_SIZE", PSF_KERNEL_SIZE),
    ("KERNEL_NUM_NEIGHBOURS", KERNEL_NUM_NEIGHBOURS),
    ("KERNEL_SIGMA_ANAT", KERNEL_SIGMA_ANAT),
    ("KERNEL_SIGMA_DIST", KERNEL_SIGMA_DIST),
    ("KERNEL_TYPE", KERNEL_TYPE),
    ("KERNEL_NORMALIZE_FEATURES", KERNEL_NORMALIZE_FEATURES),
    ("KERNEL_NORMALIZE_KERNEL", KERNEL_NORMALIZE_KERNEL)
]:
    print(f"{name} = {val}")

# %%
import os
import re
import sys
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import sirf.STIR as pet

from cil.optimisation.operators import BlurringOperator
import cil.optimisation.operators as op
import cil.optimisation.functions as fn
import cil.optimisation.algorithms as alg
from cil.utilities.display import show2D

# Redirect STIR messages (if needed)
msg = pet.MessageRedirector()

# %%
# Determine script and data directories.
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# %% import modules from this repository
sys.path.append(os.path.join(script_dir, 'src'))
from my_kem import get_kernel_operator
from directional_operator import DirectionalOperator
from map_rl import MAPRL
from gaussian_blurring import create_gaussian_blur
# if torch available, use it
try:
    from gradient import GradientOperator
except ImportError:
    from cil.optimisation.operators import GradientOperator

# Set the random seed for reproducibility.
np.random.seed(NOISE_SEED)

def fwhm_to_sigma(fwhm):
    """
    Convert full-width-at-half-maximum (FWHM) to the Gaussian standard deviation (sigma).
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def psf(kernel_size, fwhm, voxel_size=(1, 1, 1)):
    """
    Construct a normalized 3D Gaussian PSF kernel.
    
    Parameters:
        kernel_size (int): Size of the kernel along each dimension.
        fwhm (list/tuple): FWHM values for each axis.
        voxel_size (tuple): The voxel size in each dimension.
        
    Returns:
        np.ndarray: The normalized 3D PSF kernel.
    """
    sigma_voxels = [fwhm_to_sigma(fwhm[i]) / voxel_size[i] for i in range(3)]
    axes = [np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
            for _ in range(3)]
    gauss = [np.exp(-0.5 * np.square(ax) / np.square(sigma_voxels[i]))
             for i, ax in enumerate(axes)]
    kernel_3d = np.outer(gauss[0], gauss[1]).reshape(kernel_size, kernel_size, 1) * \
                gauss[2].reshape(1, 1, kernel_size)
    return kernel_3d / np.sum(kernel_3d)

# %%
if not SHOW_PLOTS:
    # Override plt.show with a no-op
    plt.show = lambda *args, **kwargs: None

images = {}
# Load OSEM and PSF images.
images['OSEM'] = pet.ImageData(EMISSION_PATH)
images['T1'] = pet.ImageData(GUIDANCE_PATH)

if FLIP_GUIDANCE:
    # need to flip the T1 image along axis1
    t1_arr = images['T1'].as_array()
    t1_arr = np.flip(t1_arr, axis=1)
    images['T1'].fill(t1_arr)

# Display the images
fig0 = show2D([images['OSEM'], images['T1']],
                title=['OSEM', 'T1'],
                origin='upper', num_cols=2,
                fix_range=[(0, 320), (0, 1000)])
fig0.save(os.path.join(data_dir, 'OSEM_T1.png'))

# display the images at a wuarter sluce
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
ax[0,0].imshow(images['OSEM'].as_array()[images['OSEM'].shape[0]//4], cmap='gray')
ax[0,0].set_title('OSEM')
ax[0,1].imshow(images['T1'].as_array()[images['T1'].shape[0]//4], cmap='gray')
ax[0,1].set_title('T1')
#now 3/4
ax[1,0].imshow(images['OSEM'].as_array()[3*images['OSEM'].shape[0]//4], cmap='gray')
ax[1,0].set_title('OSEM')
ax[1,1].imshow(images['T1'].as_array()[3*images['T1'].shape[0]//4], cmap='gray')
ax[1,1].set_title('T1')
plt.savefig(os.path.join(data_dir, 'OSEM_T1_quarter_slices.png'))
plt.close()

print(f"size of OSEM image: {images['OSEM'].shape}")
print(f"size of T1 image: {images['T1'].shape}")

# Generate the PSF kernel and set up the blurring operator.
psf_kernel = psf(PSF_KERNEL_SIZE, fwhm=FWHM_VALUES, voxel_size=images['OSEM'].voxel_sizes())

# if cupy is available, use it
try:
    blurring_operator = create_gaussian_blur(
        fwhm_to_sigma(FWHM_VALUES),
        images['OSEM'],
        backend='auto',
    )
except ImportError:
    print("cupy not available, using numpy")
    blurring_operator = BlurringOperator(
        psf_kernel, 
        images['OSEM']
    )


# %%
def richardson_lucy(observed, blur_op, iterations, epsilon=1e-10,
                     ground_truth=None, kernel_operator=None):
    """
    Perform Richardson–Lucy deconvolution.
    
    Parameters:
        observed (pet.ImageData): The blurred image.
        blur_op: The blurring operator.
        iterations (int): Number of iterations.
        epsilon (float): Small constant to avoid division by zero.
        ground_truth (pet.ImageData, optional): Ground truth image (for RMSE computation).
        kernel_operator (Operator, optional): Additional operator for guided deconvolution.
    
    Returns:
        tuple: (deconvolved_image, objective_values, [rmse_values])
    """
    objective_values = []
    rmse_values = [] if ground_truth is not None else None

    if kernel_operator is not None:
        blur_op = op.CompositionOperator(blur_op, kernel_operator)
        sensitivity = kernel_operator.adjoint(observed.get_uniform_copy(1))
    else:
        kernel_operator = op.IdentityOperator(observed)
        sensitivity = observed.get_uniform_copy(1)

    current_estimate = observed.clone()
    estimated_blurred = blur_op.direct(current_estimate)

    for i in range(iterations):
        # Update step.
        current_estimate *= blur_op.adjoint(observed / (estimated_blurred + epsilon))
        current_estimate /= (sensitivity  + epsilon)
        if ground_truth is not None:
            error = kernel_operator.direct(current_estimate) - ground_truth
            rmse = np.sqrt((error.power(2)).sum())
            rmse_values.append(rmse)
            print(f"Iteration: {i}, RMSE: {rmse}", end="\r")
        estimated_blurred = blur_op.direct(current_estimate)
        obj_value = (estimated_blurred - (observed * (estimated_blurred + epsilon).log())).sum()
        objective_values.append(obj_value)
        print(f"Iteration: {i}, Objective: {obj_value}", end="\r")

    if ground_truth is not None:
        return current_estimate, objective_values, rmse_values

    return current_estimate, objective_values

# %%
if DO_RL:
    # Run standard Richardson–Lucy deconvolution (without kernel guidance).
    deconv_rl, obj_values_rl = richardson_lucy(
        images['OSEM'], blurring_operator,
        iterations=RL_ITERATIONS_STANDARD,
        ground_truth=None
    )

# %%
if DO_RL:
    # Display comparison for standard RL deconvolution.
    fig3 = show2D([deconv_rl, images['OSEM']],
                title=['Deconvolved (RL)', 'OSEM'],
                origin='upper', num_cols=2,
                fix_range=[(0, 320), (0, 320)])
    fig3.save(os.path.join(data_dir, f'deconv_rl_{RL_ITERATIONS_STANDARD}_iter_difference.png'))

    # Plot objective function for standard RL.
    plt.figure()
    plt.plot(obj_values_rl)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.savefig(os.path.join(data_dir, f'deconv_rl_{RL_ITERATIONS_STANDARD}_iter_objective.png'))

    # save the output
    deconv_rl.write(os.path.join(data_dir, f'deconv_rl_{RL_ITERATIONS_STANDARD}_iter.hv'))
    deconv_rl.write(os.path.join(data_dir, f'deconv_rl_{RL_ITERATIONS_STANDARD}_iter.nii'))

# %%
if DO_KRL:
    # Define kernel parameters using the global kernel constants.
    kernel_params = {
        'num_neighbours': KERNEL_NUM_NEIGHBOURS,
        'sigma_anat': KERNEL_SIGMA_ANAT,
        'sigma_dist': KERNEL_SIGMA_DIST,
        'kernel_type': KERNEL_TYPE,
        'normalize_features': KERNEL_NORMALIZE_FEATURES,
        'normalize_kernel': KERNEL_NORMALIZE_KERNEL,
        'use_mask': USE_MASK,
        'mask_k': MASK_K,
        'recalc_mask': RECALC_MASK,
    }

    kernel_op = get_kernel_operator(
        images['OSEM'],
        backend=BACKEND,
    )
    kernel_op.parameters = kernel_params
    kernel_op.set_anatomical_image(images['T1'])

# %%
if DO_KRL:
    # Run kernel-guided Richardson–Lucy deconvolution using global iteration value.
    deconv_kernel_alpha, obj_values_kernel = richardson_lucy(
        images['OSEM'],
        blurring_operator,
        iterations=RL_ITERATIONS_KERNEL,
        ground_truth=None,
        kernel_operator=kernel_op
    )

    # Map the estimated coefficients back using the kernel operator.
    deconv_kernel = kernel_op.direct(deconv_kernel_alpha)

# %%
if DO_KRL:
    # Display comparison: deconvolved image, OSEM, ground truth, and difference image.
    fig2 = show2D([deconv_kernel, images['OSEM']],
                title=['Deconvolved', 'OSEM'],
                origin='upper', num_cols=2,
                fix_range=[(0, 320), (0, 320)])
    fig2.save(os.path.join(data_dir, f'deconv_kernel_{RL_ITERATIONS_KERNEL}_iter_{KERNEL_SIGMA_ANAT}_sigma_{KERNEL_SIGMA_DIST}_kNN_{MASK_K}_difference.png'))

    # Plot objective function values for kernel-guided deconvolution.
    plt.figure()
    plt.plot(obj_values_kernel)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.savefig(os.path.join(data_dir, f'deconv_kernel_{RL_ITERATIONS_KERNEL}_{KERNEL_SIGMA_ANAT}_sigma_{KERNEL_SIGMA_DIST}_kNN_{MASK_K}_objective.png'))

    # save the output
    deconv_kernel.write(os.path.join(data_dir, f'deconv_kernel_{RL_ITERATIONS_KERNEL}_{KERNEL_SIGMA_ANAT}_sigma_{KERNEL_SIGMA_DIST}_kNN_{MASK_K}.hv'))
    deconv_kernel.write(os.path.join(data_dir, f'deconv_kernel_{RL_ITERATIONS_KERNEL}_{KERNEL_SIGMA_ANAT}_sigma_{KERNEL_SIGMA_DIST}_kNN_{MASK_K}.nii'))

    # plot the mask on the central voxel
    voxel = [images['OSEM'].shape[0] // 2, 
            images['OSEM'].shape[1] // 2,
            images['OSEM'].shape[2] // 2]
    mask = kernel_op.mask
    plt.figure()
    plt.plot(mask[voxel[0], voxel[1], voxel[2]], 'o')
    plt.savefig(os.path.join(data_dir, f'mask_{RL_ITERATIONS_KERNEL}_{KERNEL_SIGMA_ANAT}_sigma_{KERNEL_SIGMA_DIST}_kNN_{MASK_K}.png'))


# %%
if DO_DRL:
    # Set up directional TV deconvolution.
    # Data fidelity term.
    f = fn.KullbackLeibler(b=images['OSEM'], eta=images['OSEM'].get_uniform_copy(1e-6))
    df = fn.OperatorCompositionFunction(f, blurring_operator)

    # Prior term: using ALPHA from the global parameters.
    grad = GradientOperator(images['OSEM'])
    grad_ref = grad.direct(images['T1'])
    d_op = op.CompositionOperator(DirectionalOperator(grad_ref), grad)
    prior = ALPHA * fn.OperatorCompositionFunction(fn.SmoothMixedL21Norm(epsilon=1e-4), d_op)

    maprl = MAPRL(initial_estimate=images['OSEM'], data_fidelity=df, prior=prior,
                step_size=STEP_SIZE, relaxation_eta=RELAXATION_ETA,
                update_objective_interval=UPDATE_OBJ_INTERVAL)
    maprl.run(verbose=1, iterations=DTV_ITERATIONS)
    deconv_dtv = maprl.solution

# %%
if DO_DRL:
    # Display deconvolved image and difference image (DTV).
    fig4 = show2D([deconv_dtv, images['OSEM']],
                title=['Deconvolved (DTV)', 'OSEM'],
                origin='upper', num_cols=4,
                fix_range=[(0, 320), (0, 320)])
    fig4.save(os.path.join(data_dir, f'deconv_dtv_{DTV_ITERATIONS}_iter_{ALPHA}_alpha_difference.png'))

    plt.figure(figsize=(15, 5))
    plt.plot(maprl.objective)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.savefig(os.path.join(data_dir, f'deconv_dtv_{DTV_ITERATIONS}_iter_{ALPHA}_alpha_objective.png'))

    # save the output
    deconv_dtv.write(os.path.join(data_dir, f'deconv_dtv_{DTV_ITERATIONS}_iter_{ALPHA}_alpha.hv'))
    deconv_dtv.write(os.path.join(data_dir, f'deconv_dtv_{DTV_ITERATIONS}_iter_{ALPHA}_alpha.nii'))

# %%
# Plot profiles through the central slices for comparison.
center_slice = deconv_kernel.shape[0] // 2
profile_axis = deconv_kernel.shape[2] // 2

plt.figure(figsize=(15, 5))
if DO_RL:
    plt.plot(deconv_rl.as_array()[center_slice, :, profile_axis], label='Deconvolved (RL)')
if DO_KRL:
    plt.plot(deconv_kernel.as_array()[center_slice, :, profile_axis], label='Deconvolved (Kernel)')
if DO_DRL:
    plt.plot(deconv_dtv.as_array()[center_slice, :, profile_axis], label='Deconvolved (DTV)')
plt.plot(images['OSEM'].as_array()[center_slice, :, profile_axis], label='OSEM')
plt.legend()
# long save path with all algorithm hyperparameters
plt.savefig(os.path.join(data_dir, f'profile_comparison_center_{KERNEL_SIGMA_ANAT}_sigma_{KERNEL_SIGMA_DIST}_dist_{ALPHA}_alpha.png'))
# %%
