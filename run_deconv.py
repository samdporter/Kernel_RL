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
RL_ITERATIONS_KERNEL = 100       # Iterations for kernel-guided RL deconvolution
RL_ITERATIONS_STANDARD = 20      # Iterations for standard RL deconvolution
DTV_ITERATIONS = 100             # Iterations for directional TV deconvolution (MAPRL)
ALPHA = 0.02                     # Regularization parameter for TV prior
STEP_SIZE = 0.1                  # Step size for the MAPRL algorithm
RELAXATION_ETA = 0.01            # Relaxation parameter for MAPRL
UPDATE_OBJ_INTERVAL = 1          # Objective function update interval
PSF_KERNEL_SIZE = 5              # Size of the PSF kernel along each dimension

# Kernel Global Parameters
KERNEL_NUM_NEIGHBOURS = 5
KERNEL_SIGMA_ANAT = 1.0
KERNEL_SIGMA_DIST = 1.0
KERNEL_TYPE = 'neighbourhood'
KERNEL_NORMALIZE_FEATURES = True
KERNEL_NORMALIZE_KERNEL = True

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
from src.my_kem import get_kernel_operator
from src.directional_operator import DirectionalOperator
from src.map_rl import MAPRL

# %%
# Determine script and data directories.
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# %%
def find_fwhm_in_image(file_path):
    """
    Run STIR's 'find_fwhm_in_image' utility and parse its output
    to extract full-width-at-half-maximum (FWHM) values for each axis.
    """
    result = subprocess.run(['find_fwhm_in_image', file_path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running command:", result.stderr)
        return None
    fwhm_regex = r"The resolution in (.*) axis is ([\d.]+)"
    matches = re.findall(fwhm_regex, result.stdout)
    fwhm_values = {axis: float(value) for axis, value in matches}
    return fwhm_values

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
# Load images generated in the previous notebook.
image_names = ['PET', 'T1', 'uMap']
images = {}
for name in image_names:
    image_path = os.path.join(data_dir, f'{name}_b{BW_SEED}.hv')
    images[name] = pet.ImageData(image_path)

# Load OSEM and PSF images.
images['OSEM'] = pet.ImageData(os.path.join(data_dir, f'OSEM_b{BW_SEED}_n{NOISE_SEED}.hv'))
images['OSEM_psf'] = pet.ImageData(os.path.join(data_dir, f'OSEM_psf_n{NOISE_SEED}.hv'))

# %%
# Estimate FWHM from the point source measurement.
fwhm_values = list(find_fwhm_in_image(os.path.join(data_dir, f'OSEM_psf_n{NOISE_SEED}.hv')).values())
print(f'FWHM: {fwhm_values}')

# Generate the PSF kernel and set up the blurring operator.
psf_kernel = psf(PSF_KERNEL_SIZE, fwhm=fwhm_values, voxel_size=images['OSEM'].voxel_sizes())
blurring_operator = BlurringOperator(psf_kernel, images['PET'])

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
    else:
        kernel_operator = op.IdentityOperator(observed)

    current_estimate = observed.clone()
    estimated_blurred = blur_op.direct(current_estimate)

    for i in range(iterations):
        # Update step.
        current_estimate *= blur_op.adjoint(observed / (estimated_blurred + epsilon))
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
# Define kernel parameters using the global kernel constants.
kernel_params = {
    'num_neighbours': KERNEL_NUM_NEIGHBOURS,
    'sigma_anat': KERNEL_SIGMA_ANAT,
    'sigma_dist': KERNEL_SIGMA_DIST,
    'kernel_type': KERNEL_TYPE,
    'normalize_features': KERNEL_NORMALIZE_FEATURES,
    'normalize_kernel': KERNEL_NORMALIZE_KERNEL,
}

kernel_op = get_kernel_operator(images['OSEM'])
kernel_op.parameters = kernel_params
kernel_op.set_anatomical_image(images['T1'])

# %%
# Run kernel-guided Richardson–Lucy deconvolution using global iteration value.
deconv_kernel_alpha, obj_values_kernel, rmse_values_kernel = richardson_lucy(
    images['OSEM'],
    blurring_operator,
    iterations=RL_ITERATIONS_KERNEL,
    ground_truth=images['PET'],
    kernel_operator=kernel_op
)

# Map the estimated coefficients back using the kernel operator.
deconv_kernel = kernel_op.direct(deconv_kernel_alpha)

# %%
# Display the deconvolved image and kernel coefficients.
fig1 = show2D([deconv_kernel, deconv_kernel_alpha],
              fix_range=[(0, 320), (0, 320)])
fig1.save(os.path.join(data_dir, 'deconv_kernel_100_iter.png'))

# %%
# Display comparison: deconvolved image, OSEM, ground truth, and difference image.
difference_image = deconv_kernel - images['PET']
fig2 = show2D([deconv_kernel, images['OSEM'], images['PET'], difference_image],
              title=['Deconvolved', 'OSEM', 'Ground Truth', 'Difference (Deconv - GT)'],
              origin='upper', num_cols=4,
              fix_range=[(0, 320), (0, 320), (0, 320), (-100, 100)])
fig2.save(os.path.join(data_dir, 'deconv_kernel_100_iter_difference.png'))

# %%
# Plot objective function values for kernel-guided deconvolution.
plt.figure()
plt.plot(obj_values_kernel)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.savefig(os.path.join(data_dir, 'deconv_kernel_100_iter_objective.png'))

# %%
# Plot RMSE for kernel-guided deconvolution.
plt.figure()
plt.plot(rmse_values_kernel)
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.savefig(os.path.join(data_dir, 'deconv_kernel_100_iter_rmse.png'))

# %%
# Run standard Richardson–Lucy deconvolution (without kernel guidance).
deconv_rl, obj_values_rl, rmse_values_rl = richardson_lucy(
    images['OSEM'], blurring_operator,
    iterations=RL_ITERATIONS_STANDARD,
    ground_truth=images['PET']
)

# %%
# Display comparison for standard RL deconvolution.
difference_image_rl = deconv_rl - images['PET']
fig3 = show2D([deconv_rl, images['OSEM'], images['PET'], difference_image_rl],
              title=['Deconvolved (RL)', 'OSEM', 'Ground Truth', 'Difference (RL - GT)'],
              origin='upper', num_cols=4,
              fix_range=[(0, 320), (0, 320), (0, 320), (-100, 100)])
fig3.save(os.path.join(data_dir, 'deconv_rl_20_iter_difference.png'))

# Plot objective function for standard RL.
plt.figure()
plt.plot(obj_values_rl)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.savefig(os.path.join(data_dir, 'deconv_rl_20_iter_objective.png'))

# Plot RMSE for standard RL.
plt.figure()
plt.plot(rmse_values_rl)
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.savefig(os.path.join(data_dir, 'deconv_rl_20_iter_rmse.png'))

# %%
# Set up directional TV deconvolution.
# Data fidelity term.
f = fn.KullbackLeibler(b=images['OSEM'], eta=images['OSEM'].get_uniform_copy(1e-6))
df = fn.OperatorCompositionFunction(f, blurring_operator)

# Prior term: using ALPHA from the global parameters.
grad = op.GradientOperator(images['OSEM'])
grad_ref = grad.direct(images['T1'])
d_op = op.CompositionOperator(DirectionalOperator(grad_ref), grad)
prior = ALPHA * fn.OperatorCompositionFunction(fn.SmoothMixedL21Norm(epsilon=1e-4), d_op)

maprl = MAPRL(initial_estimate=images['OSEM'], data_fidelity=df, prior=prior,
              step_size=STEP_SIZE, relaxation_eta=RELAXATION_ETA,
              update_objective_interval=UPDATE_OBJ_INTERVAL)
maprl.run(verbose=1, iterations=DTV_ITERATIONS)
deconv_dtv = maprl.solution

# %%
# Display deconvolved image and difference image (DTV).
difference_image_dtv = deconv_dtv - images['PET']
fig4 = show2D([deconv_dtv, images['OSEM'], images['PET'], difference_image_dtv],
              title=['Deconvolved (DTV)', 'OSEM', 'Ground Truth', 'Difference (DTV - GT)'],
              origin='upper', num_cols=4,
              fix_range=[(0, 320), (0, 320), (0, 320), (-100, 100)])
fig4.save(os.path.join(data_dir, 'deconv_dtv_difference.png'))

plt.figure(figsize=(15, 5))
plt.plot(maprl.objective)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.savefig(os.path.join(data_dir, 'deconv_dtv_objective.png'))

# %%
# Plot profiles through the central slices for comparison.
center_slice = deconv_kernel.shape[0] // 2
profile_axis = deconv_kernel.shape[2] // 2

plt.figure(figsize=(15, 5))
plt.plot(deconv_kernel.as_array()[center_slice, :, profile_axis], label='Deconvolved (Kernel)')
plt.plot(deconv_rl.as_array()[center_slice, :, profile_axis], label='Deconvolved (RL)')
plt.plot(deconv_dtv.as_array()[center_slice, :, profile_axis], label='Deconvolved (DTV)')
plt.plot(images['OSEM'].as_array()[center_slice, :, profile_axis], label='OSEM')
plt.plot(images['PET'].as_array()[center_slice, :, profile_axis], label='Ground Truth')
plt.legend()
plt.savefig(os.path.join(data_dir, 'profile_comparison_center.png'))

# %%
# Plot profile along a specific row (e.g., row 20) of the center slice.
plt.figure(figsize=(15, 5))
plt.plot(deconv_kernel.as_array()[center_slice, 20], label='Deconvolved (Kernel)')
plt.plot(deconv_rl.as_array()[center_slice, 20], label='Deconvolved (RL)')
plt.plot(deconv_dtv.as_array()[center_slice, 20], label='Deconvolved (DTV)')
plt.plot(images['OSEM'].as_array()[center_slice, 20], label='OSEM')
plt.plot(images['PET'].as_array()[center_slice, 20], label='Ground Truth')
plt.legend()
plt.savefig(os.path.join(data_dir, 'profile_comparison_row20.png'))
