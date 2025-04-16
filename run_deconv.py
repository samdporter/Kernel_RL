# %% [markdown]
# # PET Deconvolution Exercise
#
# This exercise involves the following steps:
#
# 1. Estimate the point spread function (PSF) by using STIR’s
#    `find_fwhm_in_image` utility on a point source measurement.
#
# 2. Deconvolve the OSEM reconstruction using the Richardson–Lucy (RL)
#    algorithm with the estimated PSF.
#
# 3. Address noise amplification from RL by applying a CIL algorithm
#    (e.g., PDHG) for total variation (TV) regularised deconvolution.
#
# 4. Improve deconvolution using image guidance by incorporating a T1
#    MRI image to implement directional TV.
#
# 5. Implement a preconditioned gradient descent algorithm using a
#    smoothed directional TV prior. Analyze its convergence properties
#    and the effect of smoothing on the solution.
#
# 6. Explore joint reconstruction by incorporating additional modalities,
#    such as Amyloid PET images, in the deconvolution process.

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

# Redirect messages from STIR (if needed)
msg = pet.MessageRedirector()

# %%
from src.my_kem import get_kernel_operator
from src.directional_operator import DirectionalOperator
from src.map_rl import MAPRL

# %% Global variables
# Determine script and data directories and set the seeds.
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

NOISE_SEED = 5
BW_SEED = 1337

# %%
def find_fwhm_in_image(file_path):
    """
    Run STIR's utility 'find_fwhm_in_image' and parse the output
    to extract full-width-at-half-maximum (FWHM) values for each axis.

    Parameters:
        file_path (str): Path to the point source image file.

    Returns:
        dict or None: A dictionary mapping each axis to its FWHM value or
        None if the command failed.
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
    Convert a full-width-at-half-maximum (FWHM) value to the
    corresponding standard deviation (sigma) of a Gaussian.
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def psf(kernel_size, fwhm, voxel_size=(1, 1, 1)):
    """
    Construct a 3D point spread function (PSF) kernel using a Gaussian model.

    Parameters:
        kernel_size (int): The size of the kernel along each dimension.
        fwhm (list or tuple): FWHM values for each axis.
        voxel_size (tuple): The voxel size in each dimension.

    Returns:
        np.ndarray: A normalized 3D PSF kernel.
    """
    sigma_voxels = [
        fwhm_to_sigma(fwhm[i]) / voxel_size[i] for i in range(3)
    ]
    # Create coordinate axes for each dimension
    axes = [
        np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        for _ in range(3)
    ]
    # Compute the Gaussian along each axis
    gauss = [
        np.exp(-0.5 * np.square(ax) / np.square(sigma_voxels[i]))
        for i, ax in enumerate(axes)
    ]
    # Combine the 1D Gaussians to form a 3D kernel.
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
images['OSEM'] = pet.ImageData(
    os.path.join(data_dir, f'OSEM_b{BW_SEED}_n{NOISE_SEED}.hv'))
images['OSEM_psf'] = pet.ImageData(
    os.path.join(data_dir, f'OSEM_psf_n{NOISE_SEED}.hv'))

# %%
# Estimate FWHM from the point source measurement.
fwhm_values = list(find_fwhm_in_image(
    os.path.join(data_dir, f'OSEM_psf_n{NOISE_SEED}.hv')).values())
print(f'FWHM: {fwhm_values}')

# Generate the PSF kernel and set up the blurring operator.
psf_kernel = psf(5, fwhm=fwhm_values, voxel_size=images['OSEM'].voxel_sizes())
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
        ground_truth (pet.ImageData, optional): Ground truth image for RMSE computation.
        kernel_operator (Operator, optional): Additional operator (e.g., for guided deconvolution).

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
        # Update step of the RL algorithm.
        current_estimate *= blur_op.adjoint(observed / (estimated_blurred + epsilon))

        if ground_truth is not None:
            error = kernel_operator.direct(current_estimate) - ground_truth
            rmse = np.sqrt((error.power(2)).sum())
            rmse_values.append(rmse)
            print(f"Iteration: {i}, RMSE: {rmse}", end="\r")

        estimated_blurred = blur_op.direct(current_estimate)

        # Update the objective function value.
        obj_value = (estimated_blurred - (observed * (estimated_blurred + epsilon).log())).sum()
        objective_values.append(obj_value)
        print(f"Iteration: {i}, Objective: {obj_value}", end="\r")

    if ground_truth is not None:
        return current_estimate, objective_values, rmse_values

    return current_estimate, objective_values

# Define parameters for the kernel operator used in guided deconvolution.
kernel_params = {
    'num_neighbours': 5,
    'sigma_anat': 1.0,
    'sigma_dist': 1.0,
    'kernel_type': 'neighbourhood',
    # Toggle these normalizations:
    'normalize_features': True,   # Feature normalization
    'normalize_kernel': True,     # Row-sum normalization
}

kernel_op = get_kernel_operator(images['OSEM'])
kernel_op.parameters = kernel_params
kernel_op.set_anatomical_image(images['T1'])

# %%
# Run the Richardson–Lucy deconvolution with kernel guidance for 5 iterations.
deconv_kernel_alpha_100, obj_values_kernel_100, rmse_values_kernel_100 = richardson_lucy(
    images['OSEM'],
    blurring_operator,
    iterations=100,
    ground_truth=images['PET'],
    kernel_operator=kernel_op
)

# Map the estimated coefficients back using the kernel operator.
deconv_kernel_100 = kernel_op.direct(deconv_kernel_alpha_100)

# %%
# Display the deconvolved image and the kernel coefficients.
fig1 = show2D([deconv_kernel_100, deconv_kernel_alpha_100],
              fix_range=[(0, 320), (0, 320)])
fig1.save(os.path.join(data_dir, 'deconv_kernel_100_iter.png'))

# %%
# Display comparison of deconvolved image, OSEM, ground truth, and the difference image.
difference_image = deconv_kernel_100 - images['PET']
fig2 = show2D([deconv_kernel_100, images['OSEM'], images['PET'], difference_image],
              title=['Deconvolved', 'OSEM', 'Ground Truth', 'Difference (Deconv - GT)'],
              origin='upper', num_cols=4, fix_range=[(0, 320), (0, 320), (0, 320), (-100,100)])
fig2.save(os.path.join(data_dir, 'deconv_kernel_100_iter_difference.png'))

# %%
# Plot the objective function values for the kernel-guided deconvolution.
plt.figure()
plt.plot(obj_values_kernel_100)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.savefig(os.path.join(data_dir, 'deconv_kernel_100_iter_objective.png'))

# %%
# Plot the RMSE for the kernel-guided deconvolution.
plt.figure()
plt.plot(rmse_values_kernel_100)
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.savefig(os.path.join(data_dir, 'deconv_kernel_100_iter_rmse.png'))

# %%
# Run standard Richardson–Lucy deconvolution (without kernel guidance) for 5 iterations.
deconv_rl_20, obj_values_rl_20, rmse_values_rl_20 = richardson_lucy(
    images['OSEM'], blurring_operator, iterations=20, ground_truth=images['PET']
)

# %% display the comparison of deconvolved image, OSEM, ground truth, and the difference image.
difference_image_rl = deconv_rl_20 - images['PET']
fig3 = show2D([deconv_rl_20, images['OSEM'], images['PET'], difference_image_rl],
                title=['Deconvolved (RL)', 'OSEM', 'Ground Truth', 'Difference (RL - GT)'],
                origin='upper', num_cols=4, fix_range=[(0, 320), (0, 320), (0, 320), (-100,100)])
fig3.save(os.path.join(data_dir, 'deconv_rl_20_iter_difference.png'))

# Plot the objective function for the standard RL deconvolution.
plt.figure()
plt.plot(obj_values_rl_20)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.savefig(os.path.join(data_dir, 'deconv_rl_20_iter_objective.png'))

# Plot the RMSE for the standard RL deconvolution.
plt.figure()
plt.plot(rmse_values_rl_20)
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.savefig(os.path.join(data_dir, 'deconv_rl_20_iter_rmse.png'))

# %% set up directional TV

# set up the data fidelity term
f = fn.KullbackLeibler(b=images['OSEM'], eta=images['OSEM'].get_uniform_copy(1e-6))
df = fn.OperatorCompositionFunction(f, blurring_operator)

# set up the prior term
alpha = 0.02
grad = op.GradientOperator(images['OSEM'])
grad_ref = grad.direct(images['T1'])
d_op = op.CompositionOperator(
    DirectionalOperator(grad_ref),
    grad
)
prior = alpha * fn.OperatorCompositionFunction(fn.SmoothMixedL21Norm(epsilon=1e-4), d_op)

maprl = MAPRL(initial_estimate=images['OSEM'], data_fidelity=df, prior=prior, 
              step_size=0.1, relaxation_eta=0.01, update_objective_interval=1)
maprl.run(verbose=1, iterations=100)

deconv_dtv = maprl.solution

# %%
# Display the deconvolved image and the difference image.
difference_image_dtv = deconv_dtv - images['PET']
fig4 = show2D([deconv_dtv, images['OSEM'], images['PET'], difference_image_dtv],
                title=['Deconvolved (DTV)', 'OSEM', 'Ground Truth', 'Difference (DTV - GT)'],
                origin='upper', num_cols=4, fix_range=[(0, 320), (0, 320), (0, 320), (-100,100)])
fig4.save(os.path.join(data_dir, 'deconv_dtv_difference.png'))

plt.figure(figsize=(15, 5))
plt.plot(maprl.objective)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.savefig(os.path.join(data_dir, 'deconv_dtv_objective.png'))

# %%
# Plot profiles through the central slices for comparison.
center_slice = deconv_kernel_100.shape[0] // 2
profile_axis = deconv_kernel_100.shape[2] // 2

plt.figure(figsize=(15, 5))
plt.plot(deconv_kernel_100.as_array()[center_slice, :, profile_axis], label='Deconvolved (Kernel)')
plt.plot(deconv_rl_20.as_array()[center_slice, :, profile_axis], label='Deconvolved (RL)')
plt.plot(deconv_dtv.as_array()[center_slice, :, profile_axis], label='Deconvolved (DTV)')
plt.plot(images['OSEM'].as_array()[center_slice, :, profile_axis], label='OSEM')
plt.plot(images['PET'].as_array()[center_slice, :, profile_axis], label='Ground Truth')
plt.legend()
plt.savefig(os.path.join(data_dir, 'profile_comparison_center.png'))

# %%
# Plot profile through a specific row (e.g., row 20) of the center slice.
plt.figure(figsize=(15, 5))
plt.plot(deconv_kernel_100.as_array()[center_slice, 20], label='Deconvolved (Kernel)')
plt.plot(deconv_rl_20.as_array()[center_slice, 20], label='Deconvolved (RL)')
plt.plot(deconv_dtv.as_array()[center_slice, 20], label='Deconvolved (DTV)')
plt.plot(images['OSEM'].as_array()[center_slice, 20], label='OSEM')
plt.plot(images['PET'].as_array()[center_slice, 20], label='Ground Truth')
plt.legend()
plt.savefig(os.path.join(data_dir, 'profile_comparison_row20.png'))
