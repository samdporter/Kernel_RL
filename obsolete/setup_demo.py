# %% [markdown]
# # PET Deconvolution Data Setup Script
#
# In this script we simulate 3D FDG brain PET acquisitions. The steps are:
#
# 1. Create FDG (PET), uMap, T1, and T2 images as NumPy arrays and wrap them as SIRF `ImageData`.
# 2. Construct a `BlurringOperator` using CIL and an acquisition model using SIRF. Use CIL's `CompositionOperator`
#    to combine them into a blurred forward model.
# 3. Generate simulated PET acquisition data using the blurred forward model and add Poisson noise.
# 4. Reconstruct an OSEM image with SIRF's `OSMAPOSLReconstructor` (yielding a noisy, blurred image).
# 5. Simulate a point source measurement (using the same forward model) to later estimate the scanner’s PSF.
#
# The following code implements these steps.

# %%
import os
import re
import subprocess

import numpy as np
import matplotlib.pyplot as plt

import sirf.STIR as pet
from sirf.Utilities import examples_data_path
import brainweb

from cil.utilities.display import show2D
from cil.optimisation.operators import BlurringOperator, CompositionOperator

# Suppress STIR messages.
msg = pet.MessageRedirector()

# Create data directory based on current script path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# %%
# Global parameters to ensure reproducibility.
NOISE_SEED = 5
NOISE_LEVEL = 1
BW_SEED = 1337

# %%
# Load a BrainWeb file.
fname, url = sorted(brainweb.utils.LINKS.items())[0]
file_path = brainweb.get_file(fname, url, DATA_DIR)
data = brainweb.load_file(os.path.join(DATA_DIR, fname))

brainweb.seed(BW_SEED)
vol = brainweb.get_mmr_fromfile(
    os.path.join(DATA_DIR, fname),
    petNoise=1, t1Noise=0.75, t2Noise=0.75,
    petSigma=1, t1Sigma=1, t2Sigma=1
)

# Extract ground truth FDG PET, T1, T2, and attenuation (uMap) images.
arr_dict = {
    'PET': vol['PET'] * 2,
    'T1': vol['T1'],
    'T2': vol['T2'],
    'uMap': vol['uMap']
}

# Define desired output dimensions.
CROP_DIM = (8, 102, 102)


def crop_array(arr, target_dim):
    """
    Crop and downsample a 3D image to the target dimensions.

    Parameters:
        arr (np.ndarray): Input 3D image.
        target_dim (tuple): Desired dimensions (num_slices, height, width).

    Returns:
        np.ndarray: Cropped and downsampled 3D image.
    """
    # Crop spatial dimensions (axis 1 and 2).
    y_start = (arr.shape[1] - target_dim[1]) // 2
    x_start = (arr.shape[2] - target_dim[2]) // 2
    cropped = arr[:, y_start:y_start + target_dim[1], x_start:x_start + target_dim[2]]

    # Downsample slices to match target number along axis 0.
    num_slices = cropped.shape[0]
    stride = num_slices // target_dim[0]
    selected_indices = np.linspace(0, num_slices - stride, target_dim[0], dtype=int)
    downsampled = cropped[selected_indices, :, :]
    return downsampled


# Crop all images to CROP_DIM.
for key in arr_dict.keys():
    arr_dict[key] = crop_array(arr_dict[key], CROP_DIM)

# Display the ground truth images.
fig = show2D(
    [arr_dict['PET'], arr_dict['uMap'], arr_dict['T1'], arr_dict['T2']],
    title=['Ground Truth PET', 'uMap', 'T1 weighted MRI', 'T2 weighted MRI'],
    origin='upper', num_cols=2
)
fig.save(os.path.join(DATA_DIR, 'ground_truth.png'))

# %%
# Create SIRF ImageData objects and save them.
image_dict = {}
VOXEL_SIZE = (6.75, 2.2, 2.2)  # mm
for key, image in arr_dict.items():
    img_data = pet.ImageData()
    img_data.initialise(dim=CROP_DIM, vsize=VOXEL_SIZE)
    img_data.fill(image)
    image_dict[key] = img_data
    img_data.write(os.path.join(DATA_DIR, f'{key}_b{BW_SEED}.hv'))


def allocate_uniform(image, value):
    """Allocate an image filled with a constant value."""
    geometry = getattr(image, 'geometry', None)
    if geometry is not None and hasattr(geometry, 'allocate'):
        return geometry.allocate(value=value)
    clone = image.clone()
    clone.fill(value)
    return clone

# %%
# Define functions to create a 3D Gaussian kernel (PSF).
def fwhm_to_sigma(fwhm):
    """Convert full-width-at-half-maximum (FWHM) to Gaussian sigma."""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def psf(kernel_size, fwhm, voxel_size=(1, 1, 1)):
    """
    Create a 3D Gaussian point spread function (PSF).

    Parameters:
        kernel_size (int): Size of the kernel along each dimension.
        fwhm (tuple): FWHM values for each dimension.
        voxel_size (tuple): Voxel sizes in each dimension.

    Returns:
        np.ndarray: Normalized 3D PSF kernel.
    """
    sigma_vox = [fwhm_to_sigma(fwhm[i]) / voxel_size[i] for i in range(3)]
    axes = [np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size) for _ in range(3)]
    gaussians = [np.exp(-0.5 * np.square(ax) / np.square(sigma_vox[i]))
                 for i, ax in enumerate(axes)]
    kernel_3d = np.outer(gaussians[0], gaussians[1]).reshape(kernel_size, kernel_size, 1) * gaussians[2].reshape(1, 1, kernel_size)
    return kernel_3d / np.sum(kernel_3d)

# %%
def make_acquisition_model(template_sino, template_img, atten_img):
    """
    Build an acquisition model for PET using ray tracing and incorporate
    the attenuation sensitivity model.

    Parameters:
        template_sino (pet.AcquisitionData): Template sinogram.
        template_img (pet.ImageData): Template image (e.g., PET ground truth).
        atten_img (pet.ImageData): Attenuation image.

    Returns:
        pet.AcquisitionModelUsingRayTracingMatrix: Set up acquisition model.
    """
    acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_num_tangential_LORs(10)

    # Create and set the sensitivity model using the attenuation image.
    acq_asm = pet.AcquisitionModelUsingRayTracingMatrix()
    acq_asm.set_num_tangential_LORs(10)
    sens_model = pet.AcquisitionSensitivityModel(atten_img, acq_asm)
    acq_model.set_acquisition_sensitivity(sens_model)

    acq_model.set_up(template_sino, template_img)
    return acq_model


def add_poisson_noise(acq_data, noise_level=1, seed=10):
    """
    Add Poisson noise to acquisition data.

    Parameters:
        acq_data (pet.AcquisitionData): Original sinogram.
        noise_level (float): Scaling factor for noise.
        seed (int): Random seed.

    Returns:
        pet.AcquisitionData: Noisy acquisition data.
    """
    np.random.seed(seed)
    noisy_array = np.random.poisson(acq_data.as_array() / noise_level) * noise_level
    return acq_data.clone().fill(noisy_array)


# %%
# Create the acquisition model.
template_sino = pet.AcquisitionData(os.path.join(examples_data_path('PET'), 'brain', 'template_sinogram.hs'))
acq_model = make_acquisition_model(template_sino, image_dict['PET'], image_dict['uMap'])

# Create the noiseless sinogram.
sinogram = acq_model.direct(image_dict['PET'])

# Create a 3D Gaussian PSF with FWHM of 7 mm in each dimension.
PSF_KERNEL = psf(kernel_size=15, fwhm=(7, 7, 7), voxel_size=VOXEL_SIZE)
blur_operator = BlurringOperator(PSF_KERNEL, image_dict['PET'])

# Compose the forward model (acquisition model followed by blurring).
blurred_acq_model = CompositionOperator(acq_model, blur_operator)

# Generate the blurred sinogram and add Poisson noise.
blurred_sinogram = blurred_acq_model.direct(image_dict['PET'])
blurred_noisy_sinogram = add_poisson_noise(blurred_sinogram, noise_level=NOISE_LEVEL, seed=NOISE_SEED)

# Save the sinograms.
sinogram.write(os.path.join(DATA_DIR, f'bw_sinogram_b{BW_SEED}.hs'))
blurred_noisy_sinogram.write(os.path.join(DATA_DIR, f'bw_blurred_noisy_sinogram_b{BW_SEED}_n{NOISE_SEED}.hs'))

# %%
# Reconstruction using OSEM via SIRF's OSMAPOSLReconstructor.
objective_fn = pet.make_Poisson_loglikelihood(blurred_noisy_sinogram, acq_model=acq_model)
objective_fn.set_num_subsets(8)

reconstructor = pet.OSMAPOSLReconstructor()
reconstructor.set_num_subiterations(8)
reconstructor.set_objective_function(objective_fn)
reconstructor.set_up(image_dict['PET'])

# Create a processor to truncate the image to a cylinder (limiting FOV to avoid edge artifacts).
cylinder_processor = pet.TruncateToCylinderProcessor()
cylinder_processor.set_strictly_less_than_radius(True)

current_estimate = allocate_uniform(image_dict['PET'], 1)
cylinder_processor.apply(current_estimate)

obj_values = []
NUM_ITERATIONS = 12

for i in range(NUM_ITERATIONS):
    reconstructor.reconstruct(current_estimate)
    # The objective function is the negative log-likelihood.
    obj_values.append(-objective_fn(current_estimate))
    print(f"Reconstruction Iteration: {i}, Objective: {obj_values[-1]}", end='\r')
    cylinder_processor.apply(current_estimate)

# Save the reconstructed image.
current_estimate.write(os.path.join(DATA_DIR, f'OSEM_b{BW_SEED}_n{NOISE_SEED}.hv'))

# %%
# Simulate a point source measurement for PSF estimation.
point_source = allocate_uniform(image_dict['PET'], 0)
ps_array = point_source.as_array()
# Set the central voxel to a high value.
center = (ps_array.shape[0] // 2, ps_array.shape[1] // 2, ps_array.shape[2] // 2)
ps_array[center] = 1000
point_source.fill(ps_array)

# Simulate the point source sinogram.
ps_sinogram = blurred_acq_model.direct(point_source)
ps_sinogram_noisy = add_poisson_noise(ps_sinogram, noise_level=NOISE_LEVEL, seed=NOISE_SEED)

# Set up reconstruction for the point source.
ps_objective = pet.make_Poisson_loglikelihood(ps_sinogram_noisy, acq_model=acq_model)
ps_objective.set_num_subsets(8)
ps_reconstructor = pet.OSMAPOSLReconstructor()
ps_reconstructor.set_num_subiterations(8)
ps_reconstructor.set_objective_function(ps_objective)
ps_reconstructor.set_up(image_dict['PET'])

current_ps_estimate = allocate_uniform(image_dict['PET'], 1)
cylinder_processor.apply(current_ps_estimate)
ps_obj_values = []

for i in range(NUM_ITERATIONS):
    ps_reconstructor.reconstruct(current_ps_estimate)
    ps_obj_values.append(-ps_objective(current_ps_estimate))
    print(f"Point Source Iteration: {i}, Objective: {ps_obj_values[-1]}", end='\r')
    cylinder_processor.apply(current_ps_estimate)

# Save the reconstructed point source image.
current_ps_estimate.write(os.path.join(DATA_DIR, f'OSEM_psf_n{NOISE_SEED}.hv'))

# %%
# Cleanup temporary files generated during simulation.
for file in os.listdir('.'):
    if file.startswith('tmp_') and (file.endswith('.hs') or file.endswith('.s')):
        os.remove(file)
