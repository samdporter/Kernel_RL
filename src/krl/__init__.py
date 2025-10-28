"""
KRL: Kernelised Richardson-Lucy deconvolution for PET image reconstruction.

This package provides implementations of Richardson-Lucy deconvolution with
anatomical guidance for PET imaging, including:
- Standard Richardson-Lucy (RL)
- Kernelised RL (KRL) with anatomical-guided weights
- Hybrid KRL (HKRL) mixing emission and anatomical features
- L-BFGS-B with Directional Total Variation (DTV)
"""

__version__ = "0.1.0"
__author__ = "Kjell Erlandsson"

from krl.operators.kernel_operator import (
    get_kernel_operator,
    KernelOperator,
    BaseKernelOperator,
    DEFAULT_PARAMETERS,
)
from krl.operators.blurring import GaussianBlurringOperator, create_gaussian_blur
from krl.operators.gradient import Gradient
from krl.operators.directional import DirectionalOperator
from krl.algorithms.maprl import MAPRL
from krl.algorithms.lbfgsb import LBFGSBOptimizer, LBFGSBOptions
from krl.utils import (
    get_array, load_image, save_image, 
    load_nifti_as_imagedata, prepare_brainweb_pet_dataset,
)

__all__ = [
    "__version__",
    "__author__",
    # Operators
    "get_kernel_operator",
    "KernelOperator",
    "BaseKernelOperator",
    "DEFAULT_PARAMETERS",
    "GaussianBlurringOperator",
    "create_gaussian_blur",
    "Gradient",
    "DirectionalOperator",
    # Algorithms
    "MAPRL",
    "LBFGSBOptimizer",
    "LBFGSBOptions",
    # Utils
    "get_array",
    "load_image",
    "save_image",
    "load_nifti_as_imagedata",
    "prepare_brainweb_pet_dataset",
]
