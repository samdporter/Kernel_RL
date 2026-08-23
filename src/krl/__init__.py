"""
KRL: Kernelised Richardson-Lucy deconvolution for PET image reconstruction.

This package provides implementations of Richardson-Lucy deconvolution with
anatomical guidance for PET imaging, including:
- Standard Richardson-Lucy (RL)
- Kernelised RL (KRL) with anatomical-guided weights
- Hybrid KRL (HKRL) mixing emission and anatomical features
- L-BFGS-B with Directional Total Variation (DTV)
"""

__version__ = "0.2.0"
__author__ = "Kjell Erlandsson"

from krl.algorithms.lbfgsb import LBFGSBOptimizer, LBFGSBOptions
from krl.algorithms.maprl import MAPRL
from krl.algorithms.richardson_lucy import RichardsonLucy
from krl.callbacks import NRMSECallback, SaveIterationCallback
from krl.operators.blurring import GaussianBlurringOperator, create_gaussian_blur
from krl.operators.directional import DirectionalOperator
from krl.operators.kernel_operator import (
    DEFAULT_PARAMETERS,
    BaseKernelOperator,
    KernelOperator,
    get_kernel_operator,
)
from krl.utils import (
    get_array,
    load_image,
    load_nifti_as_imagedata,
    save_image,
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
    "DirectionalOperator",
    # Algorithms
    "MAPRL",
    "RichardsonLucy",
    "LBFGSBOptimizer",
    "LBFGSBOptions",
    # Callbacks
    "NRMSECallback",
    "SaveIterationCallback",
    # Utils
    "get_array",
    "load_image",
    "save_image",
    "load_nifti_as_imagedata",
]
