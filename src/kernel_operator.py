"""Compatibility wrapper exposing the kernel operator module."""

from krl.operators.kernel_operator import (
    BaseKernelOperator,
    DEFAULT_PARAMETERS,
    KernelOperator,
    NUMBA_AVAIL,
    get_kernel_operator,
)

__all__ = [
    "BaseKernelOperator",
    "DEFAULT_PARAMETERS",
    "KernelOperator",
    "NUMBA_AVAIL",
    "get_kernel_operator",
]
