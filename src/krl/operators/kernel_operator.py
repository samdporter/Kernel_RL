import platform

import numba
import numpy as np
from cil.optimisation.operators import LinearOperator

from krl.utils import get_array

DEFAULT_PARAMETERS = {
    "num_neighbours": 5,
    "sigma_anat": 0.1,
    "sigma_dist": 10000,
    "sigma_emission": 0.1,
    "normalize_features": True,
    "normalize_kernel": True,
    "use_mask": True,
    "mask_k": 20,
    "recalc_mask": False,
    "distance_weighting": False,
    "hybrid": False,
}


def get_kernel_operator(domain_geometry, backend="auto", **kwargs):
    """
    Returns the kernel operator with automatic backend selection.

    Parameters
    ----------
    domain_geometry : ImageGeometry
        Domain geometry for the operator
    backend : str, optional
        Backend to use: 'auto', 'torch', or 'numba'
        - 'auto': On macOS resolve to numba; elsewhere try torch (GPU) first,
          falling back to numba (CPU)
        - 'torch': Use PyTorch GPU backend (requires torch + CUDA)
        - 'numba': Use Numba CPU backend
    **kwargs
        Additional parameters passed to the operator

    Returns
    -------
    BaseKernelOperator
        Kernel operator instance (TorchKernelOperator or KernelOperator)

    Examples
    --------
    Auto-select backend (prefers GPU):
    >>> op = get_kernel_operator(geometry, backend='auto')

    Force GPU:
    >>> op = get_kernel_operator(geometry, backend='torch', dtype='float32')

    Force CPU:
    >>> op = get_kernel_operator(geometry, backend='numba')
    """
    if backend == "auto":
        # Importing torch is unsafe on macOS (it breaks CIL's OpenMP), so
        # resolve to the numba CPU backend without probing for it.
        if platform.system() == "Darwin":
            backend = "numba"
        else:
            try:
                import torch
                backend = "torch" if torch.cuda.is_available() else "numba"
            except ImportError:
                backend = "numba"

    if backend == 'torch':
        try:
            from .gpu_kernel_operator import TorchKernelOperator
            return TorchKernelOperator(domain_geometry, **kwargs)
        except ImportError as e:
            raise RuntimeError(
                f"PyTorch backend not available: {e}\n"
                "Install pytorch with: pip install torch torchvision"
            )
    elif backend == 'numba':
        return KernelOperator(domain_geometry, **kwargs)
    else:
        raise ValueError(
            f"Backend '{backend}' not supported. "
            "Use 'auto', 'torch', or 'numba'."
        )


class BaseKernelOperator(LinearOperator):
    def __init__(self, domain_geometry, **kwargs):
        super().__init__(domain_geometry=domain_geometry, range_geometry=domain_geometry)
        default_parameters = DEFAULT_PARAMETERS.copy()
        self.parameters = default_parameters | kwargs
        self.anatomical_image = None
        self.mask = None
        self.backend = "numba"
        self.freeze_emission_kernel = False
        self.frozen_emission_kernel = None
        self._normalisation_map = None
        self._anatomical_weights: np.ndarray | None = None

    def _invalidate_derived_state(self):
        self.mask = None
        self._anatomical_weights = None
        self._normalisation_map = None
        self.set_norm(None)

    def set_parameters(self, parameters):
        self.parameters.update(parameters)
        self._invalidate_derived_state()

    def set_anatomical_image(self, image):
        if self.parameters["normalize_features"]:
            arr = get_array(image)
            std = arr.std()
            norm = arr / std if std > 1e-12 else arr
            tmp = image.clone()
            tmp.fill(norm)
            self.anatomical_image = tmp
        else:
            self.anatomical_image = image
        self._invalidate_derived_state()

    @staticmethod
    def _validate_single_channel_3d_geometry(geometry):
        """Reject geometries whose three trailing array axes are not three
        spatial dimensions (e.g. a 2-D multi-channel geometry allocates
        ``(channels, y, x)``, which a naive ndim==3 check accepts)."""
        if geometry is None:
            return
        channels = getattr(geometry, "channels", 1)
        if channels != 1:
            raise ValueError(
                "KernelOperator only supports single-channel 3-D volumes; "
                f"the geometry has {channels} channels."
            )

    def _validate_anatomical_image(self):
        if self.anatomical_image is None:
            raise ValueError(
                "An anatomical image must be set before applying the kernel operator."
            )
        anat = np.asarray(get_array(self.anatomical_image))
        if anat.ndim != 3:
            raise ValueError(
                "KernelOperator only supports single-channel 3-D volumes."
            )
        domain_geometry = self.domain_geometry()
        self._validate_single_channel_3d_geometry(domain_geometry)
        self._validate_single_channel_3d_geometry(
            getattr(self.anatomical_image, "geometry", None)
        )
        domain_shape = getattr(domain_geometry, "shape", None)
        if domain_shape is not None and tuple(anat.shape) != tuple(domain_shape):
            raise ValueError(
                f"Anatomical image shape {anat.shape} does not match the domain "
                f"geometry {tuple(domain_shape)}."
            )
        return anat

    def _validate_neighbourhood(self, shape):
        n = self.parameters["num_neighbours"]
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise ValueError("num_neighbours must be a positive odd integer.")
        n = int(n)
        if n <= 0 or n % 2 == 0:
            raise ValueError("num_neighbours must be a positive odd integer.")
        half = n // 2
        if half > min(shape):
            raise ValueError(
                f"num_neighbours={n} has a reflection radius of {half}, which is "
                f"larger than the smallest array dimension {min(shape)}."
            )
        return n

    def _validate_sigmas(self):
        p = self.parameters
        self._validate_sigma("sigma_anat", p["sigma_anat"])
        if p["hybrid"]:
            self._validate_sigma("sigma_emission", p["sigma_emission"])
        if p["distance_weighting"]:
            self._validate_sigma("sigma_dist", p["sigma_dist"])

    @staticmethod
    def _validate_sigma(name, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number.")

    def _validate_inputs(self, x):
        anat = self._validate_anatomical_image()
        data = np.asarray(get_array(x))
        if data.ndim != 3:
            raise ValueError(
                "KernelOperator only supports single-channel 3-D volumes."
            )
        self._validate_single_channel_3d_geometry(getattr(x, "geometry", None))
        if anat.shape != data.shape:
            raise ValueError(
                f"Anatomical image shape {anat.shape} does not match input shape {data.shape}."
            )
        self._validate_neighbourhood(data.shape)
        self._validate_sigmas()

    def precompute_mask(self):
        anat = self._validate_anatomical_image()
        n = self._validate_neighbourhood(anat.shape)
        total = n**3
        mask_k = self.parameters["mask_k"]
        k = mask_k if mask_k is not None else total
        k = max(1, min(int(k), total))
        arr = np.ascontiguousarray(anat, dtype=np.float64)
        return _nb_precompute_mask(arr, n, k)

    def precompute_anatomical_weights(self):
        """
        Pre-compute the anatomical kernel weights that remain constant across iterations.
        This includes:
        - Anatomical intensity-based Gaussian weights
        - Distance-based weights (if enabled)
        - Combined weights stored per voxel and neighbor

        Returns:
            np.ndarray: Pre-computed weights of shape (s0, s1, s2, total_neighbors)
                       where total_neighbors is n³ (without mask) or k (with mask)
        """
        anat = self._validate_anatomical_image()
        n = self._validate_neighbourhood(anat.shape)
        self._validate_sigmas()

        sigma_anat = self.parameters["sigma_anat"]
        sigma_dist = self.parameters["sigma_dist"]
        distance_weighting = self.parameters["distance_weighting"]
        use_mask = self.parameters["use_mask"]

        arr = np.ascontiguousarray(anat, dtype=np.float64)

        if use_mask:
            if self.mask is None:
                self.mask = self.precompute_mask()
            return _nb_precompute_anatomical_weights_mask(
                arr, self.mask, n, sigma_anat, sigma_dist, distance_weighting
            )
        else:
            return _nb_precompute_anatomical_weights(
                arr, n, sigma_anat, sigma_dist, distance_weighting
            )

    def _update_hybrid_reference(self, emission_array: np.ndarray) -> np.ndarray:
        """Store (or reuse) the emission image that defines the hybrid weights."""
        if not self.parameters["hybrid"]:
            return emission_array

        # If frozen, always return the frozen reference (don't update)
        if self.freeze_emission_kernel:
            if self.frozen_emission_kernel is None:
                # First call after freezing: initialize and freeze
                self.frozen_emission_kernel = np.array(emission_array, copy=True)
            return self.frozen_emission_kernel

        # Not frozen: update the reference
        if emission_array is None:
            raise ValueError("Hybrid emission reference requires an emission array.")

        self.frozen_emission_kernel = np.array(emission_array, copy=True)
        return self.frozen_emission_kernel

    def _get_hybrid_reference(self) -> np.ndarray | None:
        """Return the emission image used for the hybrid weights."""
        if not self.parameters["hybrid"]:
            return None

        if self.frozen_emission_kernel is None:
            raise RuntimeError(
                "Hybrid emission reference has not been initialised. "
                "Call direct() (or explicitly freeze a reference) before adjoint()."
            )

        return self.frozen_emission_kernel

    def _ensure_normalisation_map(self, x):
        if self._normalisation_map is not None:
            return
        if self.parameters["hybrid"]:
            self._get_hybrid_reference()
            raise RuntimeError(
                "Normalization map has not been initialised. "
                "Call direct() before adjoint() when using a hybrid kernel."
            )
        # For a fixed kernel the normalisation map depends only on the
        # anatomical weights, so any input produces the same map.
        self.apply(x)

    def apply(self, x):
        self._validate_inputs(x)
        p = self.parameters
        return self.neighbourhood_kernel(
            x,
            self.anatomical_image,
            p["num_neighbours"],
            p["sigma_anat"],
            p["sigma_dist"],
            p["sigma_emission"],
            p["normalize_kernel"],
            p["use_mask"],
            p["recalc_mask"],
            p["distance_weighting"],
            p["hybrid"],
        )

    def direct(self, x, out=None):
        res = self.apply(x)
        if out is None:
            return res
        out.fill(get_array(res))
        return out

    def adjoint(self, x, out=None):
        # default: same as forward (kernel remains self-adjoint without mask/hybrid)
        res = self.direct(x)
        if out is None:
            return res
        out.fill(get_array(res))
        return out


class KernelOperator(BaseKernelOperator):
    def __init__(self, domain_geometry, **kwargs):
        super().__init__(domain_geometry, **kwargs)
        self.backend = "numba"

    def neighbourhood_kernel(
        self,
        x,
        image,
        num_neighbours,
        sigma_anat,
        sigma_dist,
        sigma_emission,
        normalize_kernel,
        use_mask,
        recalc_mask,
        distance_weighting,
        hybrid,
    ):
        arr = get_array(image)
        x_arr = get_array(x)
        ref_arr = self._update_hybrid_reference(x_arr) if hybrid else x_arr
        norm_arr = (
            np.zeros_like(arr, dtype=np.float64)
            if normalize_kernel
            else np.zeros((1, 1, 1), dtype=np.float64)
        )
        n = num_neighbours

        if use_mask and recalc_mask:
            self.mask = self.precompute_mask()
            self._anatomical_weights = None
        # Pre-compute or retrieve cached anatomical weights
        if self._anatomical_weights is None:
            self._anatomical_weights = self.precompute_anatomical_weights()

        # Use sparse or dense pre-computed kernel based on masking
        if use_mask:
            res = _nb_kernel_precomputed_sparse(
                x_arr,
                ref_arr,
                self._anatomical_weights,
                self.mask,
                norm_arr,
                n,
                sigma_emission,
                normalize_kernel,
                hybrid,
            )
        else:
            res = _nb_kernel_precomputed(
                x_arr,
                ref_arr,
                self._anatomical_weights,
                norm_arr,
                n,
                sigma_emission,
                normalize_kernel,
                hybrid,
            )

        # Clone the input (not the anatomical image) so the output dtype
        # follows the data being transformed.
        out = x.clone()
        out.fill(res)
        self._normalisation_map = norm_arr if normalize_kernel else None
        return out

    def adjoint(self, x, out=None):
        self._validate_inputs(x)
        x_arr = get_array(x)
        p = self.parameters
        if p["normalize_kernel"]:
            self._ensure_normalisation_map(x)
            norm_arr = self._normalisation_map
        else:
            norm_arr = np.zeros((1, 1, 1), dtype=np.float64)

        n = p["num_neighbours"]

        # Pre-compute or retrieve cached anatomical weights
        if self._anatomical_weights is None:
            self._anatomical_weights = self.precompute_anatomical_weights()

        ref_arr = self._get_hybrid_reference() if p["hybrid"] else x_arr

        # Use sparse or dense pre-computed adjoint kernel based on masking
        if p["use_mask"]:
            res = _nb_adjoint_precomputed_sparse(
                x_arr,
                ref_arr,
                self._anatomical_weights,
                self.mask,
                norm_arr,
                n,
                p["sigma_emission"],
                p["hybrid"],
            )
        else:
            res = _nb_adjoint_precomputed(
                x_arr,
                ref_arr,
                self._anatomical_weights,
                norm_arr,
                n,
                p["sigma_emission"],
                p["hybrid"],
            )

        img = x.clone()
        img.fill(res)
        if out is None:
            return img
        out.fill(res)
        return out


@numba.njit(cache=True, parallel=True, fastmath=True)
def _nb_precompute_mask(anat_arr, n, k_keep):
    """
    Precompute sparse mask as integer indices of the k_keep most similar neighbors.
    Returns shape (s0, s1, s2, k_keep) with integer indices in [0, n³).
    """
    s0, s1, s2 = anat_arr.shape
    total = n ** 3
    half = n // 2
    # Return sparse indices instead of boolean mask
    mask_indices = np.zeros((s0, s1, s2, k_keep), dtype=np.int32)

    for i in numba.prange(s0):
        for j in range(s1):
            for k in range(s2):
                diffs = np.empty(total, dtype=np.float64)
                center = anat_arr[i, j, k]
                idx = 0

                for di in range(-half, half + 1):
                    ii = i + di
                    if ii < 0:
                        ii = -ii - 1
                    elif ii >= s0:
                        ii = 2 * s0 - ii - 1
                    for dj in range(-half, half + 1):
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        for dk in range(-half, half + 1):
                            kk = k + dk
                            if kk < 0:
                                kk = -kk - 1
                            elif kk >= s2:
                                kk = 2 * s2 - kk - 1

                            diffs[idx] = abs(anat_arr[ii, jj, kk] - center)
                            idx += 1

                # Find indices of k_keep smallest differences
                # Using argsort (partial sort would be better but numba doesn't support it well)
                sorted_indices = np.argsort(diffs)
                mask_indices[i, j, k, :] = sorted_indices[:k_keep]

    return mask_indices


@numba.njit(cache=True, parallel=True, fastmath=True)
def _nb_precompute_anatomical_weights(anat_arr, n, sigma_anat, sigma_dist, distance_weighting):
    """
    Pre-compute anatomical weights for all voxels and all n³ neighbors.
    Returns shape: (s0, s1, s2, n³)
    """
    s0, s1, s2 = anat_arr.shape
    half = n // 2
    total = n ** 3
    sig2_an = 2.0 * sigma_anat * sigma_anat
    dist2_an = 2.0 * sigma_dist * sigma_dist

    # Pre-compute distance weights
    wd_an = np.ones((n, n, n), dtype=np.float64)
    if distance_weighting:
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                for dk in range(-half, half + 1):
                    d2 = di * di + dj * dj + dk * dk
                    wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

    weights = np.zeros((s0, s1, s2, total), dtype=np.float64)

    for i in numba.prange(s0):
        for j in range(s1):
            for k in range(s2):
                ca = anat_arr[i, j, k]
                idx = 0

                for di in range(-half, half + 1):
                    ii = i + di
                    if ii < 0:
                        ii = -ii - 1
                    elif ii >= s0:
                        ii = 2 * s0 - ii - 1
                    for dj in range(-half, half + 1):
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        for dk in range(-half, half + 1):
                            kk = k + dk
                            if kk < 0:
                                kk = -kk - 1
                            elif kk >= s2:
                                kk = 2 * s2 - kk - 1

                            diff_an = anat_arr[ii, jj, kk] - ca
                            wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                            weights[i, j, k, idx] = wi_an * wd_an[di + half, dj + half, dk + half]
                            idx += 1

    return weights


@numba.njit(cache=True, parallel=True, fastmath=True)
def _nb_precompute_anatomical_weights_mask(anat_arr, mask_indices, n, sigma_anat, sigma_dist, distance_weighting):
    """
    Pre-compute anatomical weights for all voxels using sparse mask indices.
    mask_indices: shape (s0, s1, s2, k) with integer indices in [0, n³)
    Returns: shape (s0, s1, s2, k) with weights for valid neighbors only.
    """
    s0, s1, s2 = anat_arr.shape
    half = n // 2
    k = mask_indices.shape[3]  # Number of kept neighbors
    sig2_an = 2.0 * sigma_anat * sigma_anat
    dist2_an = 2.0 * sigma_dist * sigma_dist

    # Pre-compute distance weights for all possible offsets
    wd_an = np.ones((n, n, n), dtype=np.float64)
    if distance_weighting:
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                for dk in range(-half, half + 1):
                    d2 = di * di + dj * dj + dk * dk
                    wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

    # Sparse weights - only store k neighbors per voxel
    weights = np.zeros((s0, s1, s2, k), dtype=np.float64)

    for i in numba.prange(s0):
        for j in range(s1):
            for k_vox in range(s2):
                ca = anat_arr[i, j, k_vox]

                # Iterate only over the k valid neighbors
                for k_idx in range(k):
                    flat_idx = mask_indices[i, j, k_vox, k_idx]

                    # Convert flat index back to (di, dj, dk) offset
                    dk = (flat_idx % n) - half
                    dj = ((flat_idx // n) % n) - half
                    di = (flat_idx // (n * n)) - half

                    # Apply boundary conditions
                    ii = i + di
                    if ii < 0:
                        ii = -ii - 1
                    elif ii >= s0:
                        ii = 2 * s0 - ii - 1
                    jj = j + dj
                    if jj < 0:
                        jj = -jj - 1
                    elif jj >= s1:
                        jj = 2 * s1 - jj - 1
                    kk = k_vox + dk
                    if kk < 0:
                        kk = -kk - 1
                    elif kk >= s2:
                        kk = 2 * s2 - kk - 1

                    # Compute anatomical weight
                    diff_an = anat_arr[ii, jj, kk] - ca
                    wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                    weights[i, j, k_vox, k_idx] = wi_an * wd_an[di + half, dj + half, dk + half]

    return weights


@numba.njit(cache=True, parallel=True, fastmath=True)
def _nb_kernel_precomputed(
    x_arr,
    ref_arr,
    anat_weights,
    norm_arr,
    n,
    sigma_emission,
    normalize,
    hybrid,
):
    """
    Forward kernel using pre-computed anatomical weights (dense version, no mask).
    Only calculates emission weights (if hybrid) and applies to data.
    """
    s0, s1, s2 = x_arr.shape
    half = n // 2
    sig2_em = 2.0 * sigma_emission * sigma_emission

    out = np.empty_like(x_arr, dtype=np.float64)

    for i in numba.prange(s0):
        for j in range(s1):
            for k in range(s2):
                c_ref = ref_arr[i, j, k]
                sumv = 0.0
                wsum = 0.0
                idx = 0

                for di in range(-half, half + 1):
                    ii = i + di
                    if ii < 0:
                        ii = -ii - 1
                    elif ii >= s0:
                        ii = 2 * s0 - ii - 1
                    for dj in range(-half, half + 1):
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        for dk in range(-half, half + 1):
                            kk = k + dk
                            if kk < 0:
                                kk = -kk - 1
                            elif kk >= s2:
                                kk = 2 * s2 - kk - 1

                            # Get pre-computed anatomical weight
                            w = anat_weights[i, j, k, idx]

                            # Apply emission weight if hybrid
                            if hybrid and w > 0.0:
                                diff_em = ref_arr[ii, jj, kk] - c_ref
                                wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                w *= wi_em

                            sumv += x_arr[ii, jj, kk] * w
                            wsum += w
                            idx += 1

                if normalize:
                    if wsum > 1e-12:
                        sumv /= wsum
                        norm_arr[i, j, k] = wsum
                    else:
                        norm_arr[i, j, k] = 1.0
                out[i, j, k] = sumv

    return out


@numba.njit(cache=True, parallel=True, fastmath=True)
def _nb_kernel_precomputed_sparse(
    x_arr,
    ref_arr,
    anat_weights,
    mask_indices,
    norm_arr,
    n,
    sigma_emission,
    normalize,
    hybrid,
):
    """
    Forward kernel using pre-computed anatomical weights (sparse version with mask).
    Only iterates over k masked neighbors per voxel.
    anat_weights: shape (s0, s1, s2, k)
    mask_indices: shape (s0, s1, s2, k) with integer indices in [0, n³)
    """
    s0, s1, s2 = x_arr.shape
    half = n // 2
    k = anat_weights.shape[3]
    sig2_em = 2.0 * sigma_emission * sigma_emission

    out = np.empty_like(x_arr, dtype=np.float64)

    for i in numba.prange(s0):
        for j in range(s1):
            for k_vox in range(s2):
                c_ref = ref_arr[i, j, k_vox]
                sumv = 0.0
                wsum = 0.0

                # Iterate only over k valid neighbors
                for k_idx in range(k):
                    flat_idx = mask_indices[i, j, k_vox, k_idx]

                    # Convert flat index to (di, dj, dk) offset
                    dk = (flat_idx % n) - half
                    dj = ((flat_idx // n) % n) - half
                    di = (flat_idx // (n * n)) - half

                    # Apply boundary conditions
                    ii = i + di
                    if ii < 0:
                        ii = -ii - 1
                    elif ii >= s0:
                        ii = 2 * s0 - ii - 1
                    jj = j + dj
                    if jj < 0:
                        jj = -jj - 1
                    elif jj >= s1:
                        jj = 2 * s1 - jj - 1
                    kk = k_vox + dk
                    if kk < 0:
                        kk = -kk - 1
                    elif kk >= s2:
                        kk = 2 * s2 - kk - 1

                    # Get pre-computed anatomical weight
                    w = anat_weights[i, j, k_vox, k_idx]

                    # Apply emission weight if hybrid
                    if hybrid and w > 0.0:
                        diff_em = ref_arr[ii, jj, kk] - c_ref
                        wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                        w *= wi_em

                    sumv += x_arr[ii, jj, kk] * w
                    wsum += w

                if normalize:
                    if wsum > 1e-12:
                        sumv /= wsum
                        norm_arr[i, j, k_vox] = wsum
                    else:
                        norm_arr[i, j, k_vox] = 1.0
                out[i, j, k_vox] = sumv

    return out


@numba.njit(cache=True)
def _nb_adjoint_precomputed(
    x_arr,
    ref_arr,
    anat_weights,
    norm_arr,
    n,
    sigma_emission,
    hybrid,
):
    """
    Adjoint kernel using pre-computed anatomical weights (dense version, no mask).
    Only calculates emission weights (if hybrid) and applies to data.
    """
    s0, s1, s2 = x_arr.shape
    half = n // 2
    sig2_em = 2.0 * sigma_emission * sigma_emission

    out = np.zeros_like(x_arr, dtype=np.float64)

    for i in numba.prange(s0):
        for j in range(s1):
            for k in range(s2):
                val = x_arr[i, j, k]
                if norm_arr.shape[0] > 1:
                    norm = norm_arr[i, j, k]
                    val = val / norm if norm > 1e-12 else 0.0
                c_ref = ref_arr[i, j, k]
                idx = 0

                for di in range(-half, half + 1):
                    ii = i + di
                    if ii < 0:
                        ii = -ii - 1
                    elif ii >= s0:
                        ii = 2 * s0 - ii - 1
                    for dj in range(-half, half + 1):
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        for dk in range(-half, half + 1):
                            kk = k + dk
                            if kk < 0:
                                kk = -kk - 1
                            elif kk >= s2:
                                kk = 2 * s2 - kk - 1

                            # Get pre-computed anatomical weight
                            w = anat_weights[i, j, k, idx]

                            # Apply emission weight if hybrid
                            if hybrid and w > 0.0:
                                diff_em = ref_arr[ii, jj, kk] - c_ref
                                wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                w *= wi_em

                            out[ii, jj, kk] += val * w
                            idx += 1

    return out


@numba.njit(cache=True)
def _nb_adjoint_precomputed_sparse(
    x_arr,
    ref_arr,
    anat_weights,
    mask_indices,
    norm_arr,
    n,
    sigma_emission,
    hybrid,
):
    """
    Adjoint kernel using pre-computed anatomical weights (sparse version with mask).
    Only iterates over k masked neighbors per voxel.
    anat_weights: shape (s0, s1, s2, k)
    mask_indices: shape (s0, s1, s2, k) with integer indices in [0, n³)
    """
    s0, s1, s2 = x_arr.shape
    half = n // 2
    k = anat_weights.shape[3]
    sig2_em = 2.0 * sigma_emission * sigma_emission

    out = np.zeros_like(x_arr, dtype=np.float64)

    for i in numba.prange(s0):
        for j in range(s1):
            for k_vox in range(s2):
                val = x_arr[i, j, k_vox]
                if norm_arr.shape[0] > 1:
                    norm = norm_arr[i, j, k_vox]
                    val = val / norm if norm > 1e-12 else 0.0
                c_ref = ref_arr[i, j, k_vox]

                # Iterate only over k valid neighbors
                for k_idx in range(k):
                    flat_idx = mask_indices[i, j, k_vox, k_idx]

                    # Convert flat index to (di, dj, dk) offset
                    dk = (flat_idx % n) - half
                    dj = ((flat_idx // n) % n) - half
                    di = (flat_idx // (n * n)) - half

                    # Apply boundary conditions
                    ii = i + di
                    if ii < 0:
                        ii = -ii - 1
                    elif ii >= s0:
                        ii = 2 * s0 - ii - 1
                    jj = j + dj
                    if jj < 0:
                        jj = -jj - 1
                    elif jj >= s1:
                        jj = 2 * s1 - jj - 1
                    kk = k_vox + dk
                    if kk < 0:
                        kk = -kk - 1
                    elif kk >= s2:
                        kk = 2 * s2 - kk - 1

                    # Get pre-computed anatomical weight
                    w = anat_weights[i, j, k_vox, k_idx]

                    # Apply emission weight if hybrid
                    if hybrid and w > 0.0:
                        diff_em = ref_arr[ii, jj, kk] - c_ref
                        wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                        w *= wi_em

                    out[ii, jj, kk] += val * w

    return out
