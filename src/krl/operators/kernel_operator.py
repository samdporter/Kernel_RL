import numpy as np

try:
    from cil.optimisation.operators import LinearOperator
except (ImportError, OSError):  # pragma: no cover - optional dependency
    class LinearOperator:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            self.__dict__.update(kwargs)

from krl.utils import get_array  # noqa: E402

# Try importing numba
try:
    import numba

    NUMBA_AVAIL = True
except (ImportError, OSError):  # pragma: no cover - optional dependency
    NUMBA_AVAIL = False
    numba = None  # type: ignore


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
    Returns the kernel operator accelerated with numba.
    backend: 'auto'|'numba'
    """
    if backend == "auto":
        backend = "numba"
    if backend != "numba":
        raise ValueError(
            f"Backend '{backend}' is no longer supported. "
            "Only the numba backend is available."
        )
    if not NUMBA_AVAIL:
        raise RuntimeError(
            "Numba backend not available. Please install numba to use the kernel operator."
        )
    return KernelOperator(domain_geometry, **kwargs)


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
        self._full_mask_cache: np.ndarray | None = None

    def set_parameters(self, parameters):
        self.parameters.update(parameters)
        self.mask = None

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
        self.mask = None

    def precompute_mask(self):
        if not NUMBA_AVAIL:
            raise RuntimeError("Numba backend required for mask precomputation.")
        if self.anatomical_image is None:
            raise RuntimeError(
                "An anatomical image must be set before precomputing a mask."
            )
        n = int(self.parameters["num_neighbours"])
        total = n**3
        mask_k = self.parameters["mask_k"]
        k = mask_k if mask_k is not None else total
        k = max(1, min(int(k), total))
        arr = np.ascontiguousarray(get_array(self.anatomical_image), dtype=np.float64)
        return _nb_precompute_mask(arr, n, k)

    def _get_full_mask(self, shape: tuple[int, int, int], n: int) -> np.ndarray:
        """Return a cached all-True mask for the current geometry."""
        expected_shape = (shape[0], shape[1], shape[2], n**3)
        mask = self._full_mask_cache
        if mask is None or mask.shape != expected_shape:
            mask = np.ones(expected_shape, dtype=np.bool_)
            self._full_mask_cache = mask
        return mask

    def _update_hybrid_reference(self, emission_array: np.ndarray) -> np.ndarray:
        """Store (or reuse) the emission image that defines the hybrid weights."""
        if not self.parameters["hybrid"]:
            return emission_array

        if self.freeze_emission_kernel and self.frozen_emission_kernel is not None:
            return self.frozen_emission_kernel

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

    def apply(self, x):
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


if NUMBA_AVAIL:
    # --- existing numba kernels (_nb_kernel, _nb_kernel_mask, _nb_adjoint) ---
    # (unchanged, already include hybrid in the mask‐kernel version)

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

            if use_mask:
                if self.mask is None or recalc_mask:
                    self.mask = self.precompute_mask()
                res = _nb_kernel_mask(
                    x_arr,
                    ref_arr,
                    arr,
                    norm_arr,
                    self.mask,
                    n,
                    sigma_anat,
                    sigma_dist,
                    sigma_emission,
                    normalize_kernel,
                    distance_weighting,
                    hybrid,
                )
            elif hybrid:
                full_mask = self._get_full_mask(arr.shape, n)
                res = _nb_kernel_mask(
                    x_arr,
                    ref_arr,
                    arr,
                    norm_arr,
                    full_mask,
                    n,
                    sigma_anat,
                    sigma_dist,
                    sigma_emission,
                    normalize_kernel,
                    distance_weighting,
                    hybrid,
                )
            else:
                res = _nb_kernel(
                    x_arr,
                    ref_arr,
                    arr,
                    norm_arr,
                    n,
                    sigma_anat,
                    sigma_dist,
                    sigma_emission,
                    normalize_kernel,
                    distance_weighting,
                    hybrid,
                )

            out = image.clone()
            out.fill(res)
            self._normalisation_map = norm_arr if normalize_kernel else None
            return out

        def adjoint(self, x, out=None):
            arr = get_array(self.anatomical_image)
            x_arr = get_array(x)
            p = self.parameters
            norm_arr = (
                self._normalisation_map
                if p["normalize_kernel"]
                else np.zeros((1, 1, 1), dtype=np.float64)
            )
            if p["normalize_kernel"] and norm_arr is None:
                raise RuntimeError(
                    "Normalization map has not been initialised. "
                    "Call direct() before adjoint() when using normalize_kernel=True."
                )

            n = p["num_neighbours"]
            if p["use_mask"]:
                if self.mask is None or p["recalc_mask"]:
                    self.mask = self.precompute_mask()
                mask = self.mask
            else:
                mask = self._get_full_mask(arr.shape, n)

            ref_arr = self._get_hybrid_reference() if p["hybrid"] else x_arr

            res = _nb_adjoint(
                x_arr,
                ref_arr,
                arr,
                norm_arr,
                mask,
                p["use_mask"],
                n,
                p["sigma_anat"],
                p["sigma_dist"],
                p["sigma_emission"],
                p["distance_weighting"],
                p["hybrid"],
            )

            img = x.clone()
            img.fill(res)
            if out is None:
                return img
            out.fill(res)
            return out


    @numba.njit(cache=True, parallel=True)
    def _nb_precompute_mask(anat_arr, n, k_keep):
        s0, s1, s2 = anat_arr.shape
        total = n ** 3
        half = n // 2
        mask = np.zeros((s0, s1, s2, total), dtype=np.bool_)

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

                    sorted_diffs = np.sort(diffs)
                    threshold = sorted_diffs[k_keep - 1]
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

                                mask[i, j, k, idx] = diffs[idx] <= threshold
                                idx += 1

        return mask


    @numba.njit(cache=True, parallel=True)
    def _nb_kernel(
        x_arr, ref_arr, anat_arr, norm_arr, n,
        sigma_anat, sigma_dist, sigma_emission,
        normalize, distance_weighting, hybrid,
    ):
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist
        sig2_em = 2.0 * sigma_emission * sigma_emission

        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        out = np.empty_like(anat_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    ca = anat_arr[i, j, k]
                    c_ref = ref_arr[i, j, k]
                    sumv = 0.0
                    wsum = 0.0

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

                                # weights & accumulation MUST be inside dk-loop
                                diff_an = anat_arr[ii, jj, kk] - ca
                                wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                w = wi_an * wd_an[di + half, dj + half, dk + half]
                                if hybrid:
                                    diff_em = ref_arr[ii, jj, kk] - c_ref
                                    wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                    w *= wi_em

                                sumv += x_arr[ii, jj, kk] * w
                                wsum += w

                    if normalize:
                        if wsum > 1e-12:
                            sumv /= wsum
                            norm_arr[i, j, k] = wsum
                        else:
                            norm_arr[i, j, k] = 1.0
                    out[i, j, k] = sumv

        return out


    @numba.njit(cache=True, parallel=True)
    def _nb_kernel_mask(
        x_arr,
        ref_arr,
        anat_arr,
        norm_arr,
        mask,
        n,
        sigma_anat,
        sigma_dist,
        sigma_emission,
        normalize,
        distance_weighting,
        hybrid,
    ):
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist
        sig2_em = 2.0 * sigma_emission * sigma_emission

        # precompute spatial weights
        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)
        out = np.empty_like(anat_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    ca = anat_arr[i, j, k]
                    c_ref = ref_arr[i, j, k]
                    sumv = 0.0
                    wsum = 0.0
                    idx = 0

                    for di in range(-half, half + 1):
                        for dj in range(-half, half + 1):
                            for dk in range(-half, half + 1):
                                if mask[i, j, k, idx]:
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
                                    kk = k + dk
                                    if kk < 0:
                                        kk = -kk - 1
                                    elif kk >= s2:
                                        kk = 2 * s2 - kk - 1

                                    # anat weight
                                    diff_an = anat_arr[ii, jj, kk] - ca
                                    wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                    w = wi_an * wd_an[di + half, dj + half, dk + half]

                                    # hybrid emission
                                    if hybrid:
                                        diff_em = ref_arr[ii, jj, kk] - c_ref
                                        wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                        w *= wi_em

                                    sumv += x_arr[ii, jj, kk] * w
                                    wsum += w
                                idx += 1

                    if normalize:
                        norm = wsum if wsum > 1e-12 else 1.0
                        if wsum > 1e-12:
                            sumv /= wsum
                        norm_arr[i, j, k] = norm
                    out[i, j, k] = sumv

        return out


    @numba.njit(cache=True, parallel=True)
    def _nb_adjoint(
        x_arr,
        ref_arr,
        anat_arr,
        norm_arr,
        mask,
        use_mask,
        n,
        sigma_anat,
        sigma_dist,
        sigma_emission,
        distance_weighting,
        hybrid,
    ):
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist
        sig2_em = 2.0 * sigma_emission * sigma_emission

        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        out = np.zeros_like(anat_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    cv = anat_arr[i, j, k]
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

                                do_weight = (not use_mask) or mask[i, j, k, idx]
                                if do_weight:
                                    diff_an = anat_arr[ii, jj, kk] - cv
                                    wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                    w = wi_an * wd_an[di + half, dj + half, dk + half]

                                    if hybrid:
                                        diff_em = ref_arr[ii, jj, kk] - c_ref
                                        wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                        w *= wi_em

                                    out[ii, jj, kk] += val * w
                                idx += 1

        return out


else:

    class KernelOperator(BaseKernelOperator):  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Numba backend not available. Please install numba to use KernelOperator."
            )
