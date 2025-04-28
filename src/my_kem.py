import numpy as np
import math
from cil.optimisation.operators import LinearOperator

# Try importing optional backends
try:
    import numba
    NUMBA_AVAIL = True
except ImportError:
    NUMBA_AVAIL = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False

import numpy as np
import math
from cil.optimisation.operators import LinearOperator

# optional numba
try:
    import numba
    NUMBA_AVAIL = True
except ImportError:
    NUMBA_AVAIL = False


def get_kernel_operator(domain_geometry, backend='auto', **kwargs):
    if backend == 'auto':
        backend = 'torch' if TORCH_AVAIL else 'numba' if NUMBA_AVAIL else 'python'
    if backend == 'numba':
        return NumbaKernelOperator(domain_geometry, **kwargs)
    elif backend == 'torch':
        return TorchKernelOperator(domain_geometry, **kwargs)
    else:
        return KernelOperator(domain_geometry, **kwargs)
class BaseKernelOperator(LinearOperator):
    def __init__(self, domain_geometry, **kwargs):
        super().__init__(domain_geometry=domain_geometry,
                         range_geometry=domain_geometry)
        # parameters
        params = {
            'num_neighbours': 5,
            'sigma_anat': 2.0,
            'sigma_dist': 1.0,
            'prop_features': 1,
            'normalize_features': False,
            'normalize_kernel': False,
        }
        self.parameters = {**params, **kwargs}
        self.anatomical_image = None
        self._feature_masks = None

        # precompute fixed data
        n = self.parameters['num_neighbours']
        half = n // 2
        # all offsets in [-half..half]^3, shape (K,3)
        coords = np.arange(-half, half+1)
        grid = np.stack(np.meshgrid(coords, coords, coords, indexing='ij'), -1)
        self._offsets = grid.reshape(-1, 3).astype(np.int64)  # (K,3)

        # spatial weight per offset, shape (K,)
        D2 = (self._offsets**2).sum(axis=1)
        sd = self.parameters['sigma_dist']
        self._spatial_flat = np.exp(-D2 / (2 * sd * sd))

        # single pad tuple for reflect-padding
        self._pad = ((half, half), (half, half), (half, half))


    def set_anatomical_image(self, image):
        """
        Stores the anatomical image and precomputes, for each
        target voxel, which offsets to keep for the p-fraction.
        """
        self.anatomical_image = image
        arr = image.as_array()
        arr_p = np.pad(arr, self._pad, mode='reflect')

        S0, S1, S2 = arr.shape
        K = self._offsets.shape[0]
        p = self.parameters['prop_features']

        masks = np.empty((S0, S1, S2, K), dtype=np.bool_)
        half = self.parameters['num_neighbours'] // 2

        # for each centre voxel, gather its n^3 neighbours,
        # compute d², threshold to top-p fraction, store mask
        for i in range(S0):
            ci = i + half
            for j in range(S1):
                cj = j + half
                for k in range(S2):
                    ck = k + half
                    centre = arr_p[ci, cj, ck]

                    # vectorized gather of the K neighbours
                    di, dj, dk = self._offsets.T
                    neigh = arr_p[ci + di, cj + dj, ck + dk]   # shape (K,)

                    d2 = (neigh - centre)**2
                    if p < 1.0:
                        keep_n = max(int(math.ceil(p * K)), 1)
                        cutoff = np.partition(d2, keep_n-1)[keep_n-1]
                        masks[i, j, k] = (d2 <= cutoff)
                    else:
                        masks[i, j, k] = True

        self._feature_masks = masks


    def apply(self, x):
        return self.neighbourhood_kernel(
            x,
            self.anatomical_image,
            self.parameters['num_neighbours'],
            self.parameters['sigma_anat'],
            self.parameters['sigma_dist'],
            self.parameters['prop_features']
        )

    def direct(self, x, out=None):
        res = self.apply(x)
        if out is None:
            return res
        out.fill(res.as_array())
        return out

    def adjoint(self, x, out=None):
        return self.direct(x, out)

    def neighbourhood_kernel(self, x, image,
                             n, sigma_anat, sigma_dist,
                             prop_features):
        """To be implemented by subclasses."""
        raise NotImplementedError



class KernelOperator(BaseKernelOperator):
    def neighbourhood_kernel(self, x, image,
                             n, sigma_anat, sigma_dist,
                             prop_features):
        arr = image.as_array()
        x_arr = x.as_array()
        arr_p = np.pad(arr,   self._pad, mode='reflect')
        x_p   = np.pad(x_arr, self._pad, mode='reflect')

        S0, S1, S2 = arr.shape
        res = np.empty_like(arr, dtype=np.float64)
        half = n // 2
        di, dj, dk = self._offsets.T  # each shape (K,)

        for i in range(S0):
            ci = i + half
            for j in range(S1):
                cj = j + half
                for k in range(S2):
                    ck = k + half
                    centre = arr_p[ci, cj, ck]

                    neigh = arr_p[ci+di, cj+dj, ck+dk]
                    x_vals = x_p[ci+di, cj+dj, ck+dk]

                    mask = self._feature_masks[i, j, k]
                    d2 = (neigh - centre)**2

                    w_int = np.exp(-d2[mask] / (2 * sigma_anat * sigma_anat))
                    w = w_int * self._spatial_flat[mask]
                    if self.parameters['normalize_kernel']:
                        S = w.sum()
                        if S > 1e-12:
                            w /= S

                    res[i, j, k] = np.dot(x_vals[mask], w)

        out = image.clone()
        out.fill(res)
        return out


if NUMBA_AVAIL:
    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def _nb_kernel_mask(
        x_arr, anat_p, x_p,
        offsets, spatial_flat, masks,
        n, sigma_anat, normalize
    ):
        S0, S1, S2, K = masks.shape
        out = np.empty((S0, S1, S2), dtype=np.float64)
        half = n // 2
        sig2 = 2.0 * sigma_anat * sigma_anat

        for idx in numba.prange(S0 * S1 * S2):
            i = idx // (S1 * S2)
            rem = idx %  (S1 * S2)
            j = rem // S2
            k = rem %  S2

            ci = i + half
            cj = j + half
            ck = k + half

            centre = anat_p[ci, cj, ck]
            sumv = 0.0
            wsum = 0.0

            for m in range(K):
                if not masks[i, j, k, m]:
                    continue
                di, dj, dk = offsets[m]
                xi = x_p[ci+di, cj+dj, ck+dk]
                ai = anat_p[ci+di, cj+dj, ck+dk]
                d2 = (ai - centre)**2
                wi = math.exp(-d2 / sig2)
                w  = wi * spatial_flat[m]
                sumv += xi * w
                wsum += w

            if normalize and wsum > 1e-12:
                sumv /= wsum

            out[i, j, k] = sumv

        return out


    class NumbaKernelOperator(BaseKernelOperator):
        def __init__(self, domain_geometry, **kwargs):
            super().__init__(domain_geometry, **kwargs)
            self.backend = 'numba'

        def neighbourhood_kernel(self, x, image,
                                 n, sigma_anat, sigma_dist,
                                 prop_features):
            arr = image.as_array()
            arr_p = np.pad(arr,   self._pad, mode='reflect')
            x_arr = x.as_array()
            x_p   = np.pad(x_arr, self._pad, mode='reflect')

            res = _nb_kernel_mask(
                x_arr, arr_p, x_p,
                self._offsets, self._spatial_flat,
                self._feature_masks,
                n, sigma_anat,
                self.parameters['normalize_kernel']
            )
            out = image.clone()
            out.fill(res)
            return out

# --- PyTorch accelerated implementation ---
class TorchKernelOperator(BaseKernelOperator):
    def neighbourhood_kernel(
        self, x, image,
        n, sigma_anat, sigma_dist,
        prop_features
    ):
        # move data to GPU
        x_t = torch.from_numpy(x.as_array()).to('cuda', non_blocking=True)
        a_t = torch.from_numpy(image.as_array()).to('cuda', non_blocking=True)
        p = n // 2
        pad = (p, p, p, p, p, p)
        x_p = F.pad(x_t.unsqueeze(0).unsqueeze(0), pad, mode='reflect')
        a_p = F.pad(a_t.unsqueeze(0).unsqueeze(0), pad, mode='reflect')

        X = x_p.unfold(2, n, 1).unfold(3, n, 1).unfold(4, n, 1)
        A = a_p.unfold(2, n, 1).unfold(3, n, 1).unfold(4, n, 1)
        B, C, D, H, W, _, _, _ = X.shape
        K = n**3
        X = X.reshape(B, C, D, H, W, K)
        A = A.reshape(B, C, D, H, W, K)

        cv = a_t.unsqueeze(-1)
        wi = torch.exp(-((A - cv)**2) / (2 * sigma_anat * sigma_anat))
        # <<< use _spatial_flat, not _spatial_full >>>
        ws = torch.from_numpy(
            self._spatial_flat.reshape(1, 1, 1, 1, 1, K)
        ).to(x_t.device, non_blocking=True)
        w = wi * ws

        if prop_features < 1.0:
            diff2 = ((A - cv)**2).reshape(-1)
            k = max(int(math.ceil(prop_features * diff2.numel())), 1)
            cutoff = torch.kthvalue(diff2, k, dim=0).values
            mask = ((A - cv)**2) <= cutoff
            w = w * mask

        if self.parameters['normalize_kernel']:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)

        out = (w * X).sum(dim=-1).squeeze()
        res = image.clone()
        res.fill(out.cpu().numpy())
        return res

