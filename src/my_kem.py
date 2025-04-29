import numpy as np
np.seterr(over='raise', invalid='raise')

from cil.optimisation.operators import LinearOperator
from numpy.lib.stride_tricks import sliding_window_view

# Try importing numba
try:
    import numba
    NUMBA_AVAIL = True
except ImportError:
    NUMBA_AVAIL = False

# Try importing torch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


def get_kernel_operator(domain_geometry, backend='auto', **kwargs):
    """
    Returns the best available kernel operator.
    backend: 'auto'|'torch'|'numba'|'python'
    auto order: torch → numba → python
    """
    if backend == 'auto':
        if TORCH_AVAIL:
            backend = 'torch'
        elif NUMBA_AVAIL:
            backend = 'numba'
        else:
            backend = 'python'

    if backend == 'torch' and TORCH_AVAIL:
        return TorchKernelOperator(domain_geometry, **kwargs)
    elif backend == 'numba' and NUMBA_AVAIL:
        return NumbaKernelOperator(domain_geometry, **kwargs)
    else:
        return KernelOperator(domain_geometry, **kwargs)


class BaseKernelOperator(LinearOperator):
    def __init__(self, domain_geometry, **kwargs):
        super().__init__(domain_geometry=domain_geometry,
                         range_geometry=domain_geometry)
        default_parameters = {
            'num_neighbours':    5,
            'sigma_anat':        0.1,
            'sigma_dist':        0.1,
            'normalize_features': False,
            'normalize_kernel':   False,
            'use_mask':           False,
            'mask_k':             None,
            'recalc_mask':        False,
        }
        self.parameters = {**default_parameters, **kwargs}
        self.anatomical_image = None
        self.mask = None
        self.backend = 'python'

    def set_parameters(self, parameters):
        self.parameters.update(parameters)
        # clear mask so it will be rebuilt
        self.mask = None
        # rebuild torch spatial weights if needed
        if self.backend == 'torch':
            n  = self.parameters['num_neighbours']
            sd = self.parameters['sigma_dist']
            coords = torch.stack(torch.meshgrid(
                torch.arange(n), torch.arange(n), torch.arange(n),
                indexing='ij'), dim=-1).float()
            c = (n-1)/2
            D2 = ((coords-c)**2).sum(dim=-1)
            self._spatial = torch.exp(-D2/(2*sd*sd)).reshape(-1).cuda()

    def set_anatomical_image(self, image):
        if self.parameters['normalize_features']:
            arr = image.as_array()
            std = arr.std()
            norm = arr / std if std > 1e-12 else arr
            tmp = image.clone()
            tmp.fill(norm)
            self.anatomical_image = tmp
        else:
            self.anatomical_image = image
        self.mask = None

    def precompute_mask(self):
        n = self.parameters['num_neighbours']
        K = n**3
        k = self.parameters['mask_k'] or K
        arr = self.anatomical_image.as_array()
        pad = n//2

        arr_p = np.pad(arr, pad, mode='reflect')
        neigh = sliding_window_view(arr_p, (n,n,n))    # → (S0,S1,S2,n,n,n)
        S0,S1,S2,_,_,_ = neigh.shape
        flat = neigh.reshape(S0,S1,S2,K)               # → (S0,S1,S2,K)

        center = arr[...,None]                          # → (S0,S1,S2,1)
        diff   = np.abs(flat - center)                  # → (S0,S1,S2,K)

        thresh = np.partition(diff, k-1, axis=-1)[...,k-1:k]  # (S0,S1,S2,1)
        mask   = diff <= thresh                         # boolean mask
        return mask

    def apply(self, x):
        p = self.parameters
        return self.neighbourhood_kernel(
            x,
            self.anatomical_image,
            p['num_neighbours'],
            p['sigma_anat'],
            p['sigma_dist'],
            p['normalize_kernel'],
            p['use_mask'],
            p['recalc_mask']
        )

    def direct(self, x, out=None):
        res = self.apply(x)
        if out is None:
            return res
        out.fill(res.as_array())
        return out

    def adjoint(self, x, out=None):
        # default adjoint = direct (for python backend)
        res = self.direct(x)
        if out is None:
            return res
        out.fill(res.as_array())
        return out


class KernelOperator(BaseKernelOperator):
    def neighbourhood_kernel(self,
                             x, image,
                             num_neighbours,
                             sigma_anat,
                             sigma_dist,
                             normalize_kernel,
                             use_mask,
                             recalc_mask):
        arr   = image.as_array()
        x_arr = x.as_array()
        n     = num_neighbours
        pad   = n // 2
        K     = n**3

        if use_mask:
            if self.mask is None or recalc_mask:
                self.mask = self.precompute_mask()
            mask = self.mask  # shape (S0,S1,S2,K)

        # forward: reflect-pad and sliding windows
        arr_p   = np.pad(arr, pad, mode='reflect')
        x_p     = np.pad(x_arr, pad, mode='reflect')
        neigh   = sliding_window_view(arr_p, (n,n,n))    # (S0,S1,S2,n,n,n)
        x_neigh = sliding_window_view(x_p, (n,n,n))

        S0,S1,S2,_,_,_ = neigh.shape

        # spatial weights
        coords = np.arange(-pad, pad+1)
        D2     = coords[:,None,None]**2 + coords[None,:,None]**2 + coords[None,None,:]**2
        W_dist = np.exp(-D2/(2*sigma_dist**2))

        center = arr[...,None,None,None]
        W_int  = np.exp(-((neigh - center)**2)/(2*sigma_anat**2))
        W      = W_int * W_dist                         # (S0,S1,S2,n,n,n)

        if use_mask:
            W_flat = W.reshape(S0,S1,S2,K)
            W_flat *= mask
            W      = W_flat.reshape(S0,S1,S2,n,n,n)

        if normalize_kernel:
            denom = W.sum(axis=(3,4,5), keepdims=True)
            W     = W / (denom + 1e-12)

        res = (W * x_neigh).sum(axis=(3,4,5))
        out = image.clone()
        out.fill(res)
        return out


if NUMBA_AVAIL:
    # --- Numba forward kernels ---
    @numba.njit(cache=True, parallel=True)
    def _nb_kernel(x_arr, anat_arr, n, sigma, sigma_dist, normalize):
        s0,s1,s2 = anat_arr.shape
        half     = n // 2
        sig2     = 2.0 * sigma * sigma
        dist2    = 2.0 * sigma_dist * sigma_dist
        out      = np.empty_like(anat_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    cv = anat_arr[i,j,k]
                    sumv = 0.0
                    wsum = 0.0
                    # reflect + full n^3 loop
                    for di in range(-half, half+1):
                        ii = i+di
                        if ii<0:    ii = -ii-1
                        elif ii>=s0:ii = 2*s0-ii-1
                        for dj in range(-half, half+1):
                            jj = j+dj
                            if jj<0:    jj = -jj-1
                            elif jj>=s1:jj = 2*s1-jj-1
                            for dk in range(-half, half+1):
                                kk = k+dk
                                if kk<0:    kk = -kk-1
                                elif kk>=s2:kk = 2*s2-kk-1

                                diff = anat_arr[ii,jj,kk] - cv
                                wi   = np.exp(-(diff*diff)/sig2)
                                d2   = di*di + dj*dj + dk*dk
                                wd   = np.exp(-d2/dist2)
                                w    = wi * wd

                                sumv += x_arr[ii,jj,kk] * w
                                wsum += w

                    if normalize and wsum>1e-12:
                        sumv /= wsum
                    out[i,j,k] = sumv
        return out

    @numba.njit(cache=True, parallel=True)
    def _nb_kernel_mask(x_arr, anat_arr, mask, n, sigma, sigma_dist, normalize):
        s0,s1,s2 = anat_arr.shape
        half     = n // 2
        sig2     = 2.0 * sigma * sigma
        dist2    = 2.0 * sigma_dist * sigma_dist
        out      = np.empty_like(anat_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    cv   = anat_arr[i,j,k]
                    sumv = 0.0
                    wsum = 0.0
                    idx  = 0
                    for di in range(-half, half+1):
                        for dj in range(-half, half+1):
                            for dk in range(-half, half+1):
                                if mask[i,j,k,idx] != 0:
                                    ii = i+di
                                    if ii<0:    ii = -ii-1
                                    elif ii>=s0: ii = 2*s0-ii-1
                                    jj = j+dj
                                    if jj<0:    jj = -jj-1
                                    elif jj>=s1:jj = 2*s1-jj-1
                                    kk = k+dk
                                    if kk<0:    kk = -kk-1
                                    elif kk>=s2:kk = 2*s2-kk-1

                                    diff = anat_arr[ii,jj,kk] - cv
                                    wi   = np.exp(-(diff*diff)/sig2)
                                    d2   = di*di + dj*dj + dk*dk
                                    wd   = np.exp(-d2/dist2)
                                    w    = wi * wd

                                    sumv += x_arr[ii,jj,kk] * w
                                    wsum += w
                                idx += 1

                    if normalize and wsum>1e-12:
                        sumv /= wsum
                    out[i,j,k] = sumv
        return out

    # --- Numba adjoint kernel ---
    @numba.njit(cache=True, parallel=True)
    def _nb_adjoint(x_arr, anat_arr, mask, use_mask,
                    n, sigma_anat, sigma_dist):
        s0,s1,s2 = anat_arr.shape
        half     = n // 2
        sig2     = 2.0 * sigma_anat * sigma_anat
        dist2    = 2.0 * sigma_dist * sigma_dist
        out      = np.zeros_like(anat_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    cv  = anat_arr[i,j,k]
                    val = x_arr[i,j,k]
                    idx = 0
                    for di in range(-half, half+1):
                        ii = i+di
                        if ii<0:    ii = -ii-1
                        elif ii>=s0:ii = 2*s0-ii-1
                        for dj in range(-half, half+1):
                            jj = j+dj
                            if jj<0:    jj = -jj-1
                            elif jj>=s1:jj = 2*s1-jj-1
                            for dk in range(-half, half+1):
                                kk = k+dk
                                if kk<0:    kk = -kk-1
                                elif kk>=s2:kk = 2*s2-kk-1

                                if (not use_mask) or (mask[i,j,k,idx] != 0):
                                    diff = anat_arr[ii,jj,kk] - cv
                                    wi   = np.exp(-(diff*diff)/sig2)
                                    d2   = di*di + dj*dj + dk*dk
                                    wd   = np.exp(-d2/dist2)
                                    w    = wi * wd
                                    out[ii,jj,kk] += val * w
                                idx += 1

        return out

    class NumbaKernelOperator(BaseKernelOperator):
        def __init__(self, domain_geometry, **kwargs):
            super().__init__(domain_geometry, **kwargs)
            self.backend = 'numba'

        def neighbourhood_kernel(self,
                                 x, image,
                                 num_neighbours,
                                 sigma_anat,
                                 sigma_dist,
                                 normalize_kernel,
                                 use_mask,
                                 recalc_mask):
            arr   = image.as_array()
            x_arr = x.as_array()
            n     = num_neighbours

            if use_mask:
                if self.mask is None or recalc_mask:
                    self.mask = self.precompute_mask()
                mask_int = self.mask.astype(np.int8)
                res = _nb_kernel_mask(x_arr, arr, mask_int,
                                      n, sigma_anat, sigma_dist,
                                      normalize_kernel)
            else:
                res = _nb_kernel(x_arr, arr,
                                 n, sigma_anat, sigma_dist,
                                 normalize_kernel)

            out = image.clone(); out.fill(res); return out

        def adjoint(self, x, out=None):
            arr   = self.anatomical_image.as_array()
            x_arr = x.as_array()
            p     = self.parameters
            if p['use_mask']:
                if self.mask is None or p['recalc_mask']:
                    self.mask = self.precompute_mask()
                mask_int = self.mask.astype(np.int8)
            else:
                mask_int = np.zeros((1,), dtype=np.int8)

            res = _nb_adjoint(x_arr, arr, mask_int,
                              p['use_mask'],
                              p['num_neighbours'],
                              p['sigma_anat'],
                              p['sigma_dist'])
            img = x.clone(); img.fill(res)
            if out is None:
                return img
            out.fill(res); return out