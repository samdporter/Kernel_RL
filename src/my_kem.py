import numpy as np
from cil.optimisation.operators import LinearOperator

# Try importing numba
try:
    import numba
    NUMBA_AVAIL = True
except ImportError:
    NUMBA_AVAIL = False

# Factory function
def get_kernel_operator(domain_geometry, backend='auto', **kwargs):
    """
    Returns the best available kernel operator.
    backend: 'auto'|'torch'|'cupy'|'numba'|'python'
    auto order: cupy → torch → numba → python
    """
    if backend == 'auto':
        for b in ('cupy', 'torch', 'numba'):
            try:
                __import__(b)
                backend = b
                break
            except ImportError:
                continue
        else:
            backend = 'python'

    if backend == 'torch':
        return TorchKernelOperator(domain_geometry, **kwargs)
    elif backend == 'cupy':
        return CuPyKernelOperator(domain_geometry, **kwargs)
    elif backend == 'numba':
        return NumbaKernelOperator(domain_geometry, **kwargs)
    else:
        return KernelOperator(domain_geometry, **kwargs)



class BaseKernelOperator(LinearOperator):
    def __init__(self, domain_geometry, **kwargs):
        super().__init__(domain_geometry=domain_geometry,
                         range_geometry=domain_geometry)
        # default parameters
        default_parameters = {
            'num_neighbours': 5,
            'sigma_anat': 2.0,
            'sigma_dist': 1.0,
            'normalize_features': False,
            'normalize_kernel': False,
        }
        self.parameters = {**default_parameters, **kwargs}
        self.anatomical_image = None

    def set_anatomical_image(self, image):
        if self.parameters['normalize_features']:
            arr = image.as_array()
            std = arr.std()
            norm = arr / std if std > 1e-12 else arr
            temp = image.clone()
            temp.fill(norm)
            self.anatomical_image = temp
        else:
            self.anatomical_image = image

    def apply(self, x):
        return self.neighbourhood_kernel(
            x,
            self.anatomical_image,
            self.parameters['num_neighbours'],
            self.parameters['sigma_anat']
        )

    def direct(self, x, out=None):
        res = self.apply(x)
        if out is None:
            return res
        out.fill(res.as_array())
        return out

    def adjoint(self, x, out=None):
        return self.direct(x, out)

    def neighbourhood_kernel(self, x, image, n, sigma):
        raise NotImplementedError
    
    def clear_gpu(self):
        """Release any cached GPU memory."""
        if self.backend == 'torch':
            # free PyTorch’s CUDA cache
            torch.cuda.empty_cache()
        elif self.backend == 'cupy':
            # free all blocks in CuPy’s default memory pool
            cp.get_default_memory_pool().free_all_blocks()
        else:
            # do nothing for pure Python
            pass


# --- Pure Python KernelOperator ---
class KernelOperator(BaseKernelOperator):
    def voxel_neighbourhood_kernel(self, x_arr, anat_arr, cx, cy, cz,
                                  n, sigma, sigma_dist):
        half = n//2
        s0,s1,s2 = anat_arr.shape
        i0,i1 = max(cx-half,0), min(cx+half+1,s0)
        j0,j1 = max(cy-half,0), min(cy+half+1,s1)
        k0,k1 = max(cz-half,0), min(cz+half+1,s2)
        neigh = anat_arr[i0:i1,j0:j1,k0:k1]
        x_neigh = x_arr[i0:i1,j0:j1,k0:k1]
        center = anat_arr[cx,cy,cz]
        # distance squared
        I = np.arange(i0,i1)-cx
        J = np.arange(j0,j1)-cy
        K = np.arange(k0,k1)-cz
        D2 = I[:,None,None]**2 + J[None,:,None]**2 + K[None,None,:]**2
        w_int = np.exp(-((neigh-center)**2)/(2*sigma**2))
        w_dist = np.exp(-D2/(2*sigma_dist**2))
        w = w_int * w_dist
        if self.parameters['normalize_kernel']:
            s = w.sum()
            if s>1e-12: w/=s
        return (x_neigh * w).sum()

    def neighbourhood_kernel(self, x, image, n, sigma):
        arr = image.as_array()
        x_arr = x.as_array()
        s0,s1,s2 = arr.shape
        res = np.empty_like(arr, dtype=np.float64)
        sigma_dist = self.parameters['sigma_dist']
        for i in range(s0):
            for j in range(s1):
                for k in range(s2):
                    res[i,j,k] = self.voxel_neighbourhood_kernel(
                        x_arr, arr, i, j, k,
                        n, sigma, sigma_dist)
        out = image.clone()
        out.fill(res)
        return out


# --- Numba accelerated operator ---
if NUMBA_AVAIL:
    @numba.njit(parallel=True)
    def _nb_kernel(x_arr, anat_arr, n, sigma, sigma_dist, normalize):
        s0,s1,s2 = anat_arr.shape
        out = np.empty_like(anat_arr, dtype=np.float64)
        half = n//2
        sig2 = 2*sigma*sigma
        dist2 = 2*sigma_dist*sigma_dist
        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    i0,i1 = max(i-half,0), min(i+half+1,s0)
                    j0,j1 = max(j-half,0), min(j+half+1,s1)
                    k0,k1 = max(k-half,0), min(k+half+1,s2)
                    cv = anat_arr[i,j,k]
                    sumv=0.0; wsum=0.0
                    for ii in range(i0,i1):
                        for jj in range(j0,j1):
                            for kk in range(k0,k1):
                                diff = anat_arr[ii,jj,kk]-cv
                                wi = np.exp(-(diff*diff)/sig2)
                                d2 = (ii-i)**2+(jj-j)**2+(kk-k)**2
                                wd = np.exp(-d2/dist2)
                                w = wi*wd
                                sumv += x_arr[ii,jj,kk]*w
                                wsum += w
                    if normalize and wsum>1e-12:
                        sumv/=wsum
                    out[i,j,k] = sumv
        return out

    class NumbaKernelOperator(BaseKernelOperator):
        
        def __init__(self, domain_geometry, **kwargs):
            super().__init__(domain_geometry, **kwargs)
            self.backend = 'numba'
        def neighbourhood_kernel(self, x, image, n, sigma):
            arr = image.as_array()
            x_arr = x.as_array()
            res = _nb_kernel(x_arr, arr,
                             n, sigma,
                             self.parameters['sigma_dist'],
                             self.parameters['normalize_kernel'])
            out = image.clone()
            out.fill(res)
            return out


# --- PyTorch accelerated operator ---
try:
    import torch
    import torch.nn.functional as F
    class TorchKernelOperator(BaseKernelOperator):
        def __init__(self, domain_geometry, **kwargs):
            super().__init__(domain_geometry, **kwargs)
            n = self.parameters['num_neighbours']
            coords = torch.stack(torch.meshgrid(
                torch.arange(n),torch.arange(n),torch.arange(n),
                indexing='ij'), dim=-1).float()
            c = (n-1)/2
            D2 = ((coords-c)**2).sum(dim=-1)
            sd = self.parameters['sigma_dist']
            self._spatial = torch.exp(-D2/(2*sd*sd)).reshape(-1).cuda()
            self.backend = 'torch'
        def neighbourhood_kernel(self, x, image, n, sigma_anat):
            x_t = torch.tensor(x.as_array(),dtype=torch.float32,device='cuda')
            a_t = torch.tensor(image.as_array(),dtype=torch.float32,device='cuda')
            p=n//2; pad=(p,p,p,p,p,p)
            x_p=F.pad(x_t.unsqueeze(0).unsqueeze(0),pad,mode='reflect')
            a_p=F.pad(a_t.unsqueeze(0).unsqueeze(0),pad,mode='reflect')
            X = x_p.unfold(2,n,1).unfold(3,n,1).unfold(4,n,1)
            A = a_p.unfold(2,n,1).unfold(3,n,1).unfold(4,n,1)
            B,C,D,H,W,_,_,_ = X.shape; K=n**3
            X = X.reshape(B,C,D,H,W,K); A = A.reshape(B,C,D,H,W,K)
            cv = a_t.unsqueeze(-1)
            wi= torch.exp(-((A-cv)**2)/(2*sigma_anat*sigma_anat))
            ws= self._spatial.view(1,1,1,1,1,K)
            w = wi*ws
            if self.parameters['normalize_kernel']:
                w = w/(w.sum(dim=-1,keepdim=True)+1e-12)
            out = (w*X).sum(dim=-1).squeeze()
            res = image.clone(); res.fill(out.cpu().numpy())
            del x_p,a_p,X,A,wi,ws,w
            self.clear_gpu()
            return res
except ImportError:
    pass

# --- CuPy accelerated operator ---
try:
    import cupy as cp
    try:
    # CuPy ≥9.0
        from cupy.lib.stride_tricks import sliding_window_view
        print('Using CuPy >= 9.0')
    except ImportError:
        # Older CuPy
        try:
            from cupyx.lib.stride_tricks import sliding_window_view
            print('Using CuPy < 9.0')
        except ImportError:
            raise ImportError('CuPy not found or incompatible version.')
    class CuPyKernelOperator(BaseKernelOperator):
        def __init__(self, domain_geometry, **kwargs):
            super().__init__(domain_geometry, **kwargs)
            n=self.parameters['num_neighbours']
            coords=cp.stack(cp.meshgrid(
                cp.arange(n),cp.arange(n),cp.arange(n),
                indexing='ij'),axis=-1).astype(cp.float32)
            c=(n-1)/2
            D2=cp.sum((coords-c)**2,axis=-1)
            sd=self.parameters['sigma_dist']
            self._spatial=cp.exp(-D2/(2*sd*sd)).ravel()
            self.backend='cupy'
        def neighbourhood_kernel(self,x,image,n,sigma_anat):
            arr_x=cp.asarray(x.as_array(),dtype=cp.float32)
            arr_a=cp.asarray(image.as_array(),dtype=cp.float32)
            p=n//2
            x_p=cp.pad(arr_x,((p,p),(p,p),(p,p)),mode='reflect')
            a_p=cp.pad(arr_a,((p,p),(p,p),(p,p)),mode='reflect')
            X=sliding_window_view(x_p,(n,n,n))
            A=sliding_window_view(a_p,(n,n,n))
            D,H,W,_,_,_ = X.shape; K=n**3
            X=X.reshape(D,H,W,K); A=A.reshape(D,H,W,K)
            cv=arr_a[...,None]
            wi=cp.exp(-((A-cv)**2)/(2*sigma_anat*sigma_anat))
            ws=self._spatial[None,None,None,:]
            w=wi*ws
            if self.parameters['normalize_kernel']:
                w=w/(w.sum(axis=-1,keepdims=True)+1e-12)
            out=(w*X).sum(axis=-1)
            res=image.clone();res.fill(cp.asnumpy(out))
            del x_p,a_p,X,A,wi,ws,w
            self.clear_gpu()
            return res
except ImportError:
    print('CuPy not available.')
    pass
