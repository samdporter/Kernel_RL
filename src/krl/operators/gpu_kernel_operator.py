"""PyTorch GPU implementation of the kernel operator for large-scale volumes.

This module provides a GPU-accelerated kernel operator using PyTorch CUDA kernels,
enabling processing of large volumes (e.g., 256³) on consumer GPUs with limited VRAM.

Key features:
- float32 precision (halves memory vs float64)
- Sparse neighbor masking for efficiency
- Pre-computed anatomical weights (cached on GPU)
- Automatic device management
- Compatible API with BaseKernelOperator

Memory requirements for 256³ volumes:
- Small config (n=5, k=20): ~3.6 GB (fits RTX 3060)
- Medium config (n=7, k=48): ~8.8 GB (fits RTX 4090)
"""

import numpy as np

# Import torch - define TORCH_AVAILABLE first, then try import
TORCH_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    # Create placeholder module-like objects for type checking
    torch = None  # type: ignore
    F = None  # type: ignore

# Import base class - use relative imports to work both in development and installed
try:
    # Try relative import first (works when installed as package)
    from .kernel_operator import BaseKernelOperator, DEFAULT_PARAMETERS, KernelOperator
    from ..utils import get_array
except ImportError:
    # Fall back to absolute import (works in development)
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.krl.operators.kernel_operator import BaseKernelOperator, DEFAULT_PARAMETERS, KernelOperator
    from krl.utils import get_array


class TorchKernelOperator(BaseKernelOperator):
    """
    GPU-accelerated kernel operator using PyTorch CUDA.

    Parameters
    ----------
    domain_geometry : ImageGeometry
        Domain geometry for the operator
    device : str, optional
        Device to use ('cuda', 'cuda:0', 'cpu', or 'auto'). Default: 'auto'
    dtype : str, optional
        Data type ('float32' or 'float64'). Default: 'float32' for memory efficiency
    **kwargs
        Additional parameters passed to BaseKernelOperator

    Attributes
    ----------
    device : torch.device
        PyTorch device (GPU or CPU)
    torch_dtype : torch.dtype
        PyTorch data type
    _mask_gpu : torch.Tensor
        Sparse mask indices stored on GPU
    """

    def __init__(self, domain_geometry, device='auto', dtype='float32', **kwargs):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not available. Install torch to use TorchKernelOperator."
            )

        max_gpu_batch_slices = kwargs.pop("max_gpu_batch_slices", 16)
        mask_chunk_limit_mb = kwargs.pop("mask_chunk_limit_mb", 256)

        cpu_kwargs = kwargs.copy()
        cpu_kwargs.pop("device", None)
        cpu_kwargs.pop("dtype", None)

        super().__init__(domain_geometry, **kwargs)
        self.backend = "torch"

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Set dtype
        if dtype == 'float32':
            self.torch_dtype = torch.float32
            self.numpy_dtype = np.float32
        elif dtype == 'float64':
            self.torch_dtype = torch.float64
            self.numpy_dtype = np.float64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Use 'float32' or 'float64'.")

        # GPU-specific caches
        self._mask_gpu = None
        self._mask_cpu = None
        self._normalisation_map_gpu = None
        self._normalisation_map_cpu = None
        self._cpu_init_kwargs = cpu_kwargs
        self._cpu_operator = None
        self.max_gpu_batch_slices = max(1, int(max_gpu_batch_slices))
        if mask_chunk_limit_mb is None or mask_chunk_limit_mb <= 0:
            self.mask_chunk_limit_bytes = None
        else:
            self.mask_chunk_limit_bytes = int(mask_chunk_limit_mb * 1024 * 1024)

        if self.device.type == 'cpu':
            try:
                self._cpu_operator = KernelOperator(domain_geometry, **cpu_kwargs)
                self._cpu_operator.freeze_emission_kernel = self.freeze_emission_kernel
            except RuntimeError:
                self._cpu_operator = None

        # Print device info
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_mem_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            print(f"TorchKernelOperator: Using {gpu_name} ({gpu_mem_gb:.1f} GB)")
        else:
            print("TorchKernelOperator: Using CPU (GPU not available)")

    def set_anatomical_image(self, image):
        """Override to clear GPU caches when anatomical image changes."""
        super().set_anatomical_image(image)
        self._mask_gpu = None
        self._mask_cpu = None
        if self._cpu_operator is not None:
            self._cpu_operator.set_anatomical_image(image)

    def set_parameters(self, parameters):
        """Override to clear GPU caches when parameters change."""
        super().set_parameters(parameters)
        self._mask_gpu = None
        self._mask_cpu = None
        if self._cpu_operator is not None:
            self._cpu_operator.set_parameters(parameters)

    def precompute_mask(self, dtype=None):
        """
        Precompute sparse mask as integer indices using PyTorch.

        Returns
        -------
        torch.Tensor
            Mask indices of shape (s0, s1, s2, k) with indices in [0, n³)
        """
        if dtype is None:
            if not TORCH_AVAILABLE or torch is None:
                raise RuntimeError("PyTorch is required for TorchKernelOperator.")
            dtype = torch.int32

        if self.device.type == 'cpu':
            # Reuse the well-tested CPU implementation and convert to torch tensor
            mask_np = super().precompute_mask()
            mask_tensor = torch.from_numpy(mask_np.astype(np.int32)).to(self.device)
            mask_tensor = mask_tensor.to(dtype=dtype)
            self._mask_cpu = mask_tensor.cpu()
            self._mask_gpu = None
            return mask_tensor

        if self.anatomical_image is None:
            raise RuntimeError("An anatomical image must be set before precomputing a mask.")

        n = int(self.parameters["num_neighbours"])
        total = n ** 3
        mask_k = self.parameters["mask_k"]
        k = mask_k if mask_k is not None else total
        k = max(1, min(int(k), total))

        # Get anatomical array and convert to torch tensor
        arr = np.ascontiguousarray(get_array(self.anatomical_image), dtype=self.numpy_dtype)
        anat_tensor = torch.from_numpy(arr).to(self.device)

        # Compute mask on GPU
        mask_indices = self._torch_precompute_mask(anat_tensor, n, k)
        if mask_indices.dtype != dtype:
            mask_indices = mask_indices.to(dtype=dtype)

        # Cache CPU copy and release GPU cache to minimise VRAM usage
        self._mask_cpu = mask_indices.cpu()
        self._mask_gpu = None

        return mask_indices

    def _get_mask_cpu(self, force_recompute=False, dtype=torch.int16):
        """Ensure the sparse mask is available on CPU with the desired dtype."""
        if force_recompute or self._mask_cpu is None or self._mask_cpu.dtype != dtype:
            self.precompute_mask(dtype=dtype)
            if self._mask_cpu is None:
                raise RuntimeError("Failed to precompute mask on CPU.")
            if self._mask_cpu.dtype != dtype:
                self._mask_cpu = self._mask_cpu.to(dtype)
        return self._mask_cpu

    def _select_batch_size(self, s0, s1, s2, k, mask_elem_bytes):
        """Select a batch size that respects mask chunk size limits."""
        base = min(s0, self.max_gpu_batch_slices)
        limit = self.mask_chunk_limit_bytes
        if limit is None or k <= 0:
            return base
        denom = s1 * s2 * k * max(1, int(mask_elem_bytes))
        if denom <= 0:
            return base
        max_from_limit = max(1, limit // denom)
        return max(1, min(base, max_from_limit))

    def precompute_anatomical_weights(self):
        """
        Pre-compute anatomical kernel weights. For GPU this computes weights on-the-fly
        in manageable batches and returns them as a torch tensor.
        """
        if self.device.type == 'cpu':
            weights_np = super().precompute_anatomical_weights()
            return torch.from_numpy(weights_np.astype(self.numpy_dtype)).to(self.device)

        if self.anatomical_image is None:
            raise RuntimeError("An anatomical image must be set before precomputing weights.")

        n = int(self.parameters["num_neighbours"])
        sigma_anat = self.parameters["sigma_anat"]
        sigma_dist = self.parameters["sigma_dist"]
        distance_weighting = self.parameters["distance_weighting"]
        use_mask = self.parameters["use_mask"]

        arr = np.ascontiguousarray(get_array(self.anatomical_image), dtype=self.numpy_dtype)
        anat_tensor = torch.from_numpy(arr).to(self.device)

        if use_mask:
            mask_cpu = self._get_mask_cpu(dtype=torch.int16)
            weights = self._torch_precompute_anatomical_weights_sparse(
                anat_tensor, mask_cpu, n, sigma_anat, sigma_dist, distance_weighting
            )
        else:
            weights = self._torch_precompute_anatomical_weights_dense(
                anat_tensor, n, sigma_anat, sigma_dist, distance_weighting
            )

        return weights

    def _torch_precompute_anatomical_weights_sparse(self, anat_arr, mask_cpu, n,
                                                    sigma_anat, sigma_dist, distance_weighting):
        """
        Pre-compute anatomical weights for sparse masked neighbors (GPU).
        """
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        k = mask_cpu.shape[3]
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist
        use_anat = sigma_anat > 0
        use_dist = distance_weighting and sigma_dist > 0

        anat_padded = F.pad(anat_arr.unsqueeze(0).unsqueeze(0),
                            (half, half, half, half, half, half),
                            mode='reflect').squeeze()

        weights = torch.empty((s0, s1, s2, k), dtype=self.torch_dtype, device=self.device)

        mask_elem_bytes = mask_cpu.element_size()
        batch_size = self._select_batch_size(s0, s1, s2, k, mask_elem_bytes)
        j_base = torch.arange(s1, device=self.device, dtype=torch.int64).view(1, -1, 1) + half
        k_base = torch.arange(s2, device=self.device, dtype=torch.int64).view(1, 1, -1) + half

        for i_start in range(0, s0, batch_size):
            i_end = min(i_start + batch_size, s0)
            center_vals = anat_arr[i_start:i_end, :, :]
            mask_chunk = mask_cpu[i_start:i_end].to(self.device, non_blocking=True)
            i_base = (
                torch.arange(i_start, i_end, device=self.device, dtype=torch.int64)
                .view(-1, 1, 1)
                + half
            )

            for k_idx in range(k):
                flat_indices = mask_chunk[:, :, :, k_idx].to(torch.int64)

                dk = (flat_indices % n) - half
                dj = ((flat_indices // n) % n) - half
                di = (flat_indices // (n * n)) - half

                i_indices = i_base + di
                j_indices = j_base + dj
                k_indices = k_base + dk

                neighbor_vals = anat_padded[i_indices, j_indices, k_indices]
                diff_an = neighbor_vals - center_vals

                if use_anat:
                    wi_an = torch.exp(-diff_an * diff_an / sig2_an)
                else:
                    wi_an = torch.ones_like(diff_an)

                if use_dist:
                    di_f = di.to(self.torch_dtype)
                    dj_f = dj.to(self.torch_dtype)
                    dk_f = dk.to(self.torch_dtype)
                    dist_sq = di_f * di_f + dj_f * dj_f + dk_f * dk_f
                    wi_an = wi_an * torch.exp(-dist_sq / dist2_an)

                weights[i_start:i_end, :, :, k_idx] = wi_an

        return weights

    def _torch_precompute_anatomical_weights_dense(self, anat_arr, n, sigma_anat,
                                                   sigma_dist, distance_weighting):
        """
        Pre-compute anatomical weights for all n³ neighbors (GPU).
        """
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        total = n ** 3
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist
        use_anat = sigma_anat > 0
        use_dist = distance_weighting and sigma_dist > 0

        anat_padded = F.pad(anat_arr.unsqueeze(0).unsqueeze(0),
                            (half, half, half, half, half, half),
                            mode='reflect').squeeze()

        weights = torch.empty((s0, s1, s2, total), dtype=self.torch_dtype, device=self.device)

        offsets = []
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                for dk in range(-half, half + 1):
                    offsets.append((di, dj, dk))

        batch_size = min(s0, self.max_gpu_batch_slices)

        for i_start in range(0, s0, batch_size):
            i_end = min(i_start + batch_size, s0)
            center_vals = anat_arr[i_start:i_end, :, :].unsqueeze(-1)

            for idx, (di, dj, dk) in enumerate(offsets):
                ii = slice(i_start + half + di, i_end + half + di)
                jj = slice(half + dj, half + dj + s1)
                kk = slice(half + dk, half + dk + s2)

                neighbor_vals = anat_padded[ii, jj, kk].unsqueeze(-1)
                diff_an = neighbor_vals - center_vals

                if use_anat:
                    wi_an = torch.exp(-diff_an * diff_an / sig2_an)
                else:
                    wi_an = torch.ones_like(diff_an)

                if use_dist:
                    dist_sq = (di * di + dj * dj + dk * dk)
                    dist_weight = torch.tensor(
                        np.exp(-dist_sq / dist2_an), dtype=self.torch_dtype, device=self.device
                    )
                    wi_an = wi_an * dist_weight

                weights[i_start:i_end, :, :, idx] = wi_an.squeeze(-1)

        return weights

    def _torch_precompute_mask(self, anat_arr, n, k_keep):
        """
        PyTorch implementation of sparse mask precomputation.

        For each voxel, find k_keep neighbors with smallest anatomical intensity differences.

        Parameters
        ----------
        anat_arr : torch.Tensor
            Anatomical image of shape (s0, s1, s2)
        n : int
            Neighborhood size (n×n×n cube)
        k_keep : int
            Number of neighbors to keep per voxel

        Returns
        -------
        torch.Tensor
            Integer indices of shape (s0, s1, s2, k_keep) with values in [0, n³)
        """
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        total = n ** 3

        # Create output tensor for mask indices
        mask_indices = torch.empty((s0, s1, s2, k_keep), dtype=torch.int16, device=self.device)

        # Pad anatomical array for boundary handling (reflection)
        anat_padded = F.pad(anat_arr.unsqueeze(0).unsqueeze(0),
                           (half, half, half, half, half, half),
                           mode='reflect').squeeze()

        # Build all n³ neighbor offsets
        offsets = []
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                for dk in range(-half, half + 1):
                    offsets.append((di, dj, dk))

        offset_indices = torch.arange(total, device=self.device, dtype=torch.int64)

        # For each voxel, gather neighbor values and compute differences
        # Process in batches to avoid OOM on very large volumes
        mask_elem_bytes = mask_indices.element_size()
        batch_size = self._select_batch_size(s0, s1, s2, k_keep, mask_elem_bytes)
        chunk_size = 32  # Number of neighbor offsets to process at once

        for i_start in range(0, s0, batch_size):
            i_end = min(i_start + batch_size, s0)
            batch_size_actual = i_end - i_start

            center_vals = anat_arr[i_start:i_end, :, :].unsqueeze(-1)  # (batch, s1, s2, 1)

            # Track best k differences/indices so far
            best_vals = torch.full(
                (batch_size_actual, s1, s2, k_keep),
                float("inf"),
                dtype=anat_arr.dtype,
                device=self.device,
            )
            best_idx = torch.zeros(
                (batch_size_actual, s1, s2, k_keep),
                dtype=torch.int16,
                device=self.device,
            )

            for chunk_start in range(0, total, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total)
                current_size = chunk_end - chunk_start

                # Gather neighbor values for this chunk
                neighbor_vals = torch.empty(
                    (batch_size_actual, s1, s2, current_size),
                    dtype=anat_arr.dtype,
                    device=self.device,
                )

                for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
                    di, dj, dk = offsets[global_idx]
                    ii = i_start + half + di
                    jj = slice(half + dj, half + dj + s1)
                    kk = slice(half + dk, half + dk + s2)
                    neighbor_vals[:, :, :, local_idx] = anat_padded[
                        ii:ii + batch_size_actual, jj, kk
                    ]

                diffs = torch.abs(neighbor_vals - center_vals)

                combined_vals = torch.cat([best_vals, diffs], dim=-1)
                combined_idx = torch.cat(
                    [
                        best_idx.to(torch.int64),
                        offset_indices[chunk_start:chunk_end]
                        .view(1, 1, 1, current_size)
                        .expand(batch_size_actual, s1, s2, current_size),
                    ],
                    dim=-1,
                )

                top_vals, top_pos = torch.topk(
                    combined_vals, k_keep, dim=-1, largest=False, sorted=True
                )
                best_vals = top_vals
                best_idx = torch.gather(combined_idx, -1, top_pos).to(torch.int16)

            mask_indices[i_start:i_end, :, :, :] = best_idx

        return mask_indices

    def neighbourhood_kernel(self, x, image, num_neighbours, sigma_anat, sigma_dist,
                            sigma_emission, normalize_kernel, use_mask, recalc_mask,
                            distance_weighting, hybrid):
        """
        Apply the kernel operator forward pass (GPU implementation).

        This is the main entry point called by direct().
        """
        if self.device.type == 'cpu':
            if self._cpu_operator is None:
                raise RuntimeError(
                    "Numba backend required for TorchKernelOperator CPU fallback."
                )
            self._cpu_operator.freeze_emission_kernel = self.freeze_emission_kernel
            self._cpu_operator.frozen_emission_kernel = self.frozen_emission_kernel
            result = self._cpu_operator.neighbourhood_kernel(
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
            )
            self._normalisation_map = self._cpu_operator._normalisation_map
            self._normalisation_map_gpu = None
            return result

        # Get arrays and convert to torch tensors
        arr = get_array(image)
        x_arr = get_array(x)

        # Convert to torch tensors on GPU
        x_tensor = torch.from_numpy(np.ascontiguousarray(x_arr, dtype=self.numpy_dtype)).to(self.device)
        anat_tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=self.numpy_dtype)).to(self.device)

        # Handle hybrid mode
        if hybrid:
            ref_arr = self._update_hybrid_reference(x_arr)
            ref_tensor = torch.from_numpy(np.ascontiguousarray(ref_arr, dtype=self.numpy_dtype)).to(self.device)
        else:
            ref_tensor = x_tensor

        # Normalization array
        if normalize_kernel:
            norm_dtype = torch.float16 if self.torch_dtype == torch.float32 else self.torch_dtype
            norm_tensor = torch.zeros_like(anat_tensor, dtype=norm_dtype)
        else:
            norm_tensor = torch.zeros((1, 1, 1), dtype=self.torch_dtype, device=self.device)

        n = num_neighbours

        # Use sparse or dense kernel based on masking
        if use_mask:
            mask_cpu = self._get_mask_cpu(force_recompute=recalc_mask, dtype=torch.int16)
            res_tensor = self._torch_kernel_forward_sparse(
                x_tensor,
                ref_tensor,
                anat_tensor,
                mask_cpu,
                norm_tensor,
                n,
                sigma_anat,
                sigma_dist,
                sigma_emission,
                normalize_kernel,
                hybrid,
                distance_weighting,
            )
        else:
            res_tensor = self._torch_kernel_forward_dense(
                x_tensor,
                ref_tensor,
                anat_tensor,
                norm_tensor,
                n,
                sigma_anat,
                sigma_dist,
                sigma_emission,
                normalize_kernel,
                hybrid,
                distance_weighting,
            )

        # Convert back to numpy
        res = res_tensor.cpu().numpy()

        # Store normalization map
        if normalize_kernel:
            norm_cpu = norm_tensor.cpu()
            self._normalisation_map = norm_cpu.numpy()
            self._normalisation_map_cpu = norm_cpu
            self._normalisation_map_gpu = None
        else:
            self._normalisation_map = None
            self._normalisation_map_gpu = None
            self._normalisation_map_cpu = None

        # Fill output
        out = image.clone()
        out.fill(res)

        if use_mask:
            self._mask_gpu = None

        return out

    def _torch_kernel_forward_sparse(self, x_arr, ref_arr, anat_arr, mask_cpu,
                                     norm_arr, n, sigma_anat, sigma_dist,
                                     sigma_emission, normalize, hybrid, distance_weighting):
        """
        Forward kernel computing anatomical weights on-the-fly (sparse version).

        Parameters
        ----------
        x_arr : torch.Tensor
            Input emission data (s0, s1, s2)
        ref_arr : torch.Tensor
            Reference emission data for hybrid mode (s0, s1, s2)
        anat_arr : torch.Tensor
            Anatomical data (s0, s1, s2)
        mask_cpu : torch.Tensor
            Sparse mask indices stored on CPU (s0, s1, s2, k) as int16
        norm_arr : torch.Tensor
            Normalization array (s0, s1, s2) or (1, 1, 1)
        n : int
            Neighborhood size
        sigma_anat : float
            Anatomical similarity sigma
        sigma_dist : float
            Distance weighting sigma
        sigma_emission : float
            Emission similarity sigma
        normalize : bool
            Whether to normalize kernel
        hybrid : bool
            Whether to use hybrid mode (HKRL)
        distance_weighting : bool
            Whether to apply geometric distance weighting

        Returns
        -------
        torch.Tensor
            Filtered output (s0, s1, s2)
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        k = mask_cpu.shape[3]
        sig2_an = 2.0 * sigma_anat * sigma_anat
        sig2_em = 2.0 * sigma_emission * sigma_emission
        dist2_an = 2.0 * sigma_dist * sigma_dist
        use_anat = sigma_anat > 0
        use_em = hybrid and sigma_emission > 0
        use_dist = distance_weighting and sigma_dist > 0

        out = torch.empty_like(x_arr)

        # Pad arrays for boundary handling
        x_padded = F.pad(x_arr.unsqueeze(0).unsqueeze(0),
                        (half, half, half, half, half, half),
                        mode='reflect').squeeze()
        ref_padded = F.pad(ref_arr.unsqueeze(0).unsqueeze(0),
                          (half, half, half, half, half, half),
                          mode='reflect').squeeze()
        anat_padded = F.pad(anat_arr.unsqueeze(0).unsqueeze(0),
                            (half, half, half, half, half, half),
                            mode='reflect').squeeze()

        # Process in batches to manage memory
        mask_elem_bytes = mask_cpu.element_size()
        batch_size = self._select_batch_size(s0, s1, s2, k, mask_elem_bytes)
        j_base = torch.arange(s1, device=self.device, dtype=torch.int64).view(1, -1, 1) + half
        k_base = torch.arange(s2, device=self.device, dtype=torch.int64).view(1, 1, -1) + half
        sumv_buf = torch.empty((batch_size, s1, s2), dtype=self.torch_dtype, device=self.device)
        wsum_buf = torch.empty((batch_size, s1, s2), dtype=self.torch_dtype, device=self.device)

        for i_start in range(0, s0, batch_size):
            i_end = min(i_start + batch_size, s0)
            batch_size_actual = i_end - i_start
            center_anat = anat_arr[i_start:i_end, :, :]
            mask_chunk = mask_cpu[i_start:i_end].to(self.device, non_blocking=True)
            i_base = (
                torch.arange(i_start, i_end, device=self.device, dtype=torch.int64)
                .view(-1, 1, 1)
                + half
            )

            # Get center reference values if hybrid
            if hybrid:
                c_ref = ref_arr[i_start:i_end, :, :]  # (batch, s1, s2)

            # Accumulate weighted sum and weight sum
            sumv = sumv_buf[:batch_size_actual]
            wsum = wsum_buf[:batch_size_actual]
            sumv.zero_()
            wsum.zero_()

            # Iterate over k neighbors
            for k_idx in range(k):
                # Get flat indices for this neighbor
                flat_indices = mask_chunk[:, :, :, k_idx].to(torch.int64)

                # Convert flat index to offset (di, dj, dk)
                dk = (flat_indices % n) - half
                dj = ((flat_indices // n) % n) - half
                di = (flat_indices // (n * n)) - half

                # Create index grids for gathering
                i_indices = i_base + di
                j_indices = j_base + dj
                k_indices = k_base + dk

                # Gather neighbor values
                x_neighbor = x_padded[i_indices, j_indices, k_indices]
                neighbor_anat = anat_padded[i_indices, j_indices, k_indices]

                # Compute anatomical weight
                diff_an = neighbor_anat - center_anat
                if use_anat:
                    w = torch.exp(-diff_an * diff_an / sig2_an)
                else:
                    w = torch.ones_like(diff_an)

                if use_dist:
                    di_f = di.to(self.torch_dtype)
                    dj_f = dj.to(self.torch_dtype)
                    dk_f = dk.to(self.torch_dtype)
                    dist_sq = di_f * di_f + dj_f * dj_f + dk_f * dk_f
                    w = w * torch.exp(-dist_sq / dist2_an)

                # Apply emission weight if hybrid
                if use_em:
                    ref_neighbor = ref_padded[i_indices, j_indices, k_indices]
                    diff_em = ref_neighbor - c_ref
                    w = w * torch.exp(-diff_em * diff_em / sig2_em)

                # Accumulate
                sumv += x_neighbor * w
                wsum += w

            # Normalize if requested
            if normalize:
                valid = wsum > 1e-12
                sumv = torch.where(valid, sumv / wsum, sumv)
                norm_update = torch.where(valid, wsum, torch.ones_like(wsum))
                norm_arr[i_start:i_end, :, :] = norm_update.to(norm_arr.dtype)

            out[i_start:i_end, :, :] = sumv

        return out

    def _torch_kernel_forward_dense(self, x_arr, ref_arr, anat_arr, norm_arr, n,
                                    sigma_anat, sigma_dist, sigma_emission,
                                    normalize, hybrid, distance_weighting):
        """
        Forward kernel computing anatomical weights on-the-fly (dense version, all n³ neighbors).

        This is similar to sparse but iterates over all n³ neighbors.
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        total = n ** 3
        sig2_an = 2.0 * sigma_anat * sigma_anat
        sig2_em = 2.0 * sigma_emission * sigma_emission
        dist2_an = 2.0 * sigma_dist * sigma_dist
        use_anat = sigma_anat > 0
        use_em = hybrid and sigma_emission > 0
        use_dist = distance_weighting and sigma_dist > 0

        out = torch.empty_like(x_arr)

        # Pad arrays
        x_padded = F.pad(x_arr.unsqueeze(0).unsqueeze(0),
                        (half, half, half, half, half, half),
                        mode='reflect').squeeze()
        ref_padded = F.pad(ref_arr.unsqueeze(0).unsqueeze(0),
                          (half, half, half, half, half, half),
                          mode='reflect').squeeze()
        anat_padded = F.pad(anat_arr.unsqueeze(0).unsqueeze(0),
                            (half, half, half, half, half, half),
                            mode='reflect').squeeze()

        # Build offsets
        offsets = []
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                for dk in range(-half, half + 1):
                    offsets.append((di, dj, dk))

        # Process in batches
        batch_size = min(s0, self.max_gpu_batch_slices)
        sumv_buf = torch.empty((batch_size, s1, s2), dtype=self.torch_dtype, device=self.device)
        wsum_buf = torch.empty((batch_size, s1, s2), dtype=self.torch_dtype, device=self.device)

        for i_start in range(0, s0, batch_size):
            i_end = min(i_start + batch_size, s0)
            batch_size_actual = i_end - i_start

            center_anat = anat_arr[i_start:i_end, :, :]

            if hybrid:
                c_ref = ref_arr[i_start:i_end, :, :]

            sumv = sumv_buf[:batch_size_actual]
            wsum = wsum_buf[:batch_size_actual]
            sumv.zero_()
            wsum.zero_()

            for idx, (di, dj, dk) in enumerate(offsets):
                ii = slice(i_start + half + di, i_end + half + di)
                jj = slice(half + dj, half + dj + s1)
                kk = slice(half + dk, half + dk + s2)

                x_neighbor = x_padded[ii, jj, kk]
                neighbor_anat = anat_padded[ii, jj, kk]
                diff_an = neighbor_anat - center_anat
                if use_anat:
                    w = torch.exp(-diff_an * diff_an / sig2_an)
                else:
                    w = torch.ones_like(diff_an)

                if use_dist:
                    dist_sq = (di * di + dj * dj + dk * dk)
                    dist_weight = torch.tensor(
                        np.exp(-dist_sq / dist2_an), dtype=self.torch_dtype, device=self.device
                    )
                    w = w * dist_weight

                if use_em:
                    ref_neighbor = ref_padded[ii, jj, kk]
                    diff_em = ref_neighbor - c_ref
                    w = w * torch.exp(-diff_em * diff_em / sig2_em)

                sumv += x_neighbor * w
                wsum += w

            if normalize:
                valid = wsum > 1e-12
                sumv = torch.where(valid, sumv / wsum, sumv)
                norm_update = torch.where(valid, wsum, torch.ones_like(wsum))
                norm_arr[i_start:i_end, :, :] = norm_update.to(norm_arr.dtype)

            out[i_start:i_end, :, :] = sumv

        return out

    def adjoint(self, x, out=None):
        """
        Adjoint operator (GPU implementation).
        """
        if self.device.type == 'cpu':
            if self._cpu_operator is None:
                raise RuntimeError(
                    "Numba backend required for TorchKernelOperator CPU fallback."
                )
            self._cpu_operator.freeze_emission_kernel = self.freeze_emission_kernel
            self._cpu_operator.frozen_emission_kernel = self.frozen_emission_kernel
            res = self._cpu_operator.adjoint(x, out=out)
            self._normalisation_map = self._cpu_operator._normalisation_map
            self._normalisation_map_gpu = None
            return res

        x_arr = get_array(x)
        p = self.parameters

        # Convert to torch tensors
        x_tensor = torch.from_numpy(np.ascontiguousarray(x_arr, dtype=self.numpy_dtype)).to(self.device)

        # Get normalization map
        if p["normalize_kernel"]:
            if self._normalisation_map_gpu is None:
                if self._normalisation_map_cpu is not None:
                    self._normalisation_map_gpu = self._normalisation_map_cpu.to(
                        self.device, dtype=self.torch_dtype
                    )
                else:
                    raise RuntimeError(
                        "Normalization map has not been initialised. "
                        "Call direct() before adjoint() when using normalize_kernel=True."
                    )
            norm_tensor = self._normalisation_map_gpu
        else:
            norm_tensor = torch.zeros((1, 1, 1), dtype=self.torch_dtype, device=self.device)

        n = p["num_neighbours"]

        if self.anatomical_image is None:
            raise RuntimeError("An anatomical image must be set before calling adjoint().")
        anat_arr = np.ascontiguousarray(get_array(self.anatomical_image), dtype=self.numpy_dtype)
        anat_tensor = torch.from_numpy(anat_arr).to(self.device)

        # Get reference for hybrid mode
        if p["hybrid"]:
            ref_arr = self._get_hybrid_reference()
            ref_tensor = torch.from_numpy(np.ascontiguousarray(ref_arr, dtype=self.numpy_dtype)).to(self.device)
        else:
            ref_tensor = x_tensor

        # Use sparse or dense adjoint based on masking
        if p["use_mask"]:
            mask_cpu = self._get_mask_cpu(dtype=torch.int16)
            res_tensor = self._torch_adjoint_sparse(
                x_tensor,
                ref_tensor,
                anat_tensor,
                mask_cpu,
                norm_tensor,
                n,
                p["sigma_anat"],
                p["sigma_dist"],
                p["sigma_emission"],
                p["hybrid"],
                p["distance_weighting"],
            )
        else:
            res_tensor = self._torch_adjoint_dense(
                x_tensor,
                ref_tensor,
                anat_tensor,
                norm_tensor,
                n,
                p["sigma_anat"],
                p["sigma_dist"],
                p["sigma_emission"],
                p["hybrid"],
                p["distance_weighting"],
            )

        # Convert back to numpy
        res = res_tensor.cpu().numpy()

        # Fill output
        img = x.clone()
        img.fill(res)
        if p["use_mask"]:
            self._mask_gpu = None
        if p["normalize_kernel"]:
            self._normalisation_map_gpu = None

        if out is None:
            return img
        out.fill(res)
        return out

    def _torch_adjoint_sparse(self, x_arr, ref_arr, anat_arr, mask_cpu,
                              norm_arr, n, sigma_anat, sigma_dist,
                              sigma_emission, hybrid, distance_weighting):
        """
        Adjoint kernel computing anatomical weights on-the-fly (sparse version).
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        k = mask_cpu.shape[3]
        sig2_an = 2.0 * sigma_anat * sigma_anat
        sig2_em = 2.0 * sigma_emission * sigma_emission
        dist2_an = 2.0 * sigma_dist * sigma_dist
        use_anat = sigma_anat > 0
        use_em = hybrid and sigma_emission > 0
        use_dist = distance_weighting and sigma_dist > 0

        out = torch.zeros_like(x_arr)

        # Pad reference for boundary handling
        ref_padded = F.pad(ref_arr.unsqueeze(0).unsqueeze(0),
                          (half, half, half, half, half, half),
                          mode='reflect').squeeze()
        anat_padded = F.pad(anat_arr.unsqueeze(0).unsqueeze(0),
                            (half, half, half, half, half, half),
                            mode='reflect').squeeze()

        # Process in batches
        mask_elem_bytes = mask_cpu.element_size()
        batch_size = self._select_batch_size(s0, s1, s2, k, mask_elem_bytes)
        j_base = torch.arange(s1, device=self.device, dtype=torch.int64).view(1, -1, 1)
        k_base = torch.arange(s2, device=self.device, dtype=torch.int64).view(1, 1, -1)
        j_pad = j_base + half
        k_pad = k_base + half

        for i_start in range(0, s0, batch_size):
            i_end = min(i_start + batch_size, s0)

            # Get values for this batch
            val = x_arr[i_start:i_end, :, :]

            # Apply normalization if needed
            if norm_arr.shape[0] > 1:
                norm = norm_arr[i_start:i_end, :, :]
                val = torch.where(norm > 1e-12, val / norm, torch.zeros_like(val))

            center_anat = anat_arr[i_start:i_end, :, :]
            mask_chunk = mask_cpu[i_start:i_end].to(self.device, non_blocking=True)
            i_center = (
                torch.arange(i_start, i_end, device=self.device, dtype=torch.int64)
                .view(-1, 1, 1)
            )
            i_pad = i_center + half

            if hybrid:
                c_ref = ref_arr[i_start:i_end, :, :]

            # Iterate over k neighbors
            for k_idx in range(k):
                # Get flat indices
                flat_indices = mask_chunk[:, :, :, k_idx].to(torch.int64)

                # Convert flat index to offset
                dk = (flat_indices % n) - half
                dj = ((flat_indices // n) % n) - half
                di = (flat_indices // (n * n)) - half

                # Gather anatomical neighbors
                i_indices = i_pad + di
                j_indices = j_pad + dj
                k_indices = k_pad + dk

                neighbor_anat = anat_padded[i_indices, j_indices, k_indices]
                diff_an = neighbor_anat - center_anat
                if use_anat:
                    w = torch.exp(-diff_an * diff_an / sig2_an)
                else:
                    w = torch.ones_like(diff_an)

                if use_dist:
                    di_f = di.to(self.torch_dtype)
                    dj_f = dj.to(self.torch_dtype)
                    dk_f = dk.to(self.torch_dtype)
                    dist_sq = di_f * di_f + dj_f * dj_f + dk_f * dk_f
                    w = w * torch.exp(-dist_sq / dist2_an)

                # Apply emission weight if hybrid
                if use_em:
                    ref_neighbor = ref_padded[i_indices, j_indices, k_indices]
                    diff_em = ref_neighbor - c_ref
                    w = w * torch.exp(-diff_em * diff_em / sig2_em)

                # Scatter accumulation using index_add_
                # We need to add val * w to the neighbor positions
                contrib = val * w

                # Create target indices (with proper boundary handling)
                i_target = i_center + di
                j_target = j_base + dj
                k_target = k_base + dk

                # Clamp to handle boundaries (reflection already handled in weight computation)
                i_target = torch.clamp(i_target, 0, s0 - 1)
                j_target = torch.clamp(j_target, 0, s1 - 1)
                k_target = torch.clamp(k_target, 0, s2 - 1)

                # Atomic add to output
                out[i_target, j_target, k_target] += contrib

        return out

    def _torch_adjoint_dense(self, x_arr, ref_arr, anat_arr, norm_arr,
                             n, sigma_anat, sigma_dist, sigma_emission,
                             hybrid, distance_weighting):
        """
        Adjoint kernel computing anatomical weights on-the-fly (dense version).
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        total = n ** 3
        sig2_an = 2.0 * sigma_anat * sigma_anat
        sig2_em = 2.0 * sigma_emission * sigma_emission
        dist2_an = 2.0 * sigma_dist * sigma_dist
        use_anat = sigma_anat > 0
        use_em = hybrid and sigma_emission > 0
        use_dist = distance_weighting and sigma_dist > 0

        out = torch.zeros_like(x_arr)

        # Pad reference
        ref_padded = F.pad(ref_arr.unsqueeze(0).unsqueeze(0),
                          (half, half, half, half, half, half),
                          mode='reflect').squeeze()
        anat_padded = F.pad(anat_arr.unsqueeze(0).unsqueeze(0),
                            (half, half, half, half, half, half),
                            mode='reflect').squeeze()

        # Build offsets
        offsets = []
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                for dk in range(-half, half + 1):
                    offsets.append((di, dj, dk))

        # Process in batches
        batch_size = min(s0, self.max_gpu_batch_slices)

        for i_start in range(0, s0, batch_size):
            i_end = min(i_start + batch_size, s0)

            val = x_arr[i_start:i_end, :, :]

            if norm_arr.shape[0] > 1:
                norm = norm_arr[i_start:i_end, :, :]
                val = torch.where(norm > 1e-12, val / norm, torch.zeros_like(val))

            center_anat = anat_arr[i_start:i_end, :, :]

            if hybrid:
                c_ref = ref_arr[i_start:i_end, :, :]

            for idx, (di, dj, dk) in enumerate(offsets):
                ii = slice(i_start + half + di, i_end + half + di)
                jj = slice(half + dj, half + dj + s1)
                kk = slice(half + dk, half + dk + s2)

                neighbor_anat = anat_padded[ii, jj, kk]
                diff_an = neighbor_anat - center_anat
                if use_anat:
                    w = torch.exp(-diff_an * diff_an / sig2_an)
                else:
                    w = torch.ones_like(diff_an)

                if use_dist:
                    dist_sq = (di * di + dj * dj + dk * dk)
                    dist_weight = torch.tensor(
                        np.exp(-dist_sq / dist2_an), dtype=self.torch_dtype, device=self.device
                    )
                    w = w * dist_weight

                if use_em:
                    ref_neighbor = ref_padded[ii, jj, kk]
                    diff_em = ref_neighbor - c_ref
                    w = w * torch.exp(-diff_em * diff_em / sig2_em)

                contrib = val * w

                # Add to neighbor positions
                i_target = slice(max(0, i_start + di), min(s0, i_end + di))
                j_target = slice(max(0, dj), min(s1, s1 + dj))
                k_target = slice(max(0, dk), min(s2, s2 + dk))

                out[i_target, j_target, k_target] += contrib

        return out

    def clear_gpu(self, release_cached_tensors=True):
        """
        Release GPU memory cache and optionally drop cached tensors.

        Parameters
        ----------
        release_cached_tensors : bool, optional
            If True (default), clears cached tensors such as masks and
            normalization maps so memory can be reclaimed by the CUDA allocator.
        """
        if release_cached_tensors:
            self._mask_gpu = None
            self._mask_cpu = None
            self._normalisation_map_gpu = None
            self._normalisation_map_cpu = None

        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """Clean up GPU memory on deletion."""
        # Safe cleanup: torch might be None during interpreter shutdown
        if TORCH_AVAILABLE and torch is not None:
            try:
                self.clear_gpu()
            except Exception:
                pass  # Ignore errors during cleanup


def get_torch_kernel_operator(domain_geometry, **kwargs):
    """
    Factory function to create a TorchKernelOperator.

    Parameters
    ----------
    domain_geometry : ImageGeometry
        Domain geometry
    **kwargs
        Parameters passed to TorchKernelOperator

    Returns
    -------
    TorchKernelOperator
        GPU-accelerated kernel operator
    """
    return TorchKernelOperator(domain_geometry, **kwargs)
