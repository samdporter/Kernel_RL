"""Richardson-Lucy deconvolution algorithm compatible with CIL's Algorithm interface."""

import numpy as np

try:
    from cil.optimisation.algorithms import Algorithm
    import cil.optimisation.operators as op
except ImportError:
    # Fallback for testing or when CIL is not available
    class Algorithm:
        def __init__(self, **kwargs):
            self.iteration = 0
            self.loss = []
            self.configured = False

        def run(self, iterations, verbose=0, callbacks=None):
            for i in range(iterations):
                self.iteration = i + 1
                self.update()
                if callbacks:
                    for cb in callbacks:
                        cb(self)


class RichardsonLucy(Algorithm):
    """
    Richardson-Lucy (RL) deconvolution algorithm.

    This implements the classic RL algorithm for image deconvolution, with optional
    support for kernel operators (KRL/HKRL) and freezing.

    The RL algorithm iteratively refines an estimate x by:
        x = x * (A^T (y / Ax)) / sensitivity

    where A is the forward operator (blur, or blur ∘ kernel for KRL),
    y is the observed data, and sensitivity is A^T(1).

    Parameters
    ----------
    initial_estimate : ImageData
        Initial estimate (typically the observed blurred image)
    blurring_operator : LinearOperator
        Blurring operator (PSF)
    observed_data : ImageData
        Observed blurred data
    kernel_operator : LinearOperator, optional
        Kernel operator for anatomical guidance (KRL/HKRL). If provided,
        the forward operator becomes blur ∘ kernel.
    freeze_iteration : int, optional
        If > 0 and kernel_operator is provided, freeze the kernel operator
        at this iteration (default: 0, no freezing)
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-10)
    update_objective_interval : int, optional
        Compute objective function every N iterations (default: 1)

    Attributes
    ----------
    x : ImageData
        Current estimate (latent image for KRL, reconstructed image for standard RL)
    solution : ImageData
        Alias for x (CIL convention)
    objective : list
        Objective function values (KL divergence)

    Examples
    --------
    Standard RL:

    >>> rl = RichardsonLucy(
    ...     initial_estimate=observed,
    ...     blurring_operator=blur_op,
    ...     observed_data=observed
    ... )
    >>> rl.run(iterations=100, verbose=1)

    KRL with NRMSE callback:

    >>> krl = RichardsonLucy(
    ...     initial_estimate=observed,
    ...     blurring_operator=blur_op,
    ...     observed_data=observed,
    ...     kernel_operator=kernel_op,
    ...     freeze_iteration=5
    ... )
    >>> from krl.callbacks import NRMSECallback
    >>> callback = NRMSECallback(ground_truth, "nrmse.csv", kernel_operator=kernel_op)
    >>> krl.run(iterations=100, verbose=1, callbacks=[callback])
    """

    def __init__(
        self,
        initial_estimate,
        blurring_operator,
        observed_data,
        kernel_operator=None,
        freeze_iteration=0,
        epsilon=1e-10,
        update_objective_interval=1,
        **kwargs
    ):
        super(RichardsonLucy, self).__init__(**kwargs)

        self.observed_data = observed_data
        self.blurring_operator = blurring_operator
        self.kernel_operator = kernel_operator
        self.freeze_iteration = freeze_iteration
        self.epsilon = epsilon
        self.update_objective_interval = update_objective_interval

        # Initialize estimate
        self.x = initial_estimate.clone()

        # Build effective forward operator
        if kernel_operator is not None:
            # KRL/HKRL mode: compose blur and kernel
            try:
                self.forward_operator = op.CompositionOperator(blurring_operator, kernel_operator)
            except NameError:
                # Fallback if op is not available
                self.forward_operator = None
        else:
            # Standard RL mode
            self.forward_operator = blurring_operator

        # Initialize estimated blur first (required before adjoint when using normalize_kernel=True)
        if self.forward_operator is not None:
            self.est_blur = self.forward_operator.direct(self.x)
        else:
            self.est_blur = self.x.clone()

        # Compute sensitivity (normalization factor) after direct() call
        geometry = observed_data.geometry
        if self.forward_operator is not None:
            self.sensitivity = self.forward_operator.adjoint(geometry.allocate(value=1))
        else:
            self.sensitivity = geometry.allocate(value=1)

        self.configured = True

    @property
    def solution(self):
        """Return current solution (CIL convention)."""
        return self.x

    @solution.setter
    def solution(self, value):
        """Set current solution (CIL convention)."""
        self.x = value

    def update(self):
        """Perform one RL iteration."""
        # RL update: x *= (A^T (y / Ax)) / sensitivity
        ratio = self.observed_data / (self.est_blur + self.epsilon)
        correction = self.forward_operator.adjoint(ratio)
        self.x *= correction
        self.x /= (self.sensitivity + self.epsilon)

        # Enforce non-negativity
        with np.errstate(invalid="ignore"):
            self.x.maximum(0, out=self.x)

        # Update estimated blur for next iteration
        self.est_blur = self.forward_operator.direct(self.x)

        # For hybrid kernels, sensitivity must be recomputed each iteration until freezing
        # because the kernel operator changes with the emission reference
        if (
            self.kernel_operator is not None
            and getattr(self.kernel_operator, 'parameters', {}).get('hybrid', False)
            and not getattr(self.kernel_operator, 'freeze_emission_kernel', False)
        ):
            # Recompute sensitivity for changing hybrid kernel
            geometry = self.observed_data.geometry
            self.sensitivity = self.forward_operator.adjoint(geometry.allocate(value=1))

        # Handle freezing AFTER the iteration completes
        # Note: self.iteration is incremented AFTER this method returns (in CIL's __next__)
        # So when self.iteration == freeze_iteration, this is the freeze iteration itself
        # We freeze AFTER this iteration, so that iteration+1 uses this iteration's emission state
        if (
            self.freeze_iteration > 0
            and self.iteration == self.freeze_iteration
            and self.kernel_operator is not None
        ):
            self.kernel_operator.freeze_emission_kernel = True
            # Recompute sensitivity one final time with frozen kernel
            geometry = self.observed_data.geometry
            self.sensitivity = self.forward_operator.adjoint(geometry.allocate(value=1))

    def update_objective(self):
        """Compute and store the objective function (KL divergence)."""
        # KL divergence: sum(Ax - y * log(Ax + eps))
        obj = (self.est_blur - self.observed_data * (self.est_blur + self.epsilon).log()).sum()
        self.loss.append(obj)

    def get_output(self):
        """
        Get the final reconstructed image.

        For standard RL, this is just x.
        For KRL/HKRL, this applies the kernel operator to get the final reconstruction.

        Returns
        -------
        ImageData
            Reconstructed image
        """
        if self.kernel_operator is not None:
            return self.kernel_operator.direct(self.x)
        else:
            return self.x
