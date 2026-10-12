import warnings

import numpy as np
from cil.framework import BlockGeometry
from cil.optimisation.operators import LinearOperator


class DirectionalOperator(LinearOperator):
    """
    Projects out the component of the iterates along the anatomical gradient.

    Parameters
    ----------
    anatomical_gradient : BlockDataContainer
        Gradient components defining the direction field.
    gamma : float, optional
        Scale of the projection term.
    eta : float, optional
        Regularisation added in quadrature to the gradient magnitude. An
        explicit value should be positive: ``eta = 0`` makes the normalisation
        0/0 (NaN) wherever the anatomical gradient is zero, and a UserWarning
        is issued for any ``eta <= 0``. The default ``None`` derives a
        positive scale from the gradient percentiles and avoids the NaN risk.
    """

    def __init__(self, anatomical_gradient, gamma = 1, eta=None):

        self.anatomical_gradient = anatomical_gradient
        geometries = tuple(container.geometry for container in anatomical_gradient.containers)
        geometry = BlockGeometry(*geometries)
        self.tmp = self.anatomical_gradient.containers[0].clone()

        self.gamma = gamma

        _ALPHA = 0.05
        _P_LO  = 0.01
        _P_HI  = 0.99

        if eta is None:
            # Compute norm of gradient manually to avoid pnorm allocation issues
            # Stack all gradient components to compute L2 norm
            grad_arrays = [c.as_array() for c in self.anatomical_gradient.containers]
            grad_stack = np.stack(grad_arrays, axis=-1)
            den_arr = np.linalg.norm(grad_stack, axis=-1)

            # Percentile-based scale (robust to outliers and absolute CT scaling)
            q_lo, q_hi = np.quantile(
                den_arr.flatten(), np.array([_P_LO, _P_HI])
            )
            s = (q_hi - q_lo).item()
            if not np.isfinite(s) or s <= 0.0:
                # Fallback: use upper percentile itself; if still 0, use 1.0 to avoid zero eta
                s = q_hi.item()
                if not np.isfinite(s) or s <= 0.0:
                    s = 1.0
            eta = _ALPHA * s
        elif eta <= 0:
            warnings.warn(
                f"eta={eta!r} is not positive: eta=0 makes the direction "
                "field normalisation 0/0 (NaN) wherever the anatomical "
                "gradient is zero, and negative eta is not a meaningful "
                "scale. The default eta=None computes a positive scale and "
                "avoids the NaN risk.",
                UserWarning,
                stacklevel=2,
            )

        self.xi = self.anatomical_gradient/(self.anatomical_gradient.pnorm().power(2)+eta**2).sqrt()

        super(DirectionalOperator, self).__init__(domain_geometry=geometry,
                                       range_geometry=geometry,)

    def direct(self, x, out=None):

        result = x - self.gamma * self.xi * self.dot(self.xi, x)
        if out is None:
            return result
        out.fill(result)
        return out

    def adjoint(self, x, out=None):
        # This is the same as the direct operator
        return self.direct(x, out)

    def dot(self, x, y):
        """
        Inner product of two block containers.

        The returned container is an internal buffer reused by this operator:
        it is overwritten by every subsequent ``dot`` call, so clone it if the
        result must be kept.
        """
        self.tmp.fill(0)
        for el_x, el_y in zip(x.containers, y.containers):
            self.tmp += el_x * el_y
        return self.tmp
