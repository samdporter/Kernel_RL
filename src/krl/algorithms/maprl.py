import logging
import warnings

import numpy as np
from cil.optimisation.algorithms import Algorithm

LOGGER = logging.getLogger(__name__)


class MAPRL(Algorithm):
    """MAPRL algorithm with EM-style updates and an Armijo line search.

    Parameters
    ----------
    eps_preconditioner : float, optional
        Deprecated and ignored: it has no effect on any update or line-search
        equation. Stabilisation, if needed, belongs inside the supplied
        preconditioner; ``eps`` is what scales the unpreconditioned EM update.
        Omitting it keeps the legacy stored attribute value ``1e-6``;
        supplying any value raises a ``FutureWarning``.
    """

    def __init__(
        self,
        initial_estimate,
        data_fidelity,
        prior,
        step_size=5,
        relaxation_eta=0.01,
        eps=0,
        initial_line_search=True,
        armijo_beta=0.75,
        armijo_sigma=1e-1,
        armijo_max_iter=20,
        armijo_iterations=25,
        armijo_update_initial=0,
        armijo_update_interval=10,
        preconditioner=None,
        preconditioner_update_initial=25,
        preconditioner_update_interval=5,
        eps_preconditioner=None,
        **kwargs,
    ):
        if not np.isfinite(relaxation_eta) or relaxation_eta < 0:
            raise ValueError("relaxation_eta must be a finite non-negative number.")
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be a finite positive number.")
        self.initial_estimate = initial_estimate
        self.initial_step_size = step_size
        self.relaxation_eta = relaxation_eta
        self.eps = eps
        self.initial_line_search = initial_line_search
        self.armijo_beta = armijo_beta
        self.armijo_sigma = armijo_sigma
        self.armijo_max_iter = armijo_max_iter
        self.armijo_iterations = armijo_iterations
        self.armijo_update_initial = armijo_update_initial
        self.armijo_update_interval = armijo_update_interval
        self._current_step_size = step_size  # Track the current step size found by Armijo
        # Counts actual updates independently of CIL's reporting label so that
        # scheduling remains correct for any update_objective_interval.
        self._update_count = 0

        self.x = initial_estimate.clone()
        self.data_fidelity = data_fidelity
        self.prior = prior

        # Preconditioner parameters
        self.preconditioner = preconditioner  # Can be callable or ImageData
        self.preconditioner_update_initial = preconditioner_update_initial
        self.preconditioner_update_interval = preconditioner_update_interval
        if eps_preconditioner is not None:
            warnings.warn(
                "eps_preconditioner is deprecated and has no effect. Apply "
                "stabilisation inside the supplied preconditioner. The eps "
                "parameter affects only the unpreconditioned EM scaling.",
                FutureWarning,
                stacklevel=2,
            )
        # Retained for API compatibility; see the class docstring.
        self.eps_preconditioner = (
            1e-6 if eps_preconditioner is None else eps_preconditioner
        )
        self._preconditioner_image = None

        super(MAPRL, self).__init__(**kwargs)
        self.configured = True

        if self.initial_line_search:
            if self.preconditioner is not None:
                if callable(self.preconditioner):
                    self._preconditioner_image = self.preconditioner(self.x)
                else:
                    self._preconditioner_image = self.preconditioner
            LOGGER.info(
                "MAPRL Armijo: starting search with suggested step %.4g",
                self.initial_step_size,
            )
            step, loss = self._armijo_initial_step(self.initial_step_size)
            self.initial_step_size = step
            self._current_step_size = step
            LOGGER.info(
                "MAPRL Armijo: accepted step %.4g (objective %.6g)",
                step,
                loss,
            )

    def _armijo_initial_block(self):
        """Number of contiguous iterations that should always run Armijo."""
        if self.armijo_update_initial > 0:
            if self.armijo_iterations > 0:
                return min(self.armijo_update_initial, self.armijo_iterations)
            return self.armijo_update_initial
        return max(self.armijo_iterations, 0)

    def step_size(self):
        initial_block = self._armijo_initial_block()
        if initial_block > 0 and self._update_count <= initial_block:
            return self._current_step_size
        baseline_step = self.initial_step_size or self._current_step_size
        decay_iter = max(self._update_count, 1)
        denom = 1 + self.relaxation_eta * decay_iter
        if denom == 0:
            return baseline_step
        return baseline_step / denom

    def update(self):
        self._update_count += 1

        # Update preconditioner if needed
        if self.preconditioner is not None:
            # Always initialise on first use, then during the initial window,
            # then periodically if a positive interval is configured.
            should_update = (
                self._update_count == 1
                or self._update_count <= self.preconditioner_update_initial
            )
            if (
                self.preconditioner_update_interval > 0
                and self._update_count % self.preconditioner_update_interval == 0
            ):
                should_update = True

            if should_update:
                if callable(self.preconditioner):
                    self._preconditioner_image = self.preconditioner(self.x)
                else:
                    self._preconditioner_image = self.preconditioner

                # Log preconditioner statistics
                if hasattr(self._preconditioner_image, 'as_array'):
                    prec_arr = self._preconditioner_image.as_array()
                    LOGGER.info(
                        "MAPRL: updated preconditioner at update %d (min=%.3e, max=%.3e, mean=%.3e)",
                        self._update_count,
                        float(np.min(prec_arr)),
                        float(np.max(prec_arr)),
                        float(np.mean(prec_arr)),
                    )
                else:
                    LOGGER.info("MAPRL: updated preconditioner at update %d", self._update_count)

        # Determine if we should perform Armijo line search at this update
        initial_block = self._armijo_initial_block()
        should_armijo = initial_block > 0 and self._update_count <= initial_block
        if (
            not should_armijo
            and self.armijo_iterations > 0
            and self._update_count <= self.armijo_iterations
            and self.armijo_update_interval > 0
            and self._update_count % self.armijo_update_interval == 0
        ):
            # After the initial block, fall back to periodic Armijo updates
            should_armijo = True

        # Perform Armijo line search if scheduled
        if should_armijo:
            LOGGER.info(
                "MAPRL Armijo: performing line search at update %d (starting from step %.4g)",
                self._update_count,
                self.initial_step_size,
            )
            # Always restart from initial_step_size for Armijo search
            step, loss = self._armijo_step(self.initial_step_size)
            # Update current step size to use the found step
            self._current_step_size = step
            LOGGER.info(
                "MAPRL Armijo: update %d accepted step %.4g (objective %.6g)",
                self._update_count,
                step,
                loss,
            )
        else:
            grad = self.data_fidelity.gradient(self.x) + self.prior.gradient(self.x)

            # Apply preconditioner if available
            if self._preconditioner_image is not None:
                # the preconditioner replaces the (x + eps) EM scaling
                precond_grad = grad * self._preconditioner_image
                self.x = self.x - precond_grad * self.step_size()
            else:
                # Standard EM-style update without prior preconditioner
                self.x = self.x - (self.x + self.eps) * grad * self.step_size()

            with np.errstate(invalid="ignore"):
                self.x.maximum(0, out=self.x)

    def update_objective(self):
        self.loss.append(self.data_fidelity(self.x) + self.prior(self.x))

    def _armijo_initial_step(self, suggested_step):
        current = self.x.clone()
        grad = self.data_fidelity.gradient(current) + self.prior.gradient(current)

        # Apply preconditioner if available
        if self._preconditioner_image is not None:
            direction = grad * self._preconditioner_image
        else:
            direction = (current + self.eps) * grad

        slope = self._inner_product(direction, grad)
        if not np.isfinite(slope) or slope <= 0:
            return suggested_step, self._evaluate_loss(current)

        reference_loss = self._evaluate_loss(current)
        step = float(suggested_step)
        last_step = step
        last_loss = reference_loss
        best_step = None
        best_loss = None

        for iteration in range(1, self.armijo_max_iter + 1):
            candidate = current - direction * step
            with np.errstate(invalid="ignore"):
                candidate.maximum(0, out=candidate)
            candidate_loss = self._evaluate_loss(candidate)
            if candidate_loss <= reference_loss - self.armijo_sigma * step * slope:
                LOGGER.debug(
                    "MAPRL Armijo: iter %d accepted %.4g (objective %.6g)",
                    iteration,
                    step,
                    candidate_loss,
                )
                return step, candidate_loss
            last_step = step
            last_loss = candidate_loss
            if np.isfinite(candidate_loss) and (best_loss is None or candidate_loss < best_loss):
                best_loss = candidate_loss
                best_step = step
            step *= self.armijo_beta
            LOGGER.debug(
                "MAPRL Armijo: iter %d rejected, new candidate step %.4g (objective %.6g)",
                iteration,
                step,
                candidate_loss,
            )
            if step <= 0:
                break

        LOGGER.warning(
            "MAPRL Armijo: fallback to last tested step %.4g (objective %.6g)",
            last_step,
            last_loss,
        )
        # Every candidate failed the sufficient-decrease test, so do not
        # return a rejected step as accepted. If some candidate still did not
        # worsen the reference loss, offer the best one; otherwise keep the
        # suggested step so the constructor never seeds an uphill step.
        if best_loss is not None and best_loss <= reference_loss:
            return best_step, best_loss
        return float(suggested_step), reference_loss

    def _armijo_step(self, suggested_step):
        """Perform Armijo line search and update self.x with the result."""
        current = self.x.clone()
        grad = self.data_fidelity.gradient(current) + self.prior.gradient(current)

        # Apply preconditioner if available
        if self._preconditioner_image is not None:
            direction = grad * self._preconditioner_image
        else:
            direction = (current + self.eps) * grad

        slope = self._inner_product(direction, grad)
        if not np.isfinite(slope) or slope <= 0:
            # If slope is not valid, just do a regular update
            self.x = current - direction * suggested_step
            with np.errstate(invalid="ignore"):
                self.x.maximum(0, out=self.x)
            return suggested_step, self._evaluate_loss(self.x)

        reference_loss = self._evaluate_loss(current)
        step = float(suggested_step)
        last_step = step
        last_loss = reference_loss
        best_step = None
        best_loss = None

        for iteration in range(1, self.armijo_max_iter + 1):
            candidate = current - direction * step
            with np.errstate(invalid="ignore"):
                candidate.maximum(0, out=candidate)
            candidate_loss = self._evaluate_loss(candidate)
            if candidate_loss <= reference_loss - self.armijo_sigma * step * slope:
                LOGGER.debug(
                    "MAPRL Armijo: iter %d accepted %.4g (objective %.6g)",
                    iteration,
                    step,
                    candidate_loss,
                )
                self.x = candidate
                return step, candidate_loss
            last_step = step
            last_loss = candidate_loss
            if np.isfinite(candidate_loss) and (best_loss is None or candidate_loss < best_loss):
                best_loss = candidate_loss
                best_step = step
            step *= self.armijo_beta
            LOGGER.debug(
                "MAPRL Armijo: iter %d rejected, new candidate step %.4g (objective %.6g)",
                iteration,
                step,
                candidate_loss,
            )
            if step <= 0:
                break

        LOGGER.warning(
            "MAPRL Armijo: fallback to last tested step %.4g (objective %.6g)",
            last_step,
            last_loss,
        )
        # An Armijo rejection means no candidate achieved the sufficient
        # decrease test; never apply one whose loss is worse than the
        # reference. Apply the best candidate only if it does not worsen the
        # reference, otherwise leave self.x unchanged.
        if best_loss is not None and best_loss <= reference_loss:
            self.x = current - direction * best_step
            with np.errstate(invalid="ignore"):
                self.x.maximum(0, out=self.x)
            return best_step, best_loss
        return last_step, reference_loss

    def _evaluate_loss(self, x):
        return float(self.data_fidelity(x) + self.prior(x))

    @staticmethod
    def _inner_product(a, b):
        if hasattr(a, "containers") and hasattr(b, "containers"):
            return sum(MAPRL._inner_product(ac, bc) for ac, bc in zip(a.containers, b.containers))
        if hasattr(a, "as_array"):
            arr_a = a.as_array()
        else:
            arr_a = np.asarray(a)
        if hasattr(b, "as_array"):
            arr_b = b.as_array()
        else:
            arr_b = np.asarray(b)
        return float(np.sum(arr_a * arr_b))
