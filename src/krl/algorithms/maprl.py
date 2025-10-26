import logging
import os
import numpy as np

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
level_name = os.environ.get("MAPRL_LOG_LEVEL")
if level_name is None:
    LOGGER.setLevel(logging.INFO)
else:
    try:
        LOGGER.setLevel(getattr(logging, level_name.upper()))
    except AttributeError:
        LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

try:
    from cil.optimisation.algorithms import Algorithm
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

class MAPRL(Algorithm):
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
        eps_preconditioner=1e-6,
        **kwargs,
    ):
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
        self._initial_loss = None
        self._current_step_size = step_size  # Track the current step size found by Armijo

        self.x = initial_estimate.clone()
        self.data_fidelity = data_fidelity
        self.prior = prior

        # Preconditioner parameters
        self.preconditioner = preconditioner  # Can be callable or ImageData
        self.preconditioner_update_initial = preconditioner_update_initial
        self.preconditioner_update_interval = preconditioner_update_interval
        self.eps_preconditioner = eps_preconditioner
        self._preconditioner_image = None

        super(MAPRL, self).__init__(**kwargs)
        self.configured = True

        if self.initial_line_search:
            LOGGER.info(
                "MAPRL Armijo: starting search with suggested step %.4g",
                self.initial_step_size,
            )
            step, loss = self._armijo_initial_step(self.initial_step_size)
            self.initial_step_size = step
            self._current_step_size = step
            self._initial_loss = loss
            LOGGER.info(
                "MAPRL Armijo: accepted step %.4g (objective %.6g)",
                step,
                loss,
            )

    def step_size(self):
        # Use the current step size found by Armijo for the first armijo_iterations
        if self.iteration <= self.armijo_iterations:
            return self._current_step_size
        # After Armijo period, continue from the last Armijo-found step size
        # and apply relaxation schedule from that point
        return self._current_step_size / (1 + self.relaxation_eta * (self.iteration - self.armijo_iterations))

    def update(self):
        # Update preconditioner if needed
        if self.preconditioner is not None:
            should_update = False
            if self.iteration <= self.preconditioner_update_initial:
                # Update every iteration for first N iterations
                should_update = True
            elif self.iteration % self.preconditioner_update_interval == 0:
                # Then update periodically
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
                        "MAPRL: updated preconditioner at iteration %d (min=%.3e, max=%.3e, mean=%.3e)",
                        self.iteration,
                        float(np.min(prec_arr)),
                        float(np.max(prec_arr)),
                        float(np.mean(prec_arr)),
                    )
                else:
                    LOGGER.info("MAPRL: updated preconditioner at iteration %d", self.iteration)

        # Determine if we should perform Armijo line search at this iteration
        should_armijo = False
        if self.iteration <= self.armijo_update_initial:
            # Update every iteration for first N iterations
            should_armijo = True
        elif self.iteration <= self.armijo_iterations and self.iteration % self.armijo_update_interval == 0:
            # Then update periodically until armijo_iterations
            should_armijo = True

        # Perform Armijo line search if scheduled
        if should_armijo:
            LOGGER.info(
                "MAPRL Armijo: performing line search at iteration %d (starting from step %.4g)",
                self.iteration,
                self.initial_step_size,
            )
            # Always restart from initial_step_size for Armijo search
            step, loss = self._armijo_step(self.initial_step_size)
            # Update current step size to use the found step
            self._current_step_size = step
            LOGGER.info(
                "MAPRL Armijo: iteration %d accepted step %.4g (objective %.6g)",
                self.iteration,
                step,
                loss,
            )
        else:
            grad = self.data_fidelity.gradient(self.x) + self.prior.gradient(self.x)

            # Apply preconditioner if available
            if self._preconditioner_image is not None:
                # then multiplied by (x + eps) as usual
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
        return last_step, last_loss

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
        # Update self.x with the last tested step
        self.x = current - direction * last_step
        with np.errstate(invalid="ignore"):
            self.x.maximum(0, out=self.x)
        return last_step, last_loss

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
