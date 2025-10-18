import numpy as np

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

    def __init__(self, initial_estimate, data_fidelity, prior, step_size=1, relaxation_eta=0.01, eps=1e-8, **kwargs):

        self.initial_estimate = initial_estimate
        self.initial_step_size = step_size
        self.relaxation_eta = relaxation_eta
        self.eps = eps

        self.x = initial_estimate.clone()
        self.data_fidelity = data_fidelity
        self.prior = prior

        super(MAPRL, self).__init__(**kwargs)
        self.configured = True


    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.iteration)
    
    def update(self):
        
        grad = self.data_fidelity.gradient(self.x) + self.prior.gradient(self.x)
        self.x = self.x - (self.x + self.eps) * grad * self.step_size()
        with np.errstate(invalid="ignore"):
            self.x.maximum(0, out=self.x)

    def update_objective(self):
        self.loss.append(self.data_fidelity(self.x) + self.prior(self.x))
