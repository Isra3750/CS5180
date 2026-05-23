# part b - three policies
import numpy as np

# accept with probability 1/T each week
class RandomPolicy:
    def __init__(self, T: int):
        self.p = 1.0 / T

    def act(self, obs) -> int:
        return 1 if np.random.rand() < self.p else 0

# accept if quality (u_t) >= u_min
class ThresholdPolicy:
    def __init__(self, u_min: float):
        self.u_min = u_min

    def act(self, obs) -> int:
        
        t, u_obs = obs
        return 1 if u_obs >= self.u_min else 0

# accept if quality (u_t) >= precise threshold (problem 1c hardcoded)
class OptimalPolicy:
    def __init__(self):
        # Backwards induction results
        self.thresholds = {
            1: 3.25,
            2: 3.00,
            3: 2.50,
            4: 0.00
        }

    def act(self, obs) -> int:        
        t, u_t = obs
        thresh = self.thresholds.get(int(t), 0.0)
        return 1 if u_t >= thresh else 0