from typing import Optional, Dict, Any
import numpy as np
from drishti.baselines.base import Scheduler


class RandomScan(Scheduler):
    """
    Uniform random frequency scanner:
    Samples next band uniformly at random: b ~ Uniform(0, B - 1).
    """

    def __init__(self, num_bands: int, seed: Optional[int] = None) -> None:
        super().__init__(num_bands=num_bands, name="Random Scan")
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.last_action = None
        self.last_explanation = {}

    def choose_action(self, observation: np.ndarray) -> int:
        action = int(self.rng.integers(0, self.num_bands))
        self.last_action = action
        self.last_explanation = {
            "scheduler": self.name,
            "chosen_band": action,
            "reason": f"Random scan step: uniform random draw with p = 1/{self.num_bands}.",
            "scores": {b: 1.0 / self.num_bands for b in range(self.num_bands)},
        }
        return action
