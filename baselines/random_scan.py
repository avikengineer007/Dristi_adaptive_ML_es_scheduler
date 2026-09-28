from typing import Optional, Dict, Any
import numpy as np
from baselines.base import BaseScheduler


class RandomScan(BaseScheduler):
    """
    Uniform random frequency scanner:
    Samples next band uniformly at random: b ~ Uniform(0, N - 1).
    """

    def __init__(self, num_bands: int, seed: Optional[int] = None) -> None:
        super().__init__(num_bands=num_bands, name="Random Scan")
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        return int(self.rng.integers(0, self.num_bands))

    def explain(self, obs: np.ndarray, action: int) -> str:
        return f"Random scan step: uniformly sampled band {action} with probability 1/{self.num_bands}."
