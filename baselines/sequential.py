from typing import Optional, Dict, Any
import numpy as np
from baselines.base import BaseScheduler


class SequentialSweep(BaseScheduler):
    """
    Standard sequential round-robin frequency sweep:
    Cycles systematically across bands: 0, 1, 2, ..., N-1, 0, ...
    """

    def __init__(self, num_bands: int, start_band: int = 0) -> None:
        super().__init__(num_bands=num_bands, name="Sequential Sweep")
        self.start_band = start_band
        self.current_band = start_band

    def reset(self, seed: Optional[int] = None) -> None:
        self.current_band = self.start_band

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        action = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return action

    def explain(self, obs: np.ndarray, action: int) -> str:
        return f"Sequential round-robin step: advancing to channel {action} (cycling 0 to {self.num_bands - 1})."
