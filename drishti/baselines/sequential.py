from typing import Optional, Dict, Any
import numpy as np
from drishti.baselines.base import Scheduler


class SequentialSweep(Scheduler):
    """
    Open-loop sequential round-robin frequency sweep:
    Cycles systematically across channels: 0, 1, 2, ..., B-1, 0, ...
    Guarantees every band is visited once per B steps.
    """

    def __init__(self, num_bands: int, start_band: int = 0) -> None:
        super().__init__(num_bands=num_bands, name="Sequential Sweep")
        self.start_band = start_band
        self.current_band = start_band

    def reset(self, seed: Optional[int] = None) -> None:
        self.current_band = self.start_band
        self.last_action = None
        self.last_explanation = {}

    def choose_action(self, observation: np.ndarray) -> int:
        action = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        self.last_action = action
        self.last_explanation = {
            "scheduler": self.name,
            "chosen_band": action,
            "reason": f"Round-robin open-loop step: advancing channel ({action} -> {(action + 1) % self.num_bands}).",
            "scores": {b: 1.0 if b == action else 0.0 for b in range(self.num_bands)},
        }
        return action
