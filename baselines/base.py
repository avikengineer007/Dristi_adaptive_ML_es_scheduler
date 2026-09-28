from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseScheduler(ABC):
    """Abstract base class for all ES receiver frequency scan schedulers."""

    def __init__(self, num_bands: int, name: str) -> None:
        self.num_bands = num_bands
        self.name = name

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal scheduler state and initialize random seeds."""
        pass

    @abstractmethod
    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        """
        Choose the next frequency band to dwell on.

        Args:
            obs: Observation vector from the environment.
            info: Optional info dictionary from the last step.

        Returns:
            Integer band index in range [0, num_bands - 1].
        """
        pass

    def explain(self, obs: np.ndarray, action: int) -> str:
        """Provide a human-readable explanation for why this band was selected."""
        return f"{self.name} selected band {action} based on default policy logic."
