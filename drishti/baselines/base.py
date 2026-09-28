from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np


class Scheduler(ABC):
    """
    Abstract base class for all DRISHTI frequency scan schedulers.
    """

    def __init__(self, num_bands: int, name: str) -> None:
        self.num_bands = num_bands
        self.name = name
        self.last_action: Optional[int] = None
        self.last_explanation: Dict[str, Any] = {}

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal scheduler state and initialize random seeds."""
        pass

    @abstractmethod
    def choose_action(self, observation: np.ndarray) -> int:
        """
        Selects the frequency band (0 to num_bands - 1) to monitor next.

        Args:
            observation: POMDP observation vector from the environment.

        Returns:
            Integer channel index.
        """
        pass

    def act(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        """Convenience alias for choose_action."""
        action = self.choose_action(observation)
        self.last_action = action
        return action

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """
        Updates online state using feedback from the environment.
        Default implementation is a no-op for fixed/static baseline sweeps.
        """
        pass

    def explain(self) -> Dict[str, Any]:
        """
        Returns structured decision explanation for the last selected action.
        """
        if self.last_explanation:
            return self.last_explanation
        return {
            "scheduler": self.name,
            "chosen_band": self.last_action,
            "reason": f"{self.name} selected band {self.last_action} based on default policy logic.",
            "scores": {},
        }
