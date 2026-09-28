from typing import Optional, Dict, Any, Union, List
import os
import numpy as np
import torch

from drishti.baselines.base import Scheduler
from drishti.models.periodicity import PeriodicEmitterTracker

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # type: ignore


class PPOScheduler(Scheduler):
    """
    Deep Reinforcement Learning Frequency Scan Scheduler using Proximal Policy Optimization (PPO).

    Evaluates the continuous state representation (normalized age-of-information,
    running hit/miss ratios, last detection state, and optional periodic features)
    through a deep neural network actor to produce optimal instantaneous tuning actions.
    """

    def __init__(
        self,
        num_bands: int,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        use_periodic_features: bool = False,
        deterministic: bool = True,
        name: str = "PPO Scheduler",
    ) -> None:
        super().__init__(num_bands=num_bands, name=name)
        self.use_periodic_features = use_periodic_features
        self.deterministic = deterministic
        self.model = model
        self.model_path = model_path
        self.tracker = PeriodicEmitterTracker(num_bands=num_bands) if use_periodic_features else None
        self.current_slot: int = 0
        self.last_action_probs: Optional[np.ndarray] = None

        if self.model is None and self.model_path is not None and os.path.exists(self.model_path):
            self.load(self.model_path)

    def load(self, path: str) -> None:
        """Loads trained SB3 PPO model from disk."""
        if PPO is None:
            raise ImportError("stable-baselines3 is required to load PPO models.")
        self.model = PPO.load(path)
        self.model_path = path

    def reset(self, seed: Optional[int] = None) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        self.current_slot = 0
        self.last_action = None
        self.last_action_probs = None
        self.last_explanation = {}

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        if self.tracker is None:
            return base_obs

        B = self.num_bands
        norm_periods = np.zeros(B, dtype=np.float32)
        norm_arrivals = np.ones(B, dtype=np.float32)
        coherences = np.zeros(B, dtype=np.float32)

        for b, est in enumerate(self.tracker.estimators):
            if est.estimated_period is not None:
                norm_periods[b] = float(np.clip(est.estimated_period / 150.0, 0.0, 1.0))
                coherences[b] = float(est.coherence_score)
                next_arr = est.predict_next_arrival(self.current_slot)
                if next_arr is not None:
                    delta = next_arr - self.current_slot
                    norm_arrivals[b] = float(np.clip(delta / 150.0, 0.0, 1.0))

        return np.concatenate([base_obs, norm_periods, norm_arrivals, coherences], dtype=np.float32)

    def choose_action(self, observation: np.ndarray) -> int:
        if self.model is None:
            # Fallback if model not loaded: choose band with highest AoI
            B = self.num_bands
            norm_tau = observation[0:B] if len(observation) >= B else np.zeros(B)
            action = int(np.argmax(norm_tau))
            self.last_action = action
            return action

        obs_in = self._augment_obs(observation)

        # Predict action using PPO policy
        action, _ = self.model.predict(obs_in, deterministic=self.deterministic)
        action_int = int(action)

        # Extract policy distribution probabilities for explainability
        try:
            obs_tensor = torch.as_tensor(obs_in, dtype=torch.float32).unsqueeze(0).to(self.model.policy.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
                self.last_action_probs = probs
        except Exception:
            self.last_action_probs = None

        self.last_action = action_int

        # Explain rationale
        prob_str = f" (P={self.last_action_probs[action_int]:.2f})" if self.last_action_probs is not None else ""
        reason = f"PPO Policy selected band {action_int}{prob_str} based on actor network value-maximization."

        self.last_explanation = {
            "scheduler": self.name,
            "chosen_band": action_int,
            "reason": reason,
            "scores": {b: float(self.last_action_probs[b]) for b in range(self.num_bands)} if self.last_action_probs is not None else {},
            "is_augmented": self.use_periodic_features,
        }

        return action_int

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        slot = int(info.get("slot", self.current_slot))
        detected = bool(info.get("detected", False))
        if self.tracker is not None:
            self.tracker.update(band=action, slot=slot, detected=detected)
        self.current_slot = slot + 1
