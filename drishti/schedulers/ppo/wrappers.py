from typing import Optional, Dict, Any, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from drishti.env.environment import SpectrumScanEnv
from drishti.models.periodicity import PeriodicEmitterTracker


class PeriodicFeatureWrapper(gym.Wrapper):
    """
    Observation wrapper augmenting the baseline POMDP state vector
    with real-time periodicity estimates and arrival forecasts.

    Baseline observation (dim = 4*B + 1):
      [norm_tau, hit_ratio, miss_ratio, last_seen, norm_current_band]

    Augmented observation (dim = 7*B + 1):
      Baseline features + [norm_period, norm_time_to_arrival, coherence_score] per band.
    """

    def __init__(self, env: SpectrumScanEnv, max_period_norm: float = 150.0) -> None:
        super().__init__(env)
        self.num_bands = env.num_bands
        self.max_period_norm = float(max_period_norm)
        self.tracker = PeriodicEmitterTracker(num_bands=self.num_bands)

        orig_shape = env.observation_space.shape[0]
        augmented_dim = orig_shape + (3 * self.num_bands)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(augmented_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.tracker.reset()
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment_obs(obs), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        detected = bool(info.get("detected", False))
        slot = int(info.get("slot", 0))
        self.tracker.update(band=action, slot=slot, detected=detected)
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        B = self.num_bands
        current_slot = getattr(self.env.unwrapped, "current_slot", 0)

        norm_periods = np.zeros(B, dtype=np.float32)
        norm_arrivals = np.ones(B, dtype=np.float32)
        coherences = np.zeros(B, dtype=np.float32)

        for b, est in enumerate(self.tracker.estimators):
            if est.estimated_period is not None:
                norm_periods[b] = float(np.clip(est.estimated_period / self.max_period_norm, 0.0, 1.0))
                coherences[b] = float(est.coherence_score)
                next_arr = est.predict_next_arrival(current_slot)
                if next_arr is not None:
                    delta = next_arr - current_slot
                    norm_arrivals[b] = float(np.clip(delta / self.max_period_norm, 0.0, 1.0))

        return np.concatenate([base_obs, norm_periods, norm_arrivals, coherences], dtype=np.float32)
