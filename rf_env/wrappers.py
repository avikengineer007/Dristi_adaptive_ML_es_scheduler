from typing import Tuple, Dict, Any
import gymnasium as gym
import numpy as np


class FlatObservationWrapper(gym.ObservationWrapper):
    """Ensures observations are flat float32 vectors within [0.0, 1.0]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)
