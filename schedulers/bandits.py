from collections import deque
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from baselines.base import BaseScheduler


class SlidingWindowUCB(BaseScheduler):
    """
    Sliding-Window Upper Confidence Bound (SW-UCB) Scheduler for Non-Stationary EW Spectrum.
    Maintains a sliding window of historical observations of size W.
    Computes local empirical detection/reward rates and an exploration bonus.
    """

    def __init__(
        self,
        num_bands: int,
        window_size: int = 60,
        exploration_coef: float = 1.2,
        unvisited_bonus: float = 1e4,
    ) -> None:
        super().__init__(num_bands=num_bands, name="Sliding-Window UCB")
        self.window_size = window_size
        self.exploration_coef = exploration_coef
        self.unvisited_bonus = unvisited_bonus

        self.window: deque[Tuple[int, float, bool]] = deque(maxlen=window_size)
        self.t: int = 0
        self.last_indices: np.ndarray = np.zeros(num_bands, dtype=np.float64)

    def reset(self, seed: Optional[int] = None) -> None:
        self.window.clear()
        self.t = 0
        self.last_indices.fill(0.0)

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        self.t += 1

        # Count pulls and cumulative rewards in the current sliding window
        window_pulls = np.zeros(self.num_bands, dtype=np.float64)
        window_rewards = np.zeros(self.num_bands, dtype=np.float64)

        for band, rew, detected in self.window:
            window_pulls[band] += 1.0
            window_rewards[band] += rew

        effective_w = min(self.t, self.window_size)
        indices = np.zeros(self.num_bands, dtype=np.float64)

        for b in range(self.num_bands):
            n_b = window_pulls[b]
            if n_b == 0:
                # Force exploration of bands unobserved in current window
                indices[b] = self.unvisited_bonus + (self.num_bands - b) * 0.01
            else:
                mean_r = window_rewards[b] / n_b
                bonus = self.exploration_coef * np.sqrt(2.0 * np.log(effective_w) / n_b)
                indices[b] = mean_r + bonus

        self.last_indices = indices.copy()
        return int(np.argmax(indices))

    def update(
        self,
        action: int,
        reward: float,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        detected = bool(info.get("detected", False)) if info else False
        self.window.append((action, float(reward), detected))

    def explain(self, obs: np.ndarray, action: int) -> str:
        idx_val = self.last_indices[action]
        # Count pulls in window
        n_pulls = sum(1 for b, _, _ in self.window if b == action)
        if n_pulls == 0:
            return (
                f"SW-UCB selected band {action}: channel has 0 visits in the last {self.window_size} "
                f"slots; triggering mandatory exploration to intercept newly activated emitters."
            )
        avg_r = sum(r for b, r, _ in self.window if b == action) / n_pulls
        return (
            f"SW-UCB selected band {action}: empirical reward={avg_r:.2f} over {n_pulls} window pulls, "
            f"combined UCB score={idx_val:.2f}."
        )


class DiscountedThompsonSampling(BaseScheduler):
    """
    Discounted Thompson Sampling (D-TS) for Non-Stationary EW Band Scheduling.
    Uses Beta-Bernoulli conjugate distributions with exponential discounting factor gamma in (0, 1)
    to continuously adapt to dynamic, hopping, and periodic emitter appearances.
    """

    def __init__(
        self,
        num_bands: int,
        gamma: float = 0.94,
        threat_scaling: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(num_bands=num_bands, name="Discounted Thompson Sampling")
        self.gamma = gamma
        self.threat_scaling = threat_scaling
        self.rng = np.random.default_rng(seed)

        self.alpha = np.ones(num_bands, dtype=np.float64)
        self.beta = np.ones(num_bands, dtype=np.float64)
        self.last_samples: np.ndarray = np.zeros(num_bands, dtype=np.float64)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.alpha.fill(1.0)
        self.beta.fill(1.0)
        self.last_samples.fill(0.0)

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        # Sample probability of emitter presence from Beta posterior for each arm
        samples = self.rng.beta(self.alpha, self.beta)

        # Incorporate time-since-visit exploration term from observation if available
        # obs[0 : num_bands] is normalized time-since-last-visit
        tau_bonus = 0.0
        if obs is not None and len(obs) >= self.num_bands:
            norm_tau = obs[:self.num_bands]
            tau_bonus = 0.25 * norm_tau

        effective_score = samples + tau_bonus
        self.last_samples = effective_score.copy()
        return int(np.argmax(effective_score))

    def update(
        self,
        action: int,
        reward: float,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Apply recency discount to all arms to track non-stationarity
        self.alpha = self.gamma * self.alpha + (1.0 - self.gamma)
        self.beta = self.gamma * self.beta + (1.0 - self.gamma)

        # Update targeted band based on detection outcome
        detected = bool(info.get("detected", False)) if info else (reward > 0)
        if detected:
            threat_reward = float(info.get("threat_reward", 1.0)) if info else 1.0
            # Higher threat detections give stronger positive reinforcement
            increment = min(3.0, 1.0 + 0.2 * threat_reward)
            self.alpha[action] += increment
        else:
            self.beta[action] += 1.0

    def explain(self, obs: np.ndarray, action: int) -> str:
        a_val = self.alpha[action]
        b_val = self.beta[action]
        sampled_val = self.last_samples[action]
        mean_p = a_val / (a_val + b_val)
        return (
            f"D-TS selected band {action}: posterior Beta({a_val:.2f}, {b_val:.2f}) with expected "
            f"activity={mean_p:.2%}, sampled acquisition index={sampled_val:.3f}."
        )
