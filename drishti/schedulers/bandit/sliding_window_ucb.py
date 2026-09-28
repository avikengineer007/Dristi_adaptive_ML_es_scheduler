from typing import Optional, Dict, Any, List
from collections import deque
import numpy as np

from drishti.baselines.base import Scheduler


class SlidingWindowUCB(Scheduler):
    """
    Sliding-Window Upper Confidence Bound (SW-UCB) Scheduler.

    Maintains a rolling window W of past dwell outcomes to track non-stationary
    RF activity. Computes UCB indices combining empirical detection payoff,
    exploration confidence intervals, Age-of-Information (AoI) urgency,
    and switching penalty mitigation.
    """

    def __init__(
        self,
        num_bands: int,
        window_size: int = 50,
        exploration_coef: float = 1.0,
        aoi_weight: float = 1.0,
        switch_penalty_weight: float = 0.2,
        prior_weights: Optional[Dict[int, float]] = None,
        name: str = "Sliding-Window UCB",
    ) -> None:
        super().__init__(num_bands=num_bands, name=name)
        self.window_size = int(window_size)
        self.exploration_coef = float(exploration_coef)
        self.aoi_weight = float(aoi_weight)
        self.switch_penalty_weight = float(switch_penalty_weight)
        self.prior_weights = prior_weights or {}

        # Rolling history buffer: tuples of (band, reward, detected_flag)
        self.history: deque = deque(maxlen=self.window_size)
        self.pull_counts = np.zeros(num_bands, dtype=np.int64)
        self.total_steps: int = 0
        self.rng = np.random.default_rng(42)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal history and state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.history.clear()
        self.pull_counts.fill(0)
        self.total_steps = 0
        self.last_action = None
        self.last_explanation = {}

    def choose_action(self, observation: np.ndarray) -> int:
        """
        Calculates SW-UCB indices for all bands and selects the optimal band.
        """
        B = self.num_bands
        norm_tau = observation[0:B] if len(observation) >= B else np.zeros(B, dtype=np.float32)

        # 1. Calculate window pulls and window rewards
        win_pulls = np.zeros(B, dtype=np.float32)
        win_rewards = np.zeros(B, dtype=np.float32)

        for band, r, det in self.history:
            win_pulls[band] += 1.0
            win_rewards[band] += r

        effective_w = min(self.total_steps, self.window_size)

        # Priority factors from prior intelligence (default 1.0)
        prior_arr = np.array([self.prior_weights.get(b, 1.0) for b in range(B)], dtype=np.float32)
        prior_arr_norm = prior_arr / np.max(prior_arr) if np.max(prior_arr) > 0 else np.ones(B, dtype=np.float32)

        ucb_scores = np.zeros(B, dtype=np.float32)
        empirical_means = np.zeros(B, dtype=np.float32)
        exploration_bonuses = np.zeros(B, dtype=np.float32)
        aoi_bonuses = np.zeros(B, dtype=np.float32)
        switch_penalties = np.zeros(B, dtype=np.float32)

        # Check for unpulled arms in the active sliding window
        unpulled = [b for b in range(B) if win_pulls[b] == 0]

        if unpulled:
            # Prioritize unpulled arms weighted by prior threat intelligence & AoI
            unpulled_scores = [(prior_arr_norm[b] * (1.0 + norm_tau[b])) for b in unpulled]
            best_idx = int(unpulled[np.argmax(unpulled_scores)])
            reason = f"Forced exploration: band {best_idx} unvisited in past {effective_w} slots."
            chosen_band = best_idx
        else:
            log_w = np.log(max(2.0, float(effective_w)))
            for b in range(B):
                n_b = win_pulls[b]
                mean_r = win_rewards[b] / n_b
                empirical_means[b] = mean_r

                # Classical SW-UCB exploration bonus
                bonus = self.exploration_coef * np.sqrt((2.0 * log_w) / n_b)
                exploration_bonuses[b] = bonus

                # Age-of-information urgency bonus (scaled by prior threat weight)
                aoi = self.aoi_weight * norm_tau[b] * prior_arr_norm[b]
                aoi_bonuses[b] = aoi

                # Switching penalty mitigation
                sw_pen = 0.0
                if self.last_action is not None and b != self.last_action:
                    sw_pen = -self.switch_penalty_weight
                switch_penalties[b] = sw_pen

                # Total index: empirical mean (threat-weighted) + exploration + AoI + switch penalty
                ucb_scores[b] = mean_r + bonus + aoi + sw_pen

            # Tie-breaking with small random jitter
            jitter = self.rng.uniform(0.0, 1e-6, size=B)
            chosen_band = int(np.argmax(ucb_scores + jitter))

            # Explain rationale
            if aoi_bonuses[chosen_band] > exploration_bonuses[chosen_band] and aoi_bonuses[chosen_band] > 0.5:
                reason = f"AoI Urgency: band {chosen_band} unvisited for {norm_tau[chosen_band]:.2f} norm tau."
            elif empirical_means[chosen_band] > 1.0:
                reason = f"High payoff exploitation: mean reward {empirical_means[chosen_band]:.2f} in window."
            else:
                reason = f"UCB exploration: score {ucb_scores[chosen_band]:.2f} (exploration bonus {exploration_bonuses[chosen_band]:.2f})."

        self.last_action = chosen_band
        self.last_explanation = {
            "scheduler": self.name,
            "chosen_band": chosen_band,
            "reason": reason,
            "scores": {b: float(ucb_scores[b]) for b in range(B)},
            "components": {
                "window_pulls": {b: int(win_pulls[b]) for b in range(B)},
                "empirical_means": {b: float(empirical_means[b]) for b in range(B)},
                "exploration_bonuses": {b: float(exploration_bonuses[b]) for b in range(B)},
                "aoi_bonuses": {b: float(aoi_bonuses[b]) for b in range(B)},
            },
        }

        return chosen_band

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """
        Appends dwell outcome to sliding window buffer.
        """
        detected = bool(info.get("detected", False))
        self.history.append((action, float(reward), detected))
        self.pull_counts[action] += 1
        self.total_steps += 1
