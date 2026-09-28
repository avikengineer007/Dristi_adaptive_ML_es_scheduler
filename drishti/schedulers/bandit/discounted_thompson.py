from typing import Optional, Dict, Any, List
import numpy as np

from drishti.baselines.base import Scheduler


class DiscountedThompson(Scheduler):
    """
    Discounted Thompson Sampling (D-TS) Scheduler.

    Maintains a Beta-Bernoulli conjugate model per frequency band with exponential
    recency discounting factor gamma in (0, 1). Adapts continuously to time-varying
    duty cycles, frequency-hopping emitters, and rotating search beams while
    incorporating Age-of-Information (AoI) revisit urgency.
    """

    def __init__(
        self,
        num_bands: int,
        gamma: float = 0.95,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        aoi_weight: float = 0.5,
        switch_penalty_weight: float = 0.2,
        prior_weights: Optional[Dict[int, float]] = None,
        name: str = "Discounted Thompson",
    ) -> None:
        super().__init__(num_bands=num_bands, name=name)
        self.gamma = float(gamma)
        self.alpha_0 = float(alpha_0)
        self.beta_0 = float(beta_0)
        self.aoi_weight = float(aoi_weight)
        self.switch_penalty_weight = float(switch_penalty_weight)
        self.prior_weights = prior_weights or {}

        self.alphas = np.full(num_bands, self.alpha_0, dtype=np.float64)
        self.betas = np.full(num_bands, self.beta_0, dtype=np.float64)
        self.pull_counts = np.zeros(num_bands, dtype=np.int64)
        self.rng = np.random.default_rng(42)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset Beta distribution parameters and seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.alphas.fill(self.alpha_0)
        self.betas.fill(self.beta_0)
        self.pull_counts.fill(0)
        self.last_action = None
        self.last_explanation = {}

    def choose_action(self, observation: np.ndarray) -> int:
        """
        Samples detection probabilities from Beta posteriors and applies
        threat prior weighting, AoI exploration, and switching penalty.
        """
        B = self.num_bands
        norm_tau = observation[0:B] if len(observation) >= B else np.zeros(B, dtype=np.float32)

        # Normalize prior threat weights
        prior_arr = np.array([self.prior_weights.get(b, 1.0) for b in range(B)], dtype=np.float64)
        max_prior = np.max(prior_arr) if np.max(prior_arr) > 0 else 1.0
        prior_arr_norm = prior_arr / max_prior

        # Sample from current Beta posteriors
        # Ensure numerical safety for Beta distribution parameters
        safe_alphas = np.maximum(1e-3, self.alphas)
        safe_betas = np.maximum(1e-3, self.betas)
        sampled_thetas = self.rng.beta(safe_alphas, safe_betas)

        # AoI urgency component
        aoi_bonuses = self.aoi_weight * norm_tau * prior_arr_norm

        # Switching cost penalty
        switch_penalties = np.zeros(B, dtype=np.float64)
        if self.last_action is not None:
            for b in range(B):
                if b != self.last_action:
                    switch_penalties[b] = -self.switch_penalty_weight

        # Composite score
        scores = (sampled_thetas * prior_arr_norm) + aoi_bonuses + switch_penalties

        chosen_band = int(np.argmax(scores))

        # Explain rationale
        expected_prob = safe_alphas[chosen_band] / (safe_alphas[chosen_band] + safe_betas[chosen_band])
        if aoi_bonuses[chosen_band] > sampled_thetas[chosen_band] and aoi_bonuses[chosen_band] > 0.3:
            reason = f"AoI Urgency: band {chosen_band} unvisited (norm_tau={norm_tau[chosen_band]:.2f})."
        elif expected_prob > 0.5:
            reason = f"High posterior probability: expected P_det={expected_prob:.2f} (sampled={sampled_thetas[chosen_band]:.2f})."
        else:
            reason = f"Thompson sampling exploration: sampled value {sampled_thetas[chosen_band]:.2f} on band {chosen_band}."

        self.last_action = chosen_band
        self.last_explanation = {
            "scheduler": self.name,
            "chosen_band": chosen_band,
            "reason": reason,
            "scores": {b: float(scores[b]) for b in range(B)},
            "components": {
                "sampled_thetas": {b: float(sampled_thetas[b]) for b in range(B)},
                "expected_probs": {b: float(expected_prob) for b in range(B)},
                "alphas": {b: float(self.alphas[b]) for b in range(B)},
                "betas": {b: float(self.betas[b]) for b in range(B)},
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
        Applies exponential discount gamma to all arms and updates the chosen arm.
        """
        # 1. Decay all arms towards uninformative prior
        self.alphas = (self.gamma * self.alphas) + ((1.0 - self.gamma) * self.alpha_0)
        self.betas = (self.gamma * self.betas) + ((1.0 - self.gamma) * self.beta_0)

        # 2. Add feedback from chosen arm
        detected = bool(info.get("detected", False))
        threat_reward = float(info.get("threat_reward", 0.0))

        if detected:
            # Payoff increment weighted by threat severity
            increment = max(1.0, threat_reward)
            self.alphas[action] += increment
        else:
            self.betas[action] += 1.0

        self.pull_counts[action] += 1
