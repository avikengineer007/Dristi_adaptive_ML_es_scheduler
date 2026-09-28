from typing import Optional, Dict, Any, List
import numpy as np

from drishti.baselines.base import Scheduler
from drishti.schedulers.bandit.sliding_window_ucb import SlidingWindowUCB
from drishti.models.periodicity import PeriodicEmitterTracker


class PeriodicAwareScheduler(Scheduler):
    """
    Periodic-Aware Lookahead Predictive Scan Scheduler.

    Fuses non-stationary multi-armed bandit exploration with real-time
    Circular Phase Coherence period tracking. When periodic radar bursts,
    rotating search beams, or target scanning receivers are forecasted to emit
    on a channel in slot t, the scheduler preempts general exploration and
    synchronizes its dwell window to intercept the pulse right as it illuminates.
    """

    def __init__(
        self,
        num_bands: int,
        window_size: int = 120,
        exploration_coef: float = 0.5,
        aoi_weight: float = 0.5,
        switch_penalty_weight: float = 0.2,
        coherence_threshold: float = 0.70,
        prior_weights: Optional[Dict[int, float]] = None,
        name: str = "Periodic-Aware Scheduler",
    ) -> None:
        super().__init__(num_bands=num_bands, name=name)
        self.coherence_threshold = float(coherence_threshold)
        self.prior_weights = prior_weights or {}

        # Base exploration engine
        self.bandit = SlidingWindowUCB(
            num_bands=num_bands,
            window_size=window_size,
            exploration_coef=exploration_coef,
            aoi_weight=aoi_weight,
            switch_penalty_weight=switch_penalty_weight,
            prior_weights=self.prior_weights,
            name="Periodic-Aware Base UCB",
        )

        # Multichannel period and target synchronization tracker
        self.tracker = PeriodicEmitterTracker(
            num_bands=num_bands,
            coherence_threshold=self.coherence_threshold,
        )

        self.current_slot: int = 0
        self.rng = np.random.default_rng(42)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.bandit.reset(seed=seed)
        self.tracker.reset()
        self.current_slot = 0
        self.last_action = None
        self.last_explanation = {}

    def choose_action(self, observation: np.ndarray) -> int:
        """
        Evaluates predictive periodic synchronization before falling back to SW-UCB.
        """
        B = self.num_bands
        upcoming = self.tracker.get_upcoming_bursts(self.current_slot, horizon=1)

        prior_arr = np.array([self.prior_weights.get(b, 1.0) for b in range(B)], dtype=np.float32)
        max_prior = np.max(prior_arr) if np.max(prior_arr) > 0 else 1.0
        prior_arr_norm = prior_arr / max_prior

        # Check if any periodic burst is predicted with high confidence
        high_conf_bands = [
            (b, conf) for b, conf in upcoming.items()
            if conf >= self.coherence_threshold and 0 <= b < B
        ]

        if high_conf_bands:
            # Score by confidence * prior threat
            best_band, best_conf = max(
                high_conf_bands,
                key=lambda item: item[1] * (1.0 + prior_arr_norm[item[0]])
            )
            est = self.tracker.estimators[best_band]
            period_str = f"T={est.estimated_period}" if est.estimated_period else "target cycle"
            reason = (
                f"Predictive Synchronization: Periodic emission forecasted on band {best_band} "
                f"({period_str}, confidence={best_conf:.2f})."
            )
            chosen_band = best_band
            scores = {b: (upcoming.get(b, 0.0) * prior_arr_norm[b]) for b in range(B)}
        else:
            # Fall back to SW-UCB exploration / exploitation
            chosen_band = self.bandit.act(observation)
            base_exp = self.bandit.explain()
            reason = f"SW-UCB Policy: {base_exp.get('reason', 'Default UCB scan')}"
            scores = base_exp.get("scores", {})

        self.last_action = chosen_band
        self.last_explanation = {
            "scheduler": self.name,
            "chosen_band": chosen_band,
            "reason": reason,
            "scores": scores,
            "is_predictive_sync": bool(high_conf_bands),
            "tracked_periods": {
                b: est.estimated_period
                for b, est in enumerate(self.tracker.estimators)
                if est.estimated_period is not None
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
        """Updates internal tracker and base bandit."""
        detected = bool(info.get("detected", False))
        slot = int(info.get("slot", self.current_slot))

        self.tracker.update(band=action, slot=slot, detected=detected)
        self.bandit.update(obs=obs, action=action, reward=reward, info=info)
        self.current_slot = slot + 1
