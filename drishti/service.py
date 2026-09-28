from typing import Optional, Dict, Any, List, Union
import os
import numpy as np

from drishti.baselines.base import Scheduler
from drishti.baselines.sequential import SequentialSweep
from drishti.baselines.random_scan import RandomScan
from drishti.baselines.priority_sweep import PriorityPreMissionSweep
from drishti.schedulers.bandit import SlidingWindowUCB, DiscountedThompson
from drishti.schedulers.periodic_aware import PeriodicAwareScheduler
from drishti.schedulers.ppo import PPOScheduler
from drishti.explain.logger import ExplanationLogger, DecisionRecord


class ScanScheduler:
    """
    Unified Production Service for Tactical ES Receiver Frequency Scheduling.

    Provides a clean, unified API for selecting, switching, and auto-routing
    among all DRISHTI scan strategies (Baselines, Bandits, Periodic-Aware, PPO)
    while maintaining automatic explainability and forensic JSONL audit logging.
    """

    AVAILABLE_MODES = [
        "auto",
        "periodic",
        "ppo",
        "bandit_ucb",
        "bandit_ts",
        "priority",
        "sequential",
        "random",
    ]

    def __init__(
        self,
        num_bands: int = 16,
        mode: str = "auto",
        prior_weights: Optional[Dict[int, float]] = None,
        ppo_model_path: Optional[str] = "models/ppo_augmented.zip",
        max_log_records: int = 1000,
    ) -> None:
        self.num_bands = int(num_bands)
        self.mode = mode.lower()
        self.prior_weights = prior_weights or {}
        self.ppo_model_path = ppo_model_path
        self.logger = ExplanationLogger(max_records=max_log_records)
        self.current_slot: int = 0

        # Build backend schedulers
        self._schedulers: Dict[str, Scheduler] = {
            "sequential": SequentialSweep(num_bands=self.num_bands),
            "random": RandomScan(num_bands=self.num_bands),
            "priority": PriorityPreMissionSweep(num_bands=self.num_bands, band_priorities=self.prior_weights),
            "bandit_ucb": SlidingWindowUCB(
                num_bands=self.num_bands,
                window_size=120,
                exploration_coef=0.5,
                aoi_weight=0.5,
                prior_weights=self.prior_weights,
            ),
            "bandit_ts": DiscountedThompson(
                num_bands=self.num_bands,
                gamma=0.995,
                aoi_weight=2.0,
                prior_weights=self.prior_weights,
            ),
            "periodic": PeriodicAwareScheduler(
                num_bands=self.num_bands,
                window_size=120,
                exploration_coef=0.5,
                aoi_weight=0.5,
                prior_weights=self.prior_weights,
            ),
        }

        # Initialize PPO if weights exist
        resolved_ppo_path = None
        if self.ppo_model_path and os.path.exists(self.ppo_model_path):
            resolved_ppo_path = self.ppo_model_path
        elif os.path.exists("models/ppo_augmented.zip"):
            resolved_ppo_path = "models/ppo_augmented.zip"
        elif os.path.exists("models/ppo_pure.zip"):
            resolved_ppo_path = "models/ppo_pure.zip"

        if resolved_ppo_path:
            self._schedulers["ppo"] = PPOScheduler(
                num_bands=self.num_bands,
                model_path=resolved_ppo_path,
                use_periodic_features="augmented" in resolved_ppo_path,
                name="PPO RL Service Policy",
            )

        self._active_scheduler = self._resolve_scheduler(self.mode)

    def _resolve_scheduler(self, mode: str) -> Scheduler:
        """Resolves target scheduler mode with robust fallbacks."""
        if mode == "auto":
            # Auto-selection priority:
            # 1. PPO if trained model available
            # 2. Periodic-Aware Scheduler (handles burst synchronization)
            # 3. Sliding-Window UCB
            if "ppo" in self._schedulers:
                return self._schedulers["ppo"]
            return self._schedulers["periodic"]

        if mode in self._schedulers:
            return self._schedulers[mode]

        # Safe fallback
        return self._schedulers["bandit_ucb"]

    def set_mode(self, mode: str) -> None:
        """Switches scheduler strategy at runtime."""
        mode_clean = mode.lower()
        if mode_clean not in self.AVAILABLE_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Available modes: {self.AVAILABLE_MODES}")
        self.mode = mode_clean
        self._active_scheduler = self._resolve_scheduler(self.mode)

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets all internal scheduler states and clears logger."""
        for s in self._schedulers.values():
            s.reset(seed=seed)
        self.logger.clear()
        self.current_slot = 0

    def step(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        """Selects the next frequency band to dwell on and logs decision explanation."""
        action = self._active_scheduler.act(obs, info)
        exp = self._active_scheduler.explain()

        self.logger.log(
            slot=self.current_slot,
            chosen_band=action,
            scheduler_name=self._active_scheduler.name,
            reason=exp.get("reason", "Default scan logic"),
            scores=exp.get("scores", {}),
            components=exp.get("components", {}),
        )

        return action

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Updates internal scheduler with environment feedback."""
        self._active_scheduler.update(obs=obs, action=action, reward=reward, info=info)

        # Update last logged record with outcome feedback
        if self.logger.records:
            last_rec = self.logger.records[-1]
            last_rec.detected = bool(info.get("detected", False))
            last_rec.reward = float(reward)

        self.current_slot += 1

    def explain(self) -> Dict[str, Any]:
        """Returns the structured explanation for the last action."""
        return self._active_scheduler.explain()

    def explain_contrastive(self, chosen_band: int, alternative_band: int) -> str:
        """Explains why chosen_band was selected over alternative_band."""
        return self.logger.explain_contrastive(chosen_band, alternative_band)

    def export_logs(self, filepath: str = "results/scheduler_audit_log.jsonl") -> str:
        """Exports decision audit log to JSONL file."""
        return self.logger.export_jsonl(filepath)

    def get_recent_logs(self, n: int = 50) -> List[Dict[str, Any]]:
        """Returns the n most recent decision records."""
        return self.logger.get_recent(n)
