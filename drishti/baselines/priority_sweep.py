from typing import Optional, Dict, Any, List
import numpy as np
from drishti.baselines.base import Scheduler


class PriorityPreMissionSweep(Scheduler):
    """
    Fixed Pre-Mission Priority Sweep:
    Allocates dwells across frequency channels according to pre-mission threat intelligence.
    Generates a deterministic weighted round-robin scan schedule where high-priority
    channels are visited with higher frequency, providing lower revisit latency.
    Never updated online.
    """

    def __init__(
        self,
        num_bands: int,
        band_priorities: Optional[Dict[int, float]] = None,
        default_weight: float = 1.0,
    ) -> None:
        """
        Args:
            num_bands: Number of frequency channels (B).
            band_priorities: Dictionary mapping band index to priority weight (e.g. {15: 10.0, 4: 5.0}).
            default_weight: Weight assigned to unlisted/ambient channels.
        """
        super().__init__(num_bands=num_bands, name="Priority Pre-Mission Sweep")
        self.band_priorities = band_priorities if band_priorities is not None else {}
        self.default_weight = default_weight

        # Weights vector
        self.weights = np.full(num_bands, fill_value=default_weight, dtype=np.float32)
        for band, weight in self.band_priorities.items():
            if 0 <= band < num_bands:
                self.weights[band] = float(weight)

        self._schedule: List[int] = self._build_weighted_schedule()
        self._index: int = 0

    def _build_weighted_schedule(self) -> List[int]:
        """
        Builds an interleaved weighted round-robin schedule using Bresenham / largest remainder logic.
        """
        min_weight = max(0.01, float(np.min(self.weights)))
        normalized_counts = np.maximum(1, np.round(self.weights / min_weight)).astype(int)

        schedule: List[int] = []
        remaining = normalized_counts.copy()
        total_slots = int(np.sum(remaining))

        for _ in range(total_slots):
            best_band = int(np.argmax(remaining))
            schedule.append(best_band)
            remaining[best_band] -= 1

        return schedule if schedule else list(range(self.num_bands))

    def reset(self, seed: Optional[int] = None) -> None:
        self._index = 0
        self.last_action = None
        self.last_explanation = {}

    def choose_action(self, observation: np.ndarray) -> int:
        action = self._schedule[self._index % len(self._schedule)]
        self._index += 1
        self.last_action = action
        weight = float(self.weights[action])
        self.last_explanation = {
            "scheduler": self.name,
            "chosen_band": action,
            "reason": f"Pre-mission priority sweep: selected high-threat channel {action} (assigned priority weight: {weight:.1f}).",
            "scores": {b: float(self.weights[b]) for b in range(self.num_bands)},
        }
        return action
