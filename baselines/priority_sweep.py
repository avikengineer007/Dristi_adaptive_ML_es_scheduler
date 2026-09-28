from typing import Optional, Dict, Any, List
import numpy as np
from baselines.base import BaseScheduler


class PrioritySweep(BaseScheduler):
    """
    Fixed Pre-Mission Priority Sweep:
    Allocates dwells across frequency channels according to pre-mission threat intelligence.
    Generates a deterministic weighted round-robin scan schedule where high-priority
    channels are visited with higher frequency, providing lower revisit latency.
    """

    def __init__(
        self,
        num_bands: int,
        band_priorities: Optional[Dict[int, float]] = None,
        default_weight: float = 1.0,
    ) -> None:
        """
        Args:
            num_bands: Number of frequency channels.
            band_priorities: Dictionary mapping band index to priority weight (e.g., {15: 10.0, 5: 5.0}).
            default_weight: Weight assigned to unlisted/ambient bands.
        """
        super().__init__(num_bands=num_bands, name="Priority Sweep")
        self.band_priorities = band_priorities if band_priorities is not None else {}
        self.default_weight = default_weight

        # Build weights vector
        self.weights = np.full(num_bands, fill_value=default_weight, dtype=np.float32)
        for band, weight in self.band_priorities.items():
            if 0 <= band < num_bands:
                self.weights[band] = float(weight)

        self._schedule: List[int] = self._build_weighted_schedule()
        self._index: int = 0

    def _build_weighted_schedule(self) -> List[int]:
        """
        Builds a deterministic weighted round-robin interleaved schedule.
        Allocates slots proportional to weights rounded to nearest integer.
        """
        min_weight = max(0.01, float(np.min(self.weights)))
        normalized_counts = np.maximum(1, np.round(self.weights / min_weight)).astype(int)

        # Interleave channels evenly to avoid clumping
        schedule: List[int] = []
        remaining = normalized_counts.copy()
        total_slots = int(np.sum(remaining))

        for _ in range(total_slots):
            # Select band with largest remaining normalized deficit
            best_band = int(np.argmax(remaining))
            schedule.append(best_band)
            remaining[best_band] -= 1

        return schedule if schedule else list(range(self.num_bands))

    def reset(self, seed: Optional[int] = None) -> None:
        self._index = 0

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        action = self._schedule[self._index % len(self._schedule)]
        self._index += 1
        return action

    def explain(self, obs: np.ndarray, action: int) -> str:
        weight = self.weights[action]
        return (
            f"Pre-mission priority sweep: selected high-threat band {action} "
            f"(assigned threat priority weight: {weight:.1f})."
        )
