from typing import Optional, Dict, Any, List
import numpy as np

from drishti.baselines.base import Scheduler
from drishti.schedulers.bandit.sliding_window_ucb import SlidingWindowUCB


class HierarchicalScanScheduler(Scheduler):
    """
    Two-Tier Hierarchical Coarse-to-Fine Frequency Scan Scheduler.

    Partitions B narrowband frequency channels into K coarse sectors (e.g., sub-octave banks).
    - Tier 1 (Coarse Level): Dynamically selects the most active sector using a
      sliding-window multi-armed bandit.
    - Tier 2 (Fine Level): Pinpoints the specific channel within the chosen sector
      based on localized Age-of-Information (AoI) urgency and pre-mission threat weights.
    """

    def __init__(
        self,
        num_bands: int = 16,
        bands_per_sector: int = 4,
        window_size: int = 50,
        prior_weights: Optional[Dict[int, float]] = None,
        name: str = "Hierarchical Coarse-Fine Scheduler",
    ) -> None:
        super().__init__(num_bands=num_bands, name=name)
        self.bands_per_sector = max(1, int(bands_per_sector))
        self.num_sectors = int(np.ceil(num_bands / self.bands_per_sector))
        self.prior_weights = prior_weights or {}

        # Sector-level coarse bandit
        self.coarse_bandit = SlidingWindowUCB(
            num_bands=self.num_sectors,
            window_size=window_size,
            exploration_coef=0.6,
            aoi_weight=0.5,
            name="Coarse-Sector UCB",
        )
        self.last_sector: int = 0
        self.rng = np.random.default_rng(42)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.coarse_bandit.reset(seed=seed)
        self.last_action = None
        self.last_sector = 0
        self.last_explanation = {}

    def choose_action(self, observation: np.ndarray) -> int:
        B = self.num_bands
        norm_tau = observation[0:B] if len(observation) >= B else np.zeros(B)

        # 1. Compute aggregate Sector-level observation
        sector_tau = np.zeros(self.num_sectors, dtype=np.float32)
        for s in range(self.num_sectors):
            b_start = s * self.bands_per_sector
            b_end = min(B, b_start + self.bands_per_sector)
            sector_tau[s] = float(np.mean(norm_tau[b_start:b_end]))

        # Sector observation dummy vector for coarse bandit
        sector_obs = np.zeros(4 * self.num_sectors + 1, dtype=np.float32)
        sector_obs[0 : self.num_sectors] = sector_tau

        # 2. Tier 1: Select coarse sector
        sector = self.coarse_bandit.act(sector_obs)
        self.last_sector = sector

        # 3. Tier 2: Select fine sub-band within the sector
        b_start = sector * self.bands_per_sector
        b_end = min(B, b_start + self.bands_per_sector)
        sub_bands = list(range(b_start, b_end))

        # Score fine sub-bands by Age-of-Information * prior threat
        fine_scores = []
        for b in sub_bands:
            prior = self.prior_weights.get(b, 1.0)
            score = (1.0 + norm_tau[b]) * prior
            fine_scores.append(score)

        chosen_band = sub_bands[int(np.argmax(fine_scores))]
        self.last_action = chosen_band

        reason = (
            f"Hierarchical: Tier 1 selected Sector {sector} (bands {b_start}-{b_end-1}) "
            f"via Coarse UCB; Tier 2 targeted Band {chosen_band} via localized AoI."
        )

        self.last_explanation = {
            "scheduler": self.name,
            "chosen_band": chosen_band,
            "chosen_sector": sector,
            "reason": reason,
            "scores": {b: float(norm_tau[b]) for b in range(B)},
        }

        return chosen_band

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Propagates reward feedback to Tier 1 coarse bandit."""
        sector = action // self.bands_per_sector
        self.coarse_bandit.update(obs=obs, action=sector, reward=reward, info=info)
