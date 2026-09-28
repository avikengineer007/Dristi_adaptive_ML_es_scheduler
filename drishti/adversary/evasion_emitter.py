from typing import Optional, List, Dict, Any
from collections import deque
import numpy as np

from drishti.env.emitters import Emitter


class AdversarialEvasionEmitter(Emitter):
    """
    Cognitive Electronic Protection (EP) Evasion Radar.

    Models an intelligent hostile radar system that tracks the ES receiver's
    revisit patterns and dynamically hops its carrier frequency to the band
    where the ES receiver is least likely to tune.

    Min-Max Evasion Strategy:
    1. Records the sequence of frequency bands tuned by the ES receiver.
    2. Estimates the receiver's empirical tuning distribution over a sliding window.
    3. Hops to the candidate band with the lowest receiver dwell probability,
       optionally injecting controlled stochastic exploration to prevent exploitation.
    """

    def __init__(
        self,
        emitter_id: str,
        hop_bands: List[int],
        window_size: int = 30,
        evasion_greediness: float = 0.85,
        threat_weight: float = 9.0,
        power_dbm: float = 38.0,
    ) -> None:
        super().__init__(emitter_id, threat_weight, power_dbm)
        assert len(hop_bands) >= 2, "Adversarial emitter requires at least 2 hop bands."
        self.hop_bands = list(hop_bands)
        self.window_size = int(window_size)
        self.evasion_greediness = float(evasion_greediness)

        # Sliding window of observed receiver actions
        self.receiver_history: deque = deque(maxlen=self.window_size)
        self.current_band: int = self.hop_bands[0]

    def reset(self, rng: np.random.Generator) -> None:
        self.receiver_history.clear()
        self.current_band = int(rng.choice(self.hop_bands))

    def observe_receiver_action(self, receiver_band: int) -> None:
        """Informs the adversary of the channel the ES receiver tuned to."""
        self.receiver_history.append(int(receiver_band))

    def get_emission(self, t: int, rng: np.random.Generator) -> Optional[int]:
        """
        Selects next carrier frequency to evade the ES receiver.
        """
        if not self.receiver_history or rng.random() > self.evasion_greediness:
            # Random exploration hop among candidate bands
            self.current_band = int(rng.choice(self.hop_bands))
            return self.current_band

        # Calculate receiver empirical dwell frequency on each candidate band
        counts = {b: 0 for b in self.hop_bands}
        for b_rec in self.receiver_history:
            if b_rec in counts:
                counts[b_rec] += 1

        total_obs = max(1, len(self.receiver_history))
        probabilities = {b: counts[b] / total_obs for b in self.hop_bands}

        # Min-Max evasion: Select the band the receiver has visited least
        min_prob = min(probabilities.values())
        safest_bands = [b for b, p in probabilities.items() if p == min_prob]

        # Break ties with random choice among safest bands
        self.current_band = int(rng.choice(safest_bands))
        return self.current_band
