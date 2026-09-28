from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from baselines.base import BaseScheduler
from schedulers.bandits import SlidingWindowUCB


class PeriodicEmitterTracker:
    """
    Electronic Warfare Signal Processing Module:
    Estimates fundamental emission period and phase of pulse radars and scanning beams
    from sparse, asynchronously sampled intercept timestamps using circular phase coherence.
    """

    def __init__(
        self,
        min_period: int = 6,
        max_period: int = 60,
        coherence_threshold: float = 0.82,
        min_detections: int = 3,
    ) -> None:
        """
        Args:
            min_period: Minimum expected pulse/scan repetition period (time slots).
            max_period: Maximum expected repetition period.
            coherence_threshold: Circular resultant length threshold for high confidence period lock.
            min_detections: Minimum detection hits required before running period estimation.
        """
        self.min_period = min_period
        self.max_period = max_period
        self.coherence_threshold = coherence_threshold
        self.min_detections = min_detections

        # History per band
        self.detection_times: List[List[int]] = []
        self.estimated_periods: Dict[int, float] = {}
        self.estimated_phases: Dict[int, float] = {}
        self.estimated_widths: Dict[int, float] = {}
        self.coherences: Dict[int, float] = {}

    def reset(self, num_bands: int) -> None:
        """Reset internal history and tracking models for all bands."""
        self.detection_times = [[] for _ in range(num_bands)]
        self.estimated_periods.clear()
        self.estimated_phases.clear()
        self.estimated_widths.clear()
        self.coherences.clear()

    def record_observation(self, band: int, time_slot: int, detected: bool) -> None:
        """Record receiver dwell outcome on specified band."""
        if detected:
            self.detection_times[band].append(time_slot)
            # Re-estimate period if enough detections have accumulated
            if len(self.detection_times[band]) >= self.min_detections:
                self._estimate_period(band)

    def _estimate_period(self, band: int) -> None:
        """
        Estimate period T and phase phi using Epoch Folding / Circular Phase Coherence:
        R(T) = |(1/K) sum_{k} exp(i * 2pi * t_k / T)|
        """
        times = np.array(self.detection_times[band], dtype=np.float64)
        if len(times) < self.min_detections:
            return

        candidate_periods = np.arange(self.min_period, self.max_period + 1, dtype=np.float64)
        best_period = None
        best_coherence = -1.0
        best_phase = 0.0

        for T in candidate_periods:
            # Complex phasor representation
            phases = 2.0 * np.pi * (times % T) / T
            mean_vector = np.mean(np.exp(1j * phases))
            coherence = float(np.abs(mean_vector))

            if coherence > best_coherence:
                best_coherence = coherence
                best_period = T
                # Phase offset in time slots
                circular_mean_phase = float(np.angle(mean_vector))
                if circular_mean_phase < 0:
                    circular_mean_phase += 2.0 * np.pi
                best_phase = (circular_mean_phase / (2.0 * np.pi)) * T

        if best_period is not None and best_coherence >= self.coherence_threshold:
            self.estimated_periods[band] = best_period
            self.estimated_phases[band] = best_phase
            self.coherences[band] = best_coherence

    def predict_burst_imminent(self, band: int, current_time: int, window_lead: int = 1) -> bool:
        """
        Determines whether an estimated periodic burst or scanning mainlobe
        is actively occurring or expected to arrive within window_lead slots.
        """
        if band not in self.estimated_periods:
            return False

        T = self.estimated_periods[band]
        phi = self.estimated_phases[band]

        # Phase offset relative to current time
        phase_in_cycle = (current_time - phi) % T
        # Imminent if near start of cycle (or wrap-around)
        return (phase_in_cycle <= window_lead) or (phase_in_cycle >= (T - window_lead))


class PeriodicPredictiveScheduler(BaseScheduler):
    """
    Intelligent Hybrid EW Scheduler:
    Combines Sliding-Window UCB exploratory band scheduling with a Signal Processing
    Periodic-Emitter Tracker. When a periodic pulse or rotating radar beam is forecasted
    to illuminate, receiver dwells are precisely synchronized to intercept the arrival.
    """

    def __init__(
        self,
        num_bands: int,
        window_size: int = 50,
        lead_slots: int = 1,
    ) -> None:
        super().__init__(num_bands=num_bands, name="Periodic-Predictive ML")
        self.tracker = PeriodicEmitterTracker()
        self.fallback_bandit = SlidingWindowUCB(num_bands=num_bands, window_size=window_size)
        self.lead_slots = lead_slots
        self.current_t = 0
        self.last_choice_reason: str = ""

    def reset(self, seed: Optional[int] = None) -> None:
        self.tracker.reset(self.num_bands)
        self.fallback_bandit.reset(seed)
        self.current_t = 0
        self.last_choice_reason = ""

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        self.current_t += 1

        # Check if any periodic or scanning emitter is predicted to illuminate right now
        imminent_bands = []
        for b in range(self.num_bands):
            if self.tracker.predict_burst_imminent(b, self.current_t, window_lead=self.lead_slots):
                period = self.tracker.estimated_periods[b]
                coherence = self.tracker.coherences[b]
                imminent_bands.append((b, period, coherence))

        if imminent_bands:
            # Sort by highest confidence coherence
            imminent_bands.sort(key=lambda x: x[2], reverse=True)
            chosen_band, period, coherence = imminent_bands[0]
            self.last_choice_reason = (
                f"Periodic-Predictive ML intercepted band {chosen_band}: predicted arrival "
                f"of periodic emitter (period T={period:.1f} slots, coherence={coherence:.2f})."
            )
            return chosen_band

        # Fallback to bandit for agile hopping emitters and exploration
        action = self.fallback_bandit.act(obs, info)
        self.last_choice_reason = f"Periodic-Predictive ML exploration/bandit fallback -> {self.fallback_bandit.explain(obs, action)}"
        return action

    def update(
        self,
        action: int,
        reward: float,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        detected = bool(info.get("detected", False)) if info else (reward > 0)
        # Update tracker
        self.tracker.record_observation(action, self.current_t, detected)
        # Update fallback bandit
        self.fallback_bandit.update(action, reward, obs, info)

    def explain(self, obs: np.ndarray, action: int) -> str:
        return self.last_choice_reason
