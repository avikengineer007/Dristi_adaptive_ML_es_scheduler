from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import numpy as np


class CircularPhaseCoherenceEstimator:
    """
    Sparse-sample period and phase estimator using Circular Phase Coherence (Epoch Folding).

    In EW operations, an ES receiver samples channels asynchronously and sparsely,
    leaving large gaps between detection events. Traditional FFTs and autocorrelations
    fail because the signal is not regularly sampled in continuous time.

    Epoch Folding / Circular Phase Coherence evaluates candidate trial periods T:
      theta_k(T) = 2 * pi * (t_k % T) / T
      R(T) = (1 / K) * | sum_{k=1}^K exp(j * theta_k(T)) |

    When T matches the true emitter repetition period T_true:
      All pulses fold into the exact same relative phase angle, yielding R(T) ~ 1.0.
    When T != T_true:
      Phases distribute uniformly around the unit circle, yielding R(T) ~ 0.0.
    """

    def __init__(
        self,
        min_period: int = 5,
        max_period: int = 150,
        coherence_threshold: float = 0.75,
        min_detections: int = 3,
        max_history: int = 100,
    ) -> None:
        self.min_period = max(2, int(min_period))
        self.max_period = max(self.min_period + 1, int(max_period))
        self.coherence_threshold = float(coherence_threshold)
        self.min_detections = int(min_detections)
        self.max_history = int(max_history)

        self.timestamps: deque = deque(maxlen=self.max_history)
        self.estimated_period: Optional[int] = None
        self.estimated_phase: Optional[int] = None
        self.coherence_score: float = 0.0
        self.estimated_on_time: int = 1

    def reset(self) -> None:
        """Clears observation history and cached estimates."""
        self.timestamps.clear()
        self.estimated_period = None
        self.estimated_phase = None
        self.coherence_score = 0.0
        self.estimated_on_time = 1

    def add_observation(self, time_slot: int) -> None:
        """Records a new signal detection timestamp."""
        if not self.timestamps or time_slot > self.timestamps[-1]:
            self.timestamps.append(int(time_slot))
            self._update_estimates()

    def _update_estimates(self) -> None:
        """Re-evaluates circular phase coherence across candidate periods."""
        K = len(self.timestamps)
        if K < self.min_detections:
            self.estimated_period = None
            self.estimated_phase = None
            self.coherence_score = 0.0
            return

        ts = np.array(self.timestamps, dtype=np.float64)

        # Pre-filter candidate periods using Difference Histogram GCD heuristics
        diffs = np.diff(ts)
        min_diff = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else self.min_period

        candidate_periods = range(self.min_period, min(self.max_period, int(ts[-1] - ts[0]) + 1))
        if not candidate_periods:
            return

        R_scores: Dict[int, float] = {}
        for T in candidate_periods:
            # Circular phase angles
            phases = 2.0 * np.pi * (ts % T) / T
            cos_sum = np.sum(np.cos(phases))
            sin_sum = np.sum(np.sin(phases))
            R = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / K
            R_scores[T] = float(R)

        valid_candidates = [T for T in candidate_periods if R_scores[T] >= self.coherence_threshold]
        if not valid_candidates:
            self.coherence_score = float(max(R_scores.values())) if R_scores else 0.0
            self.estimated_period = None
            self.estimated_phase = None
            return

        # Resolve subharmonics: pick highest period among top scoring periods
        # whose period does not exceed the median inter-arrival gap
        best_R = max(R_scores[T] for T in valid_candidates)
        self.coherence_score = best_R

        top_candidates = [T for T in valid_candidates if R_scores[T] >= best_R - 0.04]
        diffs = np.diff(ts)
        pos_diffs = diffs[diffs > 0]
        median_diff = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else float(self.max_period)

        eligible = [T for T in top_candidates if T <= median_diff * 1.25]
        best_T = max(eligible) if eligible else top_candidates[0]

        self.estimated_period = int(best_T)

        # Estimate phase offset by finding the modal folded bin
        folded_slots = (ts % best_T).astype(int)
        counts = np.bincount(folded_slots, minlength=best_T)
        modal_slot = int(np.argmax(counts))

        # Estimate burst on_time from width of folded cluster
        active_folded_slots = np.where(counts > 0)[0]
        if len(active_folded_slots) > 1:
            spread = np.ptp(active_folded_slots) + 1
            self.estimated_on_time = max(1, min(spread, best_T // 2))
        else:
            self.estimated_on_time = 1

        self.estimated_phase = modal_slot

    def predict_next_arrival(self, current_slot: int) -> Optional[int]:
        """
        Predicts the earliest upcoming slot >= current_slot when emitter pulses will occur.
        """
        if self.estimated_period is None or self.estimated_phase is None:
            return None

        T = self.estimated_period
        phi = self.estimated_phase

        # Target slots in cycle: [phi, phi + on_time - 1]
        cycle_pos = current_slot % T
        time_to_phase = (phi - cycle_pos) % T
        return current_slot + time_to_phase

    def is_burst_active(self, current_slot: int, margin: int = 1) -> Tuple[bool, float]:
        """
        Returns (is_active, confidence) indicating if a burst is active at current_slot.
        """
        if self.estimated_period is None or self.estimated_phase is None:
            return False, 0.0

        T = self.estimated_period
        phi = self.estimated_phase
        dur = max(1, self.estimated_on_time) + margin

        cycle_pos = (current_slot - phi) % T
        is_active = cycle_pos < dur
        confidence = self.coherence_score
        return bool(is_active), float(confidence)


class PeriodicEmitterTracker:
    """
    Multichannel tracker maintaining individual periodicity models per frequency band,
    plus target synchronization logic for PeriodicScanReceiverTarget.
    """

    def __init__(
        self,
        num_bands: int,
        min_period: int = 5,
        max_period: int = 150,
        coherence_threshold: float = 0.72,
    ) -> None:
        self.num_bands = num_bands
        self.estimators = [
            CircularPhaseCoherenceEstimator(
                min_period=min_period,
                max_period=max_period,
                coherence_threshold=coherence_threshold,
            )
            for _ in range(num_bands)
        ]
        # Target scan tracker: history of (slot, band) detections
        self.target_detections: deque = deque(maxlen=150)
        self.target_cycle_period: Optional[int] = None
        self.target_band_sequence: List[int] = []

    def reset(self) -> None:
        for est in self.estimators:
            est.reset()
        self.target_detections.clear()
        self.target_cycle_period = None
        self.target_band_sequence = []

    def update(self, band: int, slot: int, detected: bool) -> None:
        """Updates estimators with new step feedback."""
        if detected and 0 <= band < self.num_bands:
            self.estimators[band].add_observation(slot)
            self.target_detections.append((slot, band))
            self._update_target_cycle()

    def _update_target_cycle(self) -> None:
        """Infers sequence and cycle period for periodic scanning targets."""
        if len(self.target_detections) < 4:
            return

        # Extract sequence of channel onsets (transitions into a new band)
        onsets: List[Tuple[int, int]] = []
        for i, (slot, band) in enumerate(self.target_detections):
            if i == 0 or self.target_detections[i - 1][1] != band:
                onsets.append((slot, band))

        if len(onsets) < 4:
            return

        unique_ordered = [b for _, b in onsets]

        # Look for repeat cycle in channel onset sequence
        for cycle_len in range(2, min(8, len(unique_ordered) // 2 + 1)):
            subseq = unique_ordered[-cycle_len:]
            prev_subseq = unique_ordered[-2 * cycle_len : -cycle_len]
            if subseq == prev_subseq:
                self.target_band_sequence = subseq
                # Estimate cycle duration from onsets of the sequence head
                t_starts = [t for t, b in onsets if b == subseq[0]]
                if len(t_starts) >= 2:
                    diffs = np.diff(t_starts)
                    self.target_cycle_period = int(np.median(diffs))
                break

    def get_upcoming_bursts(self, current_slot: int, horizon: int = 2) -> Dict[int, float]:
        """
        Returns a dict mapping band -> confidence for any band expected to emit within horizon slots.
        """
        upcoming = {}
        for b, est in enumerate(self.estimators):
            is_active, conf = est.is_burst_active(current_slot, margin=1)
            if is_active:
                upcoming[b] = max(upcoming.get(b, 0.0), conf * 1.5)
            else:
                next_arr = est.predict_next_arrival(current_slot)
                if next_arr is not None and (next_arr - current_slot) <= horizon:
                    upcoming[b] = max(upcoming.get(b, 0.0), conf)

        # Check target receiver cycle forecast
        if self.target_cycle_period and self.target_band_sequence:
            # Predict which channel the scanning receiver will listen to next
            last_t, last_b = self.target_detections[-1]
            elapsed = current_slot - last_t
            dwell_est = max(1, self.target_cycle_period // len(self.target_band_sequence))
            hop_step = (elapsed // dwell_est) % len(self.target_band_sequence)
            predicted_target_band = self.target_band_sequence[hop_step]
            upcoming[predicted_target_band] = max(upcoming.get(predicted_target_band, 0.0), 0.95)

        return upcoming
