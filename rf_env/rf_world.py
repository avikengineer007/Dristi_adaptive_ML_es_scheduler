from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from rf_env.emitters import BaseEmitter


@dataclass
class BurstEvent:
    """Represents a continuous transmission burst from an emitter."""
    emitter_id: str
    band: int
    start_time: int
    end_time: int  # exclusive
    threat_weight: float
    intercepted: bool = False
    first_intercept_time: Optional[int] = None


@dataclass
class ChannelObservation:
    """The RF observation outcome when the ES receiver dwells on a band."""
    band: int
    time_slot: int
    detected: bool
    is_false_alarm: bool
    is_true_detection: bool
    detected_emitters: List[str]
    received_power: float
    threat_value: float


class RFWorld:
    """
    Simulates the physical RF spectrum environment, manages emitter interactions,
    and logs the ground truth matrix for full reproducibility and metric verification.
    """

    def __init__(
        self,
        num_bands: int = 16,
        emitters: Optional[List[BaseEmitter]] = None,
        noise_floor_dbm: float = -95.0,
        p_fa: float = 0.02,
        p_md: float = 0.05,
    ) -> None:
        """
        Args:
            num_bands: Total number of discrete frequency channels/bands (N).
            emitters: List of emitter instances active in the scenario.
            noise_floor_dbm: Baseline thermal noise floor in dBm.
            p_fa: Receiver false-alarm probability when no signal is present.
            p_md: Receiver missed-detection probability when signal is present.
        """
        self.num_bands = num_bands
        self.emitters: List[BaseEmitter] = emitters if emitters is not None else []
        self.noise_floor_dbm = noise_floor_dbm
        self.p_fa = p_fa
        self.p_md = p_md

        # Ground truth tracking
        self.ground_truth_matrix: Optional[np.ndarray] = None  # Shape (T, N), bool
        self.ground_truth_threats: Optional[np.ndarray] = None  # Shape (T, N), float
        self.burst_events: List[BurstEvent] = []
        self._current_step: int = 0
        self._active_bursts: Dict[str, BurstEvent] = {}

    def add_emitter(self, emitter: BaseEmitter) -> None:
        """Add an emitter to the RF world."""
        self.emitters.append(emitter)

    def reset(self, rng: np.random.Generator, max_steps: int = 1000) -> None:
        """
        Reset all emitters, generate initial phases, and allocate ground truth logs.
        """
        self._current_step = 0
        self.burst_events.clear()
        self._active_bursts.clear()

        # Reset all emitter states
        for emitter in self.emitters:
            emitter.reset(rng)

        # Allocate ground truth storage
        self.ground_truth_matrix = np.zeros((max_steps, self.num_bands), dtype=bool)
        self.ground_truth_threats = np.zeros((max_steps, self.num_bands), dtype=np.float32)

    def step_world(self, t: int, rng: np.random.Generator) -> Dict[int, List[BaseEmitter]]:
        """
        Evaluate all emitters at time step t and update ground truth.

        Returns:
            Dictionary mapping band index to list of active emitters on that band.
        """
        self._current_step = t
        active_on_bands: Dict[int, List[BaseEmitter]] = {b: [] for b in range(self.num_bands)}

        active_emitter_ids_this_step = set()

        for emitter in self.emitters:
            band = emitter.get_emission(t, rng)
            if band is not None and 0 <= band < self.num_bands:
                active_on_bands[band].append(emitter)
                active_emitter_ids_this_step.add(emitter.emitter_id)

                # Ground truth logging
                if self.ground_truth_matrix is not None and t < self.ground_truth_matrix.shape[0]:
                    self.ground_truth_matrix[t, band] = True
                    self.ground_truth_threats[t, band] = max(
                        self.ground_truth_threats[t, band], emitter.threat_weight
                    )

                # Track continuous burst events
                if emitter.emitter_id not in self._active_bursts:
                    event = BurstEvent(
                        emitter_id=emitter.emitter_id,
                        band=band,
                        start_time=t,
                        end_time=t + 1,
                        threat_weight=emitter.threat_weight,
                    )
                    self._active_bursts[emitter.emitter_id] = event
                    self.burst_events.append(event)
                else:
                    event = self._active_bursts[emitter.emitter_id]
                    # If agile emitter hopped to another band, close previous burst and open new one
                    if event.band != band:
                        event.end_time = t
                        new_event = BurstEvent(
                            emitter_id=emitter.emitter_id,
                            band=band,
                            start_time=t,
                            end_time=t + 1,
                            threat_weight=emitter.threat_weight,
                        )
                        self._active_bursts[emitter.emitter_id] = new_event
                        self.burst_events.append(new_event)
                    else:
                        event.end_time = t + 1

        # Close bursts for inactive emitters
        closed_emitters = [
            eid for eid in self._active_bursts if eid not in active_emitter_ids_this_step
        ]
        for eid in closed_emitters:
            self._active_bursts[eid].end_time = t
            del self._active_bursts[eid]

        return active_on_bands

    def observe_band(
        self,
        band: int,
        t: int,
        active_on_bands: Dict[int, List[BaseEmitter]],
        rng: np.random.Generator,
    ) -> ChannelObservation:
        """
        Simulate receiver tuning to a specific band at time step t.
        Accounts for missed detections (P_md) and false alarms (P_fa).
        """
        active_emitters = active_on_bands.get(band, [])
        is_signal_present = len(active_emitters) > 0

        detected = False
        is_false_alarm = False
        is_true_detection = False
        detected_emitters: List[str] = []
        threat_value = 0.0
        received_power = self.noise_floor_dbm

        if is_signal_present:
            # Check detection probability
            if rng.random() > self.p_md:
                detected = True
                is_true_detection = True
                detected_emitters = [e.emitter_id for e in active_emitters]
                threat_value = sum(e.threat_weight for e in active_emitters)
                max_power = max(e.power_dbm for e in active_emitters)
                received_power = max_power + float(rng.normal(0, 1.0))

                # Mark burst events as intercepted
                for e in active_emitters:
                    if e.emitter_id in self._active_bursts:
                        burst = self._active_bursts[e.emitter_id]
                        if not burst.intercepted:
                            burst.intercepted = True
                            burst.first_intercept_time = t
        else:
            # Check false alarm probability
            if rng.random() < self.p_fa:
                detected = True
                is_false_alarm = True
                received_power = self.noise_floor_dbm + float(rng.exponential(3.0))

        return ChannelObservation(
            band=band,
            time_slot=t,
            detected=detected,
            is_false_alarm=is_false_alarm,
            is_true_detection=is_true_detection,
            detected_emitters=detected_emitters,
            received_power=received_power,
            threat_value=threat_value,
        )
