from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np


class BaseEmitter(ABC):
    """Abstract base class for all RF emitters in the EW spectrum environment."""

    def __init__(
        self,
        emitter_id: str,
        threat_weight: float = 1.0,
        power_dbm: float = 20.0,
    ) -> None:
        """
        Args:
            emitter_id: Unique string identifier for the emitter.
            threat_weight: Relative threat level/priority (higher = more critical).
            power_dbm: Transmit power level proxy.
        """
        self.emitter_id = emitter_id
        self.threat_weight = float(threat_weight)
        self.power_dbm = float(power_dbm)

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        """Reset internal emitter state using the provided RNG generator."""
        pass

    @abstractmethod
    def get_emission(self, t: int, rng: np.random.Generator) -> Optional[int]:
        """
        Determine if the emitter is transmitting at time step t.

        Returns:
            The band index (0 to N-1) if transmitting, or None if inactive/silent.
        """
        pass


class FixedEmitter(BaseEmitter):
    """
    Fixed-frequency emitter transmitting continuously or with a fixed periodic duty cycle.
    Represents conventional CW or steady radar/comms transmitters.
    """

    def __init__(
        self,
        emitter_id: str,
        band: int,
        threat_weight: float = 2.0,
        power_dbm: float = 25.0,
        active_ratio: float = 1.0,
    ) -> None:
        """
        Args:
            band: Fixed frequency band index.
            active_ratio: Fraction of time active (1.0 = continuous transmission).
        """
        super().__init__(emitter_id, threat_weight, power_dbm)
        self.band = band
        self.active_ratio = active_ratio

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def get_emission(self, t: int, rng: np.random.Generator) -> Optional[int]:
        if self.active_ratio >= 1.0:
            return self.band
        # If stochastic duty cycle
        if rng.random() < self.active_ratio:
            return self.band
        return None


class PeriodicBurstEmitter(BaseEmitter):
    """
    Periodic pulsed/burst emitter on a fixed band with known period and burst duration.
    Represents pulsed radars or burst communications.
    Active when ((t + phase) % period) < burst_duration.
    """

    def __init__(
        self,
        emitter_id: str,
        band: int,
        period: int,
        burst_duration: int,
        phase: Optional[int] = None,
        threat_weight: float = 4.0,
        power_dbm: float = 30.0,
    ) -> None:
        """
        Args:
            band: Center band of the burst transmission.
            period: Repeat interval in time slots (T_period > burst_duration).
            burst_duration: Active burst length in time slots (T_on).
            phase: Optional fixed starting phase offset; if None, randomized on reset.
        """
        super().__init__(emitter_id, threat_weight, power_dbm)
        assert period > burst_duration, "Period must exceed burst duration."
        assert burst_duration > 0, "Burst duration must be positive."
        self.band = band
        self.period = period
        self.burst_duration = burst_duration
        self._fixed_phase = phase
        self.phase = phase if phase is not None else 0

    def reset(self, rng: np.random.Generator) -> None:
        if self._fixed_phase is None:
            self.phase = int(rng.integers(0, self.period))
        else:
            self.phase = self._fixed_phase

    def get_emission(self, t: int, rng: np.random.Generator) -> Optional[int]:
        if ((t + self.phase) % self.period) < self.burst_duration:
            return self.band
        return None


class FrequencyAgileEmitter(BaseEmitter):
    """
    Frequency-hopping emitter that changes carrier band every hop_dwell slots.
    Represents agile electronic countermeasures (ECM) or frequency-hopping spread spectrum (FHSS).
    """

    def __init__(
        self,
        emitter_id: str,
        hop_bands: List[int],
        hop_dwell: int = 1,
        active_ratio: float = 0.9,
        threat_weight: float = 7.0,
        power_dbm: float = 35.0,
    ) -> None:
        """
        Args:
            hop_bands: Available set of frequency bands to hop across.
            hop_dwell: Number of discrete time slots the emitter dwells on each band.
            active_ratio: Probability of transmitting during an active hop slot.
        """
        super().__init__(emitter_id, threat_weight, power_dbm)
        assert len(hop_bands) > 0, "Must specify at least one hop band."
        assert hop_dwell > 0, "Hop dwell must be at least 1 slot."
        self.hop_bands = list(hop_bands)
        self.hop_dwell = hop_dwell
        self.active_ratio = active_ratio
        self._current_band_idx = 0
        self._hop_schedule: List[int] = []

    def reset(self, rng: np.random.Generator) -> None:
        # Generate a pseudo-random hop permutation table
        self._hop_schedule = list(rng.permutation(self.hop_bands))
        self._current_band_idx = int(rng.integers(0, len(self._hop_schedule)))

    def get_emission(self, t: int, rng: np.random.Generator) -> Optional[int]:
        if not self._hop_schedule:
            self.reset(rng)
        
        # Advance hop index based on dwell time
        slot_in_dwell = t // self.hop_dwell
        band = self._hop_schedule[slot_in_dwell % len(self._hop_schedule)]

        if rng.random() < self.active_ratio:
            return band
        return None


class PeriodicScanEmitter(BaseEmitter):
    """
    Spatially scanning beam radar emitter (e.g., rotating search radar).
    Illuminates the ES receiver direction with a scanning antenna mainlobe
    once every scan_period slots for a duration of beam_width slots.
    """

    def __init__(
        self,
        emitter_id: str,
        band: int,
        scan_period: int,
        beam_width: int,
        phase: Optional[int] = None,
        threat_weight: float = 8.0,
        power_dbm: float = 40.0,
    ) -> None:
        """
        Args:
            band: Radar carrier frequency band.
            scan_period: Mechanical/electronic rotation period in time slots.
            beam_width: Mainlobe illumination dwell on the receiver in time slots.
            phase: Optional fixed starting scan angle/phase offset.
        """
        super().__init__(emitter_id, threat_weight, power_dbm)
        assert scan_period > beam_width, "Scan period must exceed beam width."
        assert beam_width > 0, "Beam width must be at least 1 slot."
        self.band = band
        self.scan_period = scan_period
        self.beam_width = beam_width
        self._fixed_phase = phase
        self.phase = phase if phase is not None else 0

    def reset(self, rng: np.random.Generator) -> None:
        if self._fixed_phase is None:
            self.phase = int(rng.integers(0, self.scan_period))
        else:
            self.phase = self._fixed_phase

    def get_emission(self, t: int, rng: np.random.Generator) -> Optional[int]:
        if ((t + self.phase) % self.scan_period) < self.beam_width:
            return self.band
        return None
