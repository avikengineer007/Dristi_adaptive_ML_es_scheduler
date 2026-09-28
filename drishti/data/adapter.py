from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import os
import numpy as np
import pandas as pd

from drishti.env.emitters import (
    Emitter,
    FixedEmitter,
    PeriodicBurstEmitter,
    FrequencyAgileEmitter,
    ScanningEmitter,
    PeriodicScanReceiverTarget,
)


class DataSource(ABC):
    """Abstract interface for RF spectrum scenario data sources."""

    @abstractmethod
    def get_emitters(self, num_bands: int, rng: np.random.Generator) -> List[Emitter]:
        """Returns the list of active emitters for an episode."""
        pass


class SyntheticSource(DataSource):
    """
    Config-driven synthetic emitter scenario generator.
    Parses YAML scenario specifications and instantiates physics-based emitter models.
    """

    def __init__(self, emitter_specs: Optional[List[Dict[str, Any]]] = None) -> None:
        self.emitter_specs = emitter_specs if emitter_specs is not None else []

    def get_emitters(self, num_bands: int, rng: np.random.Generator) -> List[Emitter]:
        if not self.emitter_specs:
            return self._build_default_mix(num_bands)

        emitters: List[Emitter] = []
        for spec in self.emitter_specs:
            e_type = spec.get("type", "").lower()
            e_id = spec.get("id", f"EMITTER_{len(emitters)}")
            threat = float(spec.get("threat_weight", 1.0))
            power = float(spec.get("power_dbm", 30.0))

            if e_type == "fixed":
                band = min(num_bands - 1, int(spec.get("band", 0)))
                active_ratio = float(spec.get("active_ratio", 1.0))
                emitters.append(FixedEmitter(e_id, band, threat, power, active_ratio))

            elif e_type == "periodic_burst":
                band = min(num_bands - 1, int(spec.get("band", 0)))
                period = int(spec.get("period", 20))
                on_time = int(spec.get("on_time", 4))
                jitter = float(spec.get("jitter", 0.0))
                phase = spec.get("phase")
                emitters.append(PeriodicBurstEmitter(e_id, band, period, on_time, phase, jitter, threat, power))

            elif e_type == "frequency_agile":
                raw_bands = spec.get("hop_bands", [0, 1, 2])
                hop_bands = [min(num_bands - 1, b) for b in raw_bands]
                hop_dwell = int(spec.get("hop_dwell", 1))
                pattern = str(spec.get("pattern", "cyclic"))
                active_ratio = float(spec.get("active_ratio", 0.85))
                emitters.append(FrequencyAgileEmitter(e_id, hop_bands, hop_dwell, pattern, None, active_ratio, threat, power))

            elif e_type == "scanning":
                raw_bands = spec.get("bands", [num_bands - 1])
                bands = [min(num_bands - 1, b) for b in raw_bands] if isinstance(raw_bands, list) else min(num_bands - 1, int(raw_bands))
                scan_period = int(spec.get("scan_period", 40))
                beamwidth = int(spec.get("beamwidth_slots", 3))
                phase = spec.get("phase")
                emitters.append(ScanningEmitter(e_id, bands, scan_period, beamwidth, phase, threat, power))

            elif e_type == "periodic_scan_receiver_target":
                raw_bands = spec.get("listening_bands", [min(num_bands - 1, b) for b in [1, 3, 5, 7]])
                listening_bands = [min(num_bands - 1, b) for b in raw_bands]
                dwell_per_band = int(spec.get("dwell_per_band", 4))
                phase = spec.get("phase")
                emitters.append(PeriodicScanReceiverTarget(e_id, listening_bands, dwell_per_band, 0.9, phase, threat, power))

        return emitters

    def _build_default_mix(self, num_bands: int) -> List[Emitter]:
        """Fallback default emitter mixture."""
        b_fixed = max(0, min(num_bands - 1, 1))
        b_b1 = max(0, min(num_bands - 1, int(0.25 * num_bands)))
        b_b2 = max(0, min(num_bands - 1, int(0.65 * num_bands)))
        hop_bands = [min(num_bands - 1, int(f * num_bands)) for f in [0.4, 0.5, 0.6]]
        b_scan = max(0, num_bands - 1)

        return [
            FixedEmitter("FIXED_SURV", b_fixed, threat_weight=2.0),
            PeriodicBurstEmitter("BURST_RADAR_1", b_b1, period=16, on_time=3, threat_weight=5.0),
            PeriodicBurstEmitter("BURST_RADAR_2", b_b2, period=30, on_time=4, threat_weight=6.0),
            FrequencyAgileEmitter("AGILE_HOPPER", hop_bands, hop_dwell=2, threat_weight=8.0),
            ScanningEmitter("SCAN_SEARCH_RADAR", b_scan, scan_period=40, beamwidth_slots=3, threat_weight=10.0),
        ]


class ReplayEmitter(Emitter):
    """Plays back pre-recorded emissions from offline datasets."""

    def __init__(self, emitter_id: str, active_slots: Dict[int, int], threat_weight: float = 1.0, power_dbm: float = 30.0) -> None:
        super().__init__(emitter_id, threat_weight, power_dbm)
        self.active_slots = active_slots  # time_slot -> band

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def get_emission(self, t: int, rng: np.random.Generator) -> Optional[int]:
        return self.active_slots.get(t, None)


class CSVSource(DataSource):
    """
    Dataset adapter for real-world and pre-recorded spectrum captures.
    
    Expected CSV Columns:
    --------------------
    - time_slot: int (discrete time step index, t >= 0)
    - band: int (frequency channel index, 0 <= band < num_bands)
    - power_dbm: float (received signal power in dBm, e.g. -45.0)
    - emitter_id: str (unique identifier or classification label of emitter)
    - threat_weight: float (optional priority weight, defaults to 1.0)
    """

    def __init__(self, csv_filepath: str) -> None:
        self.csv_filepath = csv_filepath

    def get_emitters(self, num_bands: int, rng: np.random.Generator) -> List[Emitter]:
        if not os.path.exists(self.csv_filepath):
            raise FileNotFoundError(f"Recorded spectrum dataset not found at: {self.csv_filepath}")

        df = pd.read_csv(self.csv_filepath)
        required_cols = {"time_slot", "band", "emitter_id"}
        assert required_cols.issubset(df.columns), f"CSV missing required columns: {required_cols - set(df.columns)}"

        emitters: List[Emitter] = []
        for e_id, group in df.groupby("emitter_id"):
            active_slots = dict(zip(group["time_slot"].astype(int), group["band"].astype(int)))
            threat = float(group["threat_weight"].iloc[0]) if "threat_weight" in group.columns else 1.0
            power = float(group["power_dbm"].iloc[0]) if "power_dbm" in group.columns else 30.0
            emitters.append(ReplayEmitter(str(e_id), active_slots, threat_weight=threat, power_dbm=power))

        return emitters
