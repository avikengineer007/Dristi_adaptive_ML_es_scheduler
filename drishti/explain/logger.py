from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import json
import os
from typing import Dict, Any, List, Optional
from collections import deque


@dataclass
class DecisionRecord:
    """Represents a single explained frequency tuning decision."""
    slot: int
    chosen_band: int
    scheduler_name: str
    reason: str
    scores: Dict[int, float] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    detected: bool = False
    reward: float = 0.0
    timestamp_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExplanationLogger:
    """
    Unified decision logger maintaining in-memory records and providing
    structured JSONL export and contrastive explainability for mission operators.
    """

    def __init__(self, max_records: int = 1000) -> None:
        self.max_records = max_records
        self.records: deque = deque(maxlen=max_records)

    def log(
        self,
        slot: int,
        chosen_band: int,
        scheduler_name: str,
        reason: str,
        scores: Optional[Dict[int, float]] = None,
        components: Optional[Dict[str, Any]] = None,
        detected: bool = False,
        reward: float = 0.0,
    ) -> DecisionRecord:
        """Creates and buffers a new explained decision record."""
        record = DecisionRecord(
            slot=slot,
            chosen_band=chosen_band,
            scheduler_name=scheduler_name,
            reason=reason,
            scores=scores or {},
            components=components or {},
            detected=detected,
            reward=reward,
        )
        self.records.append(record)
        return record

    def clear(self) -> None:
        """Clears all buffered records."""
        self.records.clear()

    def get_recent(self, n: int = 50, newest_first: bool = True) -> List[Dict[str, Any]]:
        """Returns the n most recent decision records as dictionaries."""
        n_items = min(n, len(self.records))
        if n_items == 0:
            return []
        if newest_first:
            return [self.records[-i].to_dict() for i in range(1, n_items + 1)]
        return [self.records[-i].to_dict() for i in range(n_items, 0, -1)]

    def export_jsonl(self, filepath: str) -> str:
        """Exports buffered records to a newline-delimited JSON (JSONL) file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec.to_dict()) + "\n")
        return filepath

    def explain_contrastive(
        self,
        band_chosen: int,
        band_alternative: int,
        record: Optional[DecisionRecord] = None,
    ) -> str:
        """
        Generates a natural-language contrastive explanation answering:
        'Why was band_chosen selected over band_alternative?'
        """
        rec = record or (self.records[-1] if self.records else None)
        if rec is None:
            return "No decision records available to explain."

        score_chosen = rec.scores.get(band_chosen, 0.0)
        score_alt = rec.scores.get(band_alternative, 0.0)
        delta = score_chosen - score_alt

        reasons = [
            f"Band {band_chosen} was selected over Band {band_alternative} with a priority advantage of +{delta:.3f}."
        ]

        # Inspect component attributions if available
        comp = rec.components
        if "aoi_bonuses" in comp:
            aoi_chosen = comp["aoi_bonuses"].get(band_chosen, 0.0)
            aoi_alt = comp["aoi_bonuses"].get(band_alternative, 0.0)
            if aoi_chosen > aoi_alt:
                reasons.append(f"Age-of-Information urgency is higher on Band {band_chosen} ({aoi_chosen:.2f} vs {aoi_alt:.2f}).")

        if "empirical_means" in comp:
            mean_chosen = comp["empirical_means"].get(band_chosen, 0.0)
            mean_alt = comp["empirical_means"].get(band_alternative, 0.0)
            if mean_chosen > mean_alt:
                reasons.append(f"Empirical reward history is stronger on Band {band_chosen} ({mean_chosen:.2f} vs {mean_alt:.2f}).")

        if "expected_probs" in comp:
            p_chosen = comp["expected_probs"].get(band_chosen, 0.0)
            p_alt = comp["expected_probs"].get(band_alternative, 0.0)
            reasons.append(f"Posterior detection probability: Band {band_chosen} ({p_chosen:.2f}) vs Band {band_alternative} ({p_alt:.2f}).")

        return " ".join(reasons)
