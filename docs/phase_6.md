# Phase 6: Explainability and the Scheduler Service
**DRISHTI: Dynamic Reinforcement-learning Intercept Scheduler for Tactical ES Intelligence**
*SIH 2026 Problem Statement 26055 (DRDO)*

---

## 1. Tactical EW Requirement: Human-Machine Teaming & Auditability

In Electronic Warfare, black-box autonomous schedulers cannot be deployed without **verifiable explainability**:
1. **Pilot / EW Officer Trust**: Operators must understand why the receiver tuned to a specific channel at a critical mission moment (e.g. *"Did we switch to Channel 4 because a SAM radar pulse was expected, or to prevent Age-of-Information starvation?"*).
2. **Post-Mission Forensics**: Intelligence analysts require an immutable, timestamped audit log of every scheduling decision, sensor detection, and policy trade-off.
3. **Contrastive Querying**: Tactical commanders need to ask contrastive questions: *"Why did the system choose Band 3 instead of Band 8?"*
4. **Unified Service API**: Avionics software systems need a single, rock-solid entry point (`ScanScheduler`) that abstracts internal algorithm complexity, provides auto-routing, and manages fail-safes.

**Phase 6** fulfills these mission-critical requirements.

---

## 2. Explainability Framework & Forensic Logging

### 2.1 The `DecisionRecord` Schema
Implemented in [`drishti/explain/logger.py`](file:///d:/Projects/Dristi_freq/drishti/explain/logger.py):
Every decision generates a structured record containing:
- `slot`: Discrete mission time slot ($t$).
- `chosen_band`: Channel tuned by the ES receiver.
- `scheduler_name`: Active strategy (e.g. `"PPO (Augmented RL)"`, `"Periodic-Aware Scheduler"`, `"Sliding-Window UCB"`).
- `reason`: Natural-language justification.
- `scores`: Per-channel priority score or policy action probability distribution.
- `components`: Granular decomposition (Age-of-Information bonus, empirical mean reward, exploration interval, periodic confidence).
- `detected`: Binary outcome whether a true signal was detected on the tuned band.
- `reward`: Step reward received.
- `timestamp_iso`: UTC ISO-8601 forensic timestamp.

### 2.2 Forensic JSONL Export
The `ExplanationLogger` maintains an in-memory ring buffer (default 1,000 records) and exports complete mission audit trails to newline-delimited JSON (`.jsonl`) via `.export_jsonl(filepath)`:
```json
{
  "slot": 42,
  "chosen_band": 7,
  "scheduler_name": "Periodic-Aware Scheduler",
  "reason": "Predictive Synchronization: Periodic emission forecasted on band 7 (T=40, confidence=0.88).",
  "scores": {"0": 0.12, "7": 0.88},
  "components": {"aoi_bonuses": {"7": 0.35}, "coherence": 0.88},
  "detected": true,
  "reward": 10.0,
  "timestamp_iso": "2026-09-28T15:47:00.123456+00:00"
}
```

### 2.3 Contrastive Explanation Engine
Implements dynamic contrastive reasoning:
```python
logger.explain_contrastive(band_chosen=3, band_alternative=1)
```
Output:
> *"Band 3 was selected over Band 1 with a priority advantage of +3.300. Age-of-Information urgency is higher on Band 3 (1.50 vs 0.20). Empirical reward history is stronger on Band 3 (3.20 vs 0.10)."*

---

## 3. Standalone `ScanScheduler` Service

Implemented in [`drishti/service.py`](file:///d:/Projects/Dristi_freq/drishti/service.py):
Provides a production-grade service class wrapping all DRISHTI schedulers:

```python
from drishti.service import ScanScheduler

# Instantiate service in intelligent auto-routing mode
scheduler = ScanScheduler(num_bands=16, mode="auto")

# Runtime cycle
action = scheduler.step(observation, info)
scheduler.update(observation, action, reward, step_info)

# Real-time explainability
explanation = scheduler.explain()
print(explanation["reason"])

# Contrastive operator query
print(scheduler.explain_contrastive(chosen_band=action, alternative_band=4))

# Export mission audit log
scheduler.export_logs("results/mission_audit.jsonl")
```

### Operating Modes:
- `"auto"`: Intelligently routes decisions to the best available policy (`PPO Augmented` $\rightarrow$ `PeriodicAware` $\rightarrow$ `SlidingWindowUCB`).
- `"periodic"`: Enforces lookahead pulse synchronization.
- `"ppo"`: Executes deep RL policy.
- `"bandit_ucb"` / `"bandit_ts"`: Executes non-stationary bandits.
- `"priority"` / `"sequential"` / `"random"`: Baseline benchmark modes.

---

## 4. Formal Model Card

The formal Model Card was compiled and published to [`docs/model_card.md`](file:///d:/Projects/Dristi_freq/docs/model_card.md), covering:
- Mathematical formulations & observation schemas
- Full 30-seed benchmark results across all 7 schedulers
- Real-time latency guarantees ($15.55\ \mu\text{s}$ CPU execution)
- Limitations, operating envelopes, and ethical considerations.

---

## 5. Verification & Test Suite

Tested in [`tests/test_phase6.py`](file:///d:/Projects/Dristi_freq/tests/test_phase6.py):
1. **Ring Buffer & JSONL Export**: Verified memory bounding, recent record retrieval, and JSONL disk serialization round-trip.
2. **Contrastive Reasoning**: Verified natural-language delta and component advantage reporting.
3. **ScanScheduler Production Service**: Verified mode switching, auto-routing, episode execution, and exception safety on invalid inputs.

**All 25 unit tests pass in 2.8s across Phases 1–6.**
