import pytest
import numpy as np

from drishti.models.periodicity import (
    CircularPhaseCoherenceEstimator,
    PeriodicEmitterTracker,
)
from drishti.models.receiver_model import ReceiverPredictiveModel
from drishti.schedulers.periodic_aware import PeriodicAwareScheduler


def test_circular_phase_coherence_recovery():
    """Verify period recovery from sparse periodic timestamps with missing pulses."""
    true_period = 25
    true_phase = 7
    est = CircularPhaseCoherenceEstimator(min_period=10, max_period=60, coherence_threshold=0.70)
    est.reset()

    # Simulate sparse detection: only 40% of pulses are intercepted
    rng = np.random.default_rng(42)
    for k in range(30):
        if rng.random() < 0.40:
            pulse_slot = true_phase + k * true_period
            est.add_observation(pulse_slot)

    assert est.estimated_period == true_period
    assert est.coherence_score >= 0.70
    assert est.estimated_phase == (true_phase % true_period)


def test_periodic_emitter_tracker_multichannel():
    """Verify multichannel tracker tracks periods and forecasts arrivals."""
    tracker = PeriodicEmitterTracker(num_bands=8, min_period=10, max_period=50)
    tracker.reset()

    # Band 3 has periodic pulses every 20 slots starting at slot 5
    for k in range(10):
        slot = 5 + k * 20
        tracker.update(band=3, slot=slot, detected=True)

    est3 = tracker.estimators[3]
    assert est3.estimated_period == 20

    # At slot 44, upcoming pulse is expected at slot 45 (within horizon 2)
    upcoming = tracker.get_upcoming_bursts(current_slot=44, horizon=2)
    assert 3 in upcoming
    assert upcoming[3] >= 0.70


def test_target_scan_receiver_rendezvous():
    """Verify tracking and rendezvous forecasting for PeriodicScanReceiverTarget."""
    tracker = PeriodicEmitterTracker(num_bands=16)
    tracker.reset()

    # Simulate listening sequence [2, 5, 9] with dwell 4 (cycle period = 12)
    sequence = [2, 5, 9]
    dwell = 4
    for cycle in range(4):
        for idx, band in enumerate(sequence):
            for d in range(dwell):
                slot = cycle * 12 + idx * dwell + d
                # Intercept some slots
                if d in (0, 1):
                    tracker.update(band=band, slot=slot, detected=True)

    assert tracker.target_band_sequence == sequence
    assert tracker.target_cycle_period == 12


def test_periodic_aware_scheduler_explanation():
    """Verify PeriodicAwareScheduler produces structured explanations and sync flags."""
    sched = PeriodicAwareScheduler(num_bands=8)
    sched.reset(seed=123)

    obs = np.zeros(4 * 8 + 1, dtype=np.float32)
    action = sched.act(obs)
    exp = sched.explain()

    assert exp["scheduler"] == "Periodic-Aware Scheduler"
    assert exp["chosen_band"] == action
    assert "reason" in exp
    assert "is_predictive_sync" in exp
    assert isinstance(exp["tracked_periods"], dict)


def test_receiver_predictive_model_calibration():
    """Verify ReceiverPredictiveModel probability predictions and Brier score."""
    model = ReceiverPredictiveModel(num_bands=4)
    model.reset()

    prob = model.predict_hit_probability(
        band=1,
        hit_ratio=0.8,
        tau_norm=0.1,
        last_seen=1.0,
        periodic_confidence=0.9,
    )
    assert 0.0 <= prob <= 1.0
    assert prob > 0.6  # High occupancy + last seen + periodic should predict high prob

    # Test calibration curve with synthetic record
    for _ in range(50):
        p = float(np.random.uniform(0.1, 0.9))
        y = bool(np.random.random() < p)
        model.record_outcome(p, y)

    calib = model.compute_calibration_curve(num_bins=5)
    assert "brier_score" in calib
    assert 0.0 <= calib["brier_score"] <= 1.0
    assert len(calib["bin_centers"]) > 0
