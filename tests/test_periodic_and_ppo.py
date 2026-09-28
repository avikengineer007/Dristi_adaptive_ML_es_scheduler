import numpy as np
import pytest
from schedulers.periodic_tracker import PeriodicEmitterTracker, PeriodicPredictiveScheduler
from schedulers.ppo_agent import PPOScheduler


def test_periodic_emitter_tracker_estimation():
    tracker = PeriodicEmitterTracker(min_period=10, max_period=30, coherence_threshold=0.8)
    tracker.reset(num_bands=4)

    # Simulate periodic pulse arrivals on band 2 with true period = 18, phase = 4
    true_period = 18
    phase = 4
    for k in range(8):
        t = k * true_period + phase
        tracker.record_observation(band=2, time_slot=t, detected=True)

    assert 2 in tracker.estimated_periods
    estimated_t = tracker.estimated_periods[2]
    # Should accurately recover true period
    assert abs(estimated_t - true_period) <= 1.0
    assert tracker.coherences[2] >= 0.8


def test_periodic_predictive_scheduler():
    num_bands = 8
    sched = PeriodicPredictiveScheduler(num_bands=num_bands)
    sched.reset(seed=42)

    obs = np.zeros(3 * num_bands, dtype=np.float32)

    # Act and update
    a = sched.act(obs)
    assert 0 <= a < num_bands

    sched.update(a, reward=4.0, obs=obs, info={"detected": True})
    explanation = sched.explain(obs, a)
    assert isinstance(explanation, str) and len(explanation) > 10


def test_ppo_scheduler_interface():
    num_bands = 8
    ppo = PPOScheduler(num_bands=num_bands)
    obs = np.zeros(3 * num_bands, dtype=np.float32)

    # Without loaded model, fallback action
    action = ppo.act(obs)
    assert 0 <= action < num_bands
