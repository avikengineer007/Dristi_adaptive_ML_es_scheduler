import numpy as np
import pytest
from baselines.sequential import SequentialSweep
from baselines.random_scan import RandomScan
from baselines.priority_sweep import PrioritySweep


def test_sequential_sweep():
    num_bands = 8
    scheduler = SequentialSweep(num_bands=num_bands, start_band=0)
    scheduler.reset()

    dummy_obs = np.zeros(3 * num_bands, dtype=np.float32)

    # First cycle
    actions = [scheduler.act(dummy_obs) for _ in range(num_bands)]
    assert actions == list(range(num_bands))

    # Second cycle
    actions_2 = [scheduler.act(dummy_obs) for _ in range(num_bands)]
    assert actions_2 == list(range(num_bands))


def test_random_scan():
    num_bands = 10
    scheduler = RandomScan(num_bands=num_bands, seed=42)
    scheduler.reset(seed=42)

    dummy_obs = np.zeros(3 * num_bands, dtype=np.float32)
    actions = [scheduler.act(dummy_obs) for _ in range(100)]

    for a in actions:
        assert 0 <= a < num_bands

    # Should visit most bands over 100 draws
    unique_actions = set(actions)
    assert len(unique_actions) >= 8


def test_priority_sweep():
    num_bands = 4
    # Band 3 has 5x weight of others
    priorities = {3: 5.0, 0: 1.0, 1: 1.0, 2: 1.0}
    scheduler = PrioritySweep(num_bands=num_bands, band_priorities=priorities)
    scheduler.reset()

    dummy_obs = np.zeros(3 * num_bands, dtype=np.float32)
    actions = [scheduler.act(dummy_obs) for _ in range(40)]

    counts = {b: actions.count(b) for b in range(num_bands)}
    # Band 3 should be sampled significantly more often than band 0
    assert counts[3] > counts[0]
    assert counts[3] > counts[1]
    assert counts[3] > counts[2]
