import numpy as np
import pytest
from drishti.baselines.sequential import SequentialSweep
from drishti.baselines.random_scan import RandomScan
from drishti.baselines.priority_sweep import PriorityPreMissionSweep
from drishti.env.environment import SpectrumScanEnv, BurstRecord
from drishti.metrics.evaluator import (
    compute_episode_metrics,
    calculate_statistical_summary,
    MultiSeedEvaluator,
)
from drishti.utils.config import load_scenario_config, create_env_from_config
from experiments.benchmark import build_baseline_schedulers


def test_sequential_sweep_periodicity():
    """Verify sequential sweep visits each band exactly once every B steps."""
    num_bands = 8
    scheduler = SequentialSweep(num_bands=num_bands)
    scheduler.reset()

    dummy_obs = np.zeros(4 * num_bands + 1, dtype=np.float32)

    # First cycle
    cycle_1 = [scheduler.choose_action(dummy_obs) for _ in range(num_bands)]
    assert cycle_1 == list(range(num_bands))

    # Second cycle
    cycle_2 = [scheduler.choose_action(dummy_obs) for _ in range(num_bands)]
    assert cycle_2 == list(range(num_bands))


def test_random_scan_bounds_and_entropy():
    """Verify random scan stays in bounds and samples across all channels."""
    num_bands = 10
    scheduler = RandomScan(num_bands=num_bands, seed=42)
    scheduler.reset(seed=42)

    dummy_obs = np.zeros(4 * num_bands + 1, dtype=np.float32)
    actions = [scheduler.choose_action(dummy_obs) for _ in range(100)]

    for a in actions:
        assert 0 <= a < num_bands
    assert len(set(actions)) >= 8


def test_priority_sweep_allocates_proportional_dwells():
    """Verify priority sweep allocates substantially more dwells to high-priority channels."""
    num_bands = 4
    # Band 3 has 5x weight
    priorities = {3: 5.0, 0: 1.0, 1: 1.0, 2: 1.0}
    scheduler = PriorityPreMissionSweep(num_bands=num_bands, band_priorities=priorities)
    scheduler.reset()

    dummy_obs = np.zeros(4 * num_bands + 1, dtype=np.float32)
    actions = [scheduler.choose_action(dummy_obs) for _ in range(40)]

    counts = {b: actions.count(b) for b in range(num_bands)}
    assert counts[3] > counts[0]
    assert counts[3] > counts[1]
    assert counts[3] > counts[2]


def test_metrics_on_hand_built_case_with_censoring():
    """Verify metric calculations with known ground truth and censored bursts."""
    env = SpectrumScanEnv(num_bands=4, max_steps=20)
    env.reset(seed=42)

    # Create 3 synthetic burst records:
    # Burst 1: on band 1, slots 0-5, intercepted at slot 2 -> latency = 2 - 0 = 2
    b1 = BurstRecord("RADAR_1", band=1, start_slot=0, end_slot=5, threat_weight=5.0, intercepted=True, first_intercept_slot=2)
    # Burst 2: on band 2, slots 6-10, intercepted at slot 7 -> latency = 7 - 6 = 1
    b2 = BurstRecord("RADAR_2", band=2, start_slot=6, end_slot=10, threat_weight=4.0, intercepted=True, first_intercept_slot=7)
    # Burst 3: on band 3, slots 12-18, NEVER intercepted (censored)
    b3 = BurstRecord("RADAR_3", band=3, start_slot=12, end_slot=18, threat_weight=8.0, intercepted=False, first_intercept_slot=None)

    env.burst_records = [b1, b2, b3]

    metrics = compute_episode_metrics(env, cumulative_reward=100.0)

    # 2 out of 3 bursts intercepted
    assert metrics.total_bursts == 3
    assert metrics.intercepted_bursts == 2
    assert metrics.censored_burst_count == 1
    assert metrics.interception_ratio_count == pytest.approx(2.0 / 3.0)

    # Average intercept time across intercepted bursts: mean(2.0, 1.0) = 1.5
    assert metrics.average_intercept_time == pytest.approx(1.5)


def test_benchmark_reproducibility():
    """Verify that benchmark evaluations across identical seeds produce exact identical metric summaries."""
    config_data = load_scenario_config("easy")
    evaluator1 = MultiSeedEvaluator(lambda: create_env_from_config(config_data), seeds=[10, 20])
    evaluator2 = MultiSeedEvaluator(lambda: create_env_from_config(config_data), seeds=[10, 20])

    s1 = SequentialSweep(num_bands=16)
    s2 = SequentialSweep(num_bands=16)

    res1 = evaluator1.evaluate_scheduler(s1)
    res2 = evaluator2.evaluate_scheduler(s2)

    for k in res1:
        assert res1[k].mean == pytest.approx(res2[k].mean)
        assert res1[k].std == pytest.approx(res2[k].std)
        assert res1[k].ci_half == pytest.approx(res2[k].ci_half)
