import numpy as np
import pytest
from metrics.evaluator import (
    compute_confidence_interval,
    calculate_episode_metrics,
    MultiSeedEvaluator,
)
from rf_env.environment import EWScanEnv
from baselines.sequential import SequentialSweep


def test_confidence_interval_math():
    # Samples with known mean 10.0
    data = [8.0, 9.0, 10.0, 11.0, 12.0]
    agg = compute_confidence_interval(data, confidence=0.95)

    assert agg.mean == pytest.approx(10.0)
    assert agg.ci_lower < 10.0
    assert agg.ci_upper > 10.0
    # Symmetric CI
    assert (agg.ci_upper - 10.0) == pytest.approx(10.0 - agg.ci_lower)


def test_calculate_episode_metrics():
    env = EWScanEnv(num_bands=8, max_steps=50)
    env.reset(seed=123)

    total_r = 0.0
    for t in range(50):
        _, r, _, _, _ = env.step(t % 8)
        total_r += r

    metrics = calculate_episode_metrics(env, total_r)

    assert 0.0 <= metrics.probability_of_detection <= 1.0
    assert 0.0 <= metrics.false_alarm_rate <= 1.0
    assert 0.0 <= metrics.interception_ratio <= 1.0
    assert metrics.average_intercept_time >= 0.0
    assert metrics.total_dwells == 50
    assert metrics.total_bursts > 0


def test_multi_seed_evaluator():
    seeds = [1, 2, 3]
    evaluator = MultiSeedEvaluator(
        env_factory=lambda: EWScanEnv(num_bands=8, max_steps=40),
        seeds=seeds,
    )
    scheduler = SequentialSweep(num_bands=8)
    results = evaluator.evaluate_scheduler(scheduler)

    assert "Probability of Detection (Pd)" in results
    assert "False Alarm Rate (FAR)" in results
    assert "Interception Ratio (IR)" in results
    assert "Avg Intercept Time (AIT)" in results
    assert "Average Episode Reward" in results

    pd_agg = results["Probability of Detection (Pd)"]
    assert 0.0 <= pd_agg.mean <= 1.0
