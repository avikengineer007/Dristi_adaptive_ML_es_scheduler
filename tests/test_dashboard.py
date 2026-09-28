import numpy as np
import pytest
from rf_env.environment import EWScanEnv
from baselines.sequential import SequentialSweep
from schedulers.periodic_tracker import PeriodicPredictiveScheduler
from dashboard.components import run_single_episode_simulation


def test_dashboard_simulation_runner():
    num_bands = 8
    env = EWScanEnv(num_bands=num_bands, max_steps=40)
    scheduler = PeriodicPredictiveScheduler(num_bands=num_bands)

    result = run_single_episode_simulation(env=env, scheduler=scheduler, seed=42)

    assert result.scheduler_name == "Periodic-Predictive ML"
    assert len(result.actions) == 40
    assert len(result.cumulative_rewards) == 40
    assert len(result.explanations) == 40
    assert result.ground_truth_matrix.shape == (40, num_bands)
    assert 0.0 <= result.metrics.probability_of_detection <= 1.0
