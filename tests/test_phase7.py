import pytest
import numpy as np

from drishti.env.environment import SpectrumScanEnv
from drishti.baselines.priority_sweep import PriorityPreMissionSweep
from drishti.schedulers.bandit import SlidingWindowUCB
from drishti.schedulers.periodic_aware import PeriodicAwareScheduler
from dashboard.components import run_single_episode_simulation, SchedulerSimulationResult


def test_dashboard_simulation_runner():
    """Verify single episode simulation generates complete trajectory records."""
    B = 8
    env = SpectrumScanEnv(num_bands=B, max_steps=50)
    sched = SlidingWindowUCB(num_bands=B, window_size=20)

    res = run_single_episode_simulation(env=env, scheduler=sched, seed=123)

    assert isinstance(res, SchedulerSimulationResult)
    assert len(res.actions) == 50
    assert len(res.rewards) == 50
    assert len(res.cumulative_rewards) == 50
    assert len(res.explanations) == 50
    assert res.ground_truth_matrix.shape == (50, B)
    assert res.metrics.cumulative_reward == res.cumulative_rewards[-1]


def test_dashboard_scenario_shock_injection():
    """Verify mid-mission shock emitter injection alters emitter dynamics."""
    B = 8
    env = SpectrumScanEnv(num_bands=B, max_steps=60)
    sched = PriorityPreMissionSweep(num_bands=B)

    res = run_single_episode_simulation(
        env=env,
        scheduler=sched,
        seed=42,
        shock_step=25,
        shock_band=4,
        shock_type="agile",
    )

    # After slot 25, the new shock emitter should be in the environment
    emitter_ids = [e.emitter_id for e in env.emitters]
    assert "SHOCK_AGILE_THREAT" in emitter_ids
    assert len(res.actions) == 60


def test_dashboard_periodic_aware_runner():
    """Verify PeriodicAwareScheduler runs cleanly in dashboard runner."""
    B = 8
    env = SpectrumScanEnv(num_bands=B, max_steps=40)
    sched = PeriodicAwareScheduler(num_bands=B)

    res = run_single_episode_simulation(env=env, scheduler=sched, seed=777)
    assert len(res.actions) == 40
    assert any("Periodic-Aware" in exp or "SW-UCB" in exp or "Selected" in exp for exp in res.explanations)
