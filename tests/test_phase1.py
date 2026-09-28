import numpy as np
import pytest
from drishti.env.environment import SpectrumScanEnv
from drishti.env.emitters import PeriodicBurstEmitter, FixedEmitter
from drishti.data.adapter import SyntheticSource
from drishti.utils.config import load_scenario_config, create_env_from_config


def test_seed_determinism_identical_trajectories():
    """Verify that runs with identical seed and actions produce exact identical observations, rewards, and ground truth."""
    config = load_scenario_config("hard")
    env1 = create_env_from_config(config)
    env2 = create_env_from_config(config)

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_allclose(obs1, obs2)

    # Execute 50 deterministic actions
    for t in range(50):
        action = t % env1.num_bands
        o1, r1, term1, trunc1, info1 = env1.step(action)
        o2, r2, term2, trunc2, info2 = env2.step(action)

        np.testing.assert_allclose(o1, o2)
        assert r1 == pytest.approx(r2)
        assert term1 == term2
        assert trunc1 == trunc2
        assert info1["detected"] == info2["detected"]
        assert info1["threat_reward"] == pytest.approx(info2["threat_reward"])

    # Ground truth matrices must be identical
    np.testing.assert_array_equal(
        env1.ground_truth_matrix[:50],
        env2.ground_truth_matrix[:50],
    )


def test_ground_truth_matches_emitter_definitions():
    """Verify periodic burst emitter on-times match the exact periodic schedule when jitter is 0."""
    period = 10
    on_time = 3
    fixed_phase = 2
    band = 5

    emitter = PeriodicBurstEmitter(
        emitter_id="TEST_BURST",
        band=band,
        period=period,
        on_time=on_time,
        phase=fixed_phase,
        jitter=0.0,
    )
    rng = np.random.default_rng(999)
    emitter.reset(rng)

    for t in range(50):
        expected_active = ((t + fixed_phase) % period) < on_time
        emission = emitter.get_emission(t, rng)
        if expected_active:
            assert emission == band
        else:
            assert emission is None


def test_observations_never_leak_ground_truth():
    """Verify that observation vector contains only partial history and never the current ground truth."""
    env = SpectrumScanEnv(num_bands=8, max_steps=50)
    obs, info = env.reset(seed=123)

    # Observation shape is 4*B + 1
    assert obs.shape == (4 * 8 + 1,)
    assert obs.dtype == np.float32

    # Step on band 0
    obs, reward, _, _, _ = env.step(0)

    # Ground truth matrix is not part of observation
    # Even if band 3 was active at step 0, obs should show time_since_visit = 1 for band 3, hits = 0, last_seen = 0
    # tau is [0:8], hits is [8:16], misses is [16:24], last_seen is [24:32], tuned_band is [32]
    assert obs[24 + 3] == 0.0  # last_seen on unvisited band 3 must be 0.0
    assert obs[8 + 3] == 0.0   # running hit ratio on unvisited band 3 must be 0.0


def test_reward_components_sum_correctly_on_mini_scenario():
    """Verify reward components (threat, first intercept bonus, dwell cost, switch cost, false alarm) sum exactly."""
    # Hand-built scenario with 1 fixed emitter on band 2
    fixed_emitter = FixedEmitter("FIXED_CH2", band=2, threat_weight=10.0, active_ratio=1.0)
    source = SyntheticSource([])
    
    # Custom env with 0 false alarms and 0 noise for exact test
    from drishti.env.receiver import ReceiverModel
    ideal_receiver = ReceiverModel(p_fa=0.0, noise_std_db=0.0)

    env = SpectrumScanEnv(
        num_bands=4,
        max_steps=10,
        dwell_time=1,
        dwell_cost=0.5,
        switch_cost=0.2,
        false_alarm_penalty=1.0,
        first_intercept_bonus=5.0,
        receiver_model=ideal_receiver,
    )
    env.reset(seed=1)
    env.emitters = [fixed_emitter]

    # Action 1: Tune to band 2 (first intercept, no prior switch penalty because it's first step)
    obs, r1, _, _, info1 = env.step(2)
    # Expected: threat (10.0) + first_intercept (5.0) - dwell_cost (0.5) - switch (0.0) = 14.5
    assert r1 == pytest.approx(14.5)
    assert info1["threat_reward"] == 10.0
    assert info1["first_intercept_bonus"] == 5.0
    assert info1["switch_penalty"] == 0.0

    # Action 2: Stay on band 2 (no switch penalty, second visit so no first_intercept bonus)
    obs, r2, _, _, info2 = env.step(2)
    # Expected: threat (10.0) + 0 - dwell_cost (0.5) - switch (0.0) = 9.5
    assert r2 == pytest.approx(9.5)
    assert info2["first_intercept_bonus"] == 0.0

    # Action 3: Switch to silent band 0 (switch penalty 0.2, dwell cost 0.5, no detection)
    obs, r3, _, _, info3 = env.step(0)
    # Expected: 0 - dwell_cost (0.5) - switch (0.2) = -0.7
    assert r3 == pytest.approx(-0.7)
    assert info3["switch_penalty"] == 0.2
