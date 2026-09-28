import numpy as np
import pytest
from rf_env.environment import EWScanEnv


def test_seed_determinism_identical_runs():
    """Verify that runs with identical seed and actions produce exact identical outputs."""
    env1 = EWScanEnv(num_bands=8, max_steps=100)
    env2 = EWScanEnv(num_bands=8, max_steps=100)

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_allclose(obs1, obs2)

    # Step through 50 actions deterministically
    actions = [i % 8 for i in range(50)]
    for a in actions:
        o1, r1, term1, trunc1, info1 = env1.step(a)
        o2, r2, term2, trunc2, info2 = env2.step(a)

        np.testing.assert_allclose(o1, o2)
        assert r1 == pytest.approx(r2)
        assert term1 == term2
        assert trunc1 == trunc2
        assert info1 == info2

    # Check ground truth matrix identity
    assert np.array_equal(
        env1.rf_world.ground_truth_matrix[:50],
        env2.rf_world.ground_truth_matrix[:50],
    )


def test_seed_divergence_different_seeds():
    """Verify that different seeds produce divergent ground truth histories."""
    env1 = EWScanEnv(num_bands=8, max_steps=100)
    env2 = EWScanEnv(num_bands=8, max_steps=100)

    env1.reset(seed=101)
    env2.reset(seed=202)

    for i in range(30):
        env1.step(i % 8)
        env2.step(i % 8)

    # Ground truth matrices should not be identical for different seeds
    gt1 = env1.rf_world.ground_truth_matrix[:30]
    gt2 = env2.rf_world.ground_truth_matrix[:30]
    assert not np.array_equal(gt1, gt2)
