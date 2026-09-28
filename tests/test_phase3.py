import pytest
import numpy as np

from drishti.schedulers.bandit import SlidingWindowUCB, DiscountedThompson
from drishti.env.environment import SpectrumScanEnv


def test_sliding_window_history_limit():
    """Verify SW-UCB window caps memory and purges old observations."""
    B = 8
    W = 10
    ucb = SlidingWindowUCB(num_bands=B, window_size=W)
    ucb.reset(seed=42)

    obs = np.zeros(4 * B + 1, dtype=np.float32)

    # Perform 25 updates on band 0
    for t in range(25):
        ucb.update(obs, action=0, reward=1.0, info={"detected": True})

    assert len(ucb.history) == W
    # Total pulls tracked separately
    assert ucb.pull_counts[0] == 25


def test_sliding_window_forced_exploration():
    """Verify SW-UCB forces visits to arms unpulled in the active window."""
    B = 4
    W = 5
    ucb = SlidingWindowUCB(num_bands=B, window_size=W)
    ucb.reset(seed=42)

    obs = np.zeros(4 * B + 1, dtype=np.float32)

    # First B steps should explore each unpulled arm
    chosen_bands = []
    for _ in range(B):
        action = ucb.act(obs)
        chosen_bands.append(action)
        ucb.update(obs, action=action, reward=0.0, info={"detected": False})

    assert len(set(chosen_bands)) == B


def test_discounted_thompson_discount_decay():
    """Verify D-TS decays older positive detections towards the prior."""
    B = 4
    gamma = 0.90
    dts = DiscountedThompson(num_bands=B, gamma=gamma, alpha_0=1.0, beta_0=1.0)
    dts.reset(seed=42)

    obs = np.zeros(4 * B + 1, dtype=np.float32)

    # Give band 0 a big hit
    dts.update(obs, action=0, reward=10.0, info={"detected": True, "threat_reward": 10.0})
    alpha_after_hit = dts.alphas[0]
    assert alpha_after_hit > 5.0

    # Dwell on other bands for 50 steps
    for _ in range(50):
        dts.update(obs, action=1, reward=0.0, info={"detected": False})

    # Band 0 alpha should have decayed close to alpha_0 = 1.0
    assert dts.alphas[0] < 2.0
    assert dts.alphas[0] >= 1.0


def test_bandit_reproducibility():
    """Verify bandits are strictly deterministic with identical seeds."""
    B = 8
    obs = np.zeros(4 * B + 1, dtype=np.float32)

    # Test SW-UCB reproducibility
    ucb1 = SlidingWindowUCB(num_bands=B, window_size=20)
    ucb2 = SlidingWindowUCB(num_bands=B, window_size=20)
    ucb1.reset(seed=999)
    ucb2.reset(seed=999)

    actions1 = [ucb1.act(obs) for _ in range(30)]
    actions2 = [ucb2.act(obs) for _ in range(30)]
    assert actions1 == actions2

    # Test D-TS reproducibility
    dts1 = DiscountedThompson(num_bands=B, gamma=0.92)
    dts2 = DiscountedThompson(num_bands=B, gamma=0.92)
    dts1.reset(seed=777)
    dts2.reset(seed=777)

    dts_actions1 = [dts1.act(obs) for _ in range(30)]
    dts_actions2 = [dts2.act(obs) for _ in range(30)]
    assert dts_actions1 == dts_actions2


def test_bandit_explanation_structure():
    """Verify structured explain() dictionary fields."""
    B = 4
    ucb = SlidingWindowUCB(num_bands=B, window_size=15)
    ucb.reset(seed=123)
    obs = np.zeros(4 * B + 1, dtype=np.float32)
    action = ucb.act(obs)

    exp = ucb.explain()
    assert exp["scheduler"] == "Sliding-Window UCB"
    assert exp["chosen_band"] == action
    assert "reason" in exp
    assert len(exp["scores"]) == B
    assert "window_pulls" in exp["components"]

    dts = DiscountedThompson(num_bands=B, gamma=0.95)
    dts.reset(seed=123)
    dts_action = dts.act(obs)
    dts_exp = dts.explain()
    assert dts_exp["scheduler"] == "Discounted Thompson"
    assert dts_exp["chosen_band"] == dts_action
    assert "sampled_thetas" in dts_exp["components"]
