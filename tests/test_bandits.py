import numpy as np
import pytest
from schedulers.bandits import SlidingWindowUCB, DiscountedThompsonSampling


def test_sliding_window_ucb_unvisited_exploration():
    num_bands = 4
    scheduler = SlidingWindowUCB(num_bands=num_bands, window_size=20)
    scheduler.reset()

    obs = np.zeros(3 * num_bands, dtype=np.float32)

    # First num_bands steps should pull each arm at least once due to unvisited bonus
    pulled = set()
    for _ in range(num_bands):
        action = scheduler.act(obs)
        assert 0 <= action < num_bands
        pulled.add(action)
        scheduler.update(action, reward=1.0, obs=obs, info={"detected": True})

    assert len(pulled) == num_bands


def test_sliding_window_ucb_exploit_high_reward():
    num_bands = 4
    scheduler = SlidingWindowUCB(num_bands=num_bands, window_size=30, exploration_coef=0.2)
    scheduler.reset()

    obs = np.zeros(3 * num_bands, dtype=np.float32)

    # Simulate band 2 having consistently high reward, others 0
    for t in range(50):
        action = scheduler.act(obs)
        reward = 10.0 if action == 2 else 0.0
        scheduler.update(action, reward=reward, obs=obs, info={"detected": (action == 2)})

    # Band 2 should be the dominant choice
    recent_actions = [scheduler.act(obs) for _ in range(10)]
    assert recent_actions.count(2) >= 7


def test_discounted_thompson_sampling_discounting():
    num_bands = 3
    gamma = 0.90
    scheduler = DiscountedThompsonSampling(num_bands=num_bands, gamma=gamma, seed=42)
    scheduler.reset(seed=42)

    obs = np.zeros(3 * num_bands, dtype=np.float32)
    scheduler.update(0, reward=5.0, obs=obs, info={"detected": True, "threat_reward": 5.0})

    # Alpha for band 0 should have increased
    assert scheduler.alpha[0] > 1.0

    # Step without detections on band 0, alpha should discount towards 1
    prev_alpha = scheduler.alpha[0]
    scheduler.update(1, reward=0.0, obs=obs, info={"detected": False})
    assert scheduler.alpha[0] < prev_alpha


def test_bandit_explanations():
    num_bands = 4
    sw = SlidingWindowUCB(num_bands=num_bands)
    dts = DiscountedThompsonSampling(num_bands=num_bands)
    sw.reset()
    dts.reset()

    obs = np.zeros(3 * num_bands, dtype=np.float32)
    a_sw = sw.act(obs)
    a_dts = dts.act(obs)

    exp_sw = sw.explain(obs, a_sw)
    exp_dts = dts.explain(obs, a_dts)

    assert isinstance(exp_sw, str) and len(exp_sw) > 10
    assert isinstance(exp_dts, str) and len(exp_dts) > 10
