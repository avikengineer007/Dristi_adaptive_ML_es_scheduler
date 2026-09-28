import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from rf_env.environment import EWScanEnv


import gymnasium as gym
import rf_env  # noqa: F401


def test_gym_compliance():
    """Verify Gymnasium standard compliance via check_env using gym.make."""
    env = gym.make("EWScan-v0", num_bands=16, max_steps=100)
    check_env(env.unwrapped)
    env.close()


def test_observation_and_action_spaces():
    num_bands = 8
    env = EWScanEnv(num_bands=num_bands, max_steps=50)
    obs, info = env.reset(seed=123)

    assert obs.shape == (3 * num_bands,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert env.action_space.n == num_bands
    assert info["num_bands"] == num_bands


def test_ground_truth_logging():
    env = EWScanEnv(num_bands=16, max_steps=40)
    env.reset(seed=777)

    for i in range(40):
        env.step(i % 16)

    gt = env.rf_world.ground_truth_matrix
    assert gt is not None
    assert gt.shape[0] >= 40
    assert gt.shape[1] == 16
    # Check that there were some ground-truth transmissions recorded
    assert np.sum(gt[:40]) > 0


def test_dwell_time_configuration():
    dwell = 3
    env = EWScanEnv(num_bands=8, max_steps=30, dwell_time=dwell)
    env.reset(seed=42)

    _, _, _, _, info = env.step(0)
    assert info["time_step"] == dwell
    assert env.time_step == dwell
