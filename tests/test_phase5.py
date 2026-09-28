import pytest
import numpy as np
import torch

from drishti.env.environment import SpectrumScanEnv
from drishti.data.adapter import SyntheticSource
from drishti.schedulers.ppo.wrappers import PeriodicFeatureWrapper
from drishti.schedulers.ppo.scheduler import PPOScheduler
from experiments.benchmark_onnx import ActorPolicyModule


def test_periodic_feature_wrapper_dimension():
    """Verify observation space expansion from 4B+1 to 7B+1."""
    B = 8
    raw_env = SpectrumScanEnv(num_bands=B, max_steps=100)
    assert raw_env.observation_space.shape[0] == 4 * B + 1

    wrapped_env = PeriodicFeatureWrapper(raw_env)
    expected_dim = (4 * B + 1) + (3 * B)
    assert wrapped_env.observation_space.shape[0] == expected_dim

    obs, info = wrapped_env.reset(seed=42)
    assert obs.shape[0] == expected_dim
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    next_obs, reward, term, trunc, info = wrapped_env.step(action=2)
    assert next_obs.shape[0] == expected_dim


def test_ppo_scheduler_fallback_and_explanation():
    """Verify PPOScheduler behaves predictably and outputs structured explanations."""
    B = 8
    sched = PPOScheduler(num_bands=B, model=None, use_periodic_features=False)
    sched.reset(seed=42)

    obs = np.zeros(4 * B + 1, dtype=np.float32)
    # Band 5 has highest age of information
    obs[5] = 0.95
    action = sched.act(obs)
    assert action == 5

    exp = sched.explain()
    assert exp["scheduler"] == "PPO Scheduler"
    assert exp["chosen_band"] == 5
    assert "reason" in exp


def test_actor_policy_module_forward():
    """Verify ActorPolicyModule extracts correct logits and probability distribution."""
    B = 16
    obs_dim = 4 * B + 1

    # Create mock policy matching SB3 MLP actor architecture
    class MockPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp_extractor = torch.nn.Module()
            self.mlp_extractor.policy_net = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
            )
            self.action_net = torch.nn.Linear(64, B)

    mock = MockPolicy()
    actor_mod = ActorPolicyModule(mock)
    actor_mod.eval()

    dummy_input = torch.randn(2, obs_dim)
    logits = actor_mod(dummy_input)
    assert logits.shape == (2, B)

    probs = torch.softmax(logits, dim=-1)
    sums = torch.sum(probs, dim=-1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)
