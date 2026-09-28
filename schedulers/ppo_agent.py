import os
from typing import Optional, Dict, Any
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from baselines.base import BaseScheduler
from rf_env.environment import EWScanEnv


class PPOScheduler(BaseScheduler):
    """
    Deep Reinforcement Learning ES Frequency Scheduler using Proximal Policy Optimization (PPO).
    Learns state-dependent dynamic tuning policies balancing exploration (time-since-visit)
    with exploitation of active and high-threat radar channels.
    """

    def __init__(
        self,
        num_bands: int,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_bands=num_bands, name="PPO Deep RL")
        self.device = device
        self.model_path = model_path
        self.model: Optional[PPO] = None

        if model_path is not None and os.path.exists(model_path):
            self.load(model_path)

    def train_agent(
        self,
        total_timesteps: int = 40000,
        save_path: str = "artifacts/ppo_ew_model.zip",
        seed: int = 42,
    ) -> None:
        """Trains a PPO policy on EWScanEnv and saves the model artifact."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def make_training_env():
            env = EWScanEnv(num_bands=self.num_bands, max_steps=500)
            env.reset(seed=seed)
            return env

        vec_env = DummyVecEnv([make_training_env])

        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.Tanh,
        )

        self.model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Maintain exploratory entropy
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=self.device,
        )

        print(f"Training PPO Agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(save_path)
        self.model_path = save_path
        print(f"PPO Agent trained and saved to: {save_path}")

    def load(self, model_path: str) -> None:
        """Loads pre-trained PPO model weights, verifying space dimensions match current num_bands."""
        loaded_model = PPO.load(model_path, device=self.device)
        expected_obs_dim = 3 * self.num_bands
        expected_action_dim = self.num_bands

        obs_match = (
            hasattr(loaded_model.observation_space, "shape")
            and loaded_model.observation_space.shape[0] == expected_obs_dim
        )
        action_match = (
            hasattr(loaded_model.action_space, "n")
            and loaded_model.action_space.n == expected_action_dim
        )

        if not (obs_match and action_match):
            print(
                f"[PPO] Space mismatch in '{model_path}': "
                f"Obs={getattr(loaded_model.observation_space, 'shape', None)}, "
                f"Actions={getattr(loaded_model.action_space, 'n', None)} "
                f"vs required Obs=({expected_obs_dim},), Actions={expected_action_dim}. "
                f"Automatically training compatible PPO agent for {self.num_bands} bands..."
            )
            self.train_agent(total_timesteps=20000, save_path=model_path)
        else:
            self.model = loaded_model
            self.model_path = model_path

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        if self.model is None:
            # Fallback to center band if not yet loaded
            return self.num_bands // 2

        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def explain(self, obs: np.ndarray, action: int) -> str:
        if self.model is None:
            return "PPO policy not initialized."

        # Compute action probabilities from the policy network
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]

        chosen_prob = probs[action]
        return (
            f"PPO Deep RL selected band {action}: policy probability={chosen_prob:.2%}, "
            f"derived from {len(obs)}-dim state vector (norm tau, hit history, last seen)."
        )
