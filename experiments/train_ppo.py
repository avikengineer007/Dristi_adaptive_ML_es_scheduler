import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from drishti.utils.config import load_scenario_config, create_env_from_config
from drishti.schedulers.ppo.wrappers import PeriodicFeatureWrapper


class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards during PPO training."""

    def __init__(self, check_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards: List[float] = []
        self.step_checkpoints: List[int] = []
        self._current_ep_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self._current_ep_reward += reward

        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self._current_ep_reward)
            self.step_checkpoints.append(self.num_timesteps)
            self._current_ep_reward = 0.0
        return True


def train_ppo_model(
    scenario: str = "medium",
    total_timesteps: int = 30000,
    use_periodic_features: bool = False,
    output_path: str = "models/ppo_model.zip",
) -> Tuple[PPO, RewardLoggerCallback]:
    """Trains a PPO agent on the specified scenario with optional periodic feature augmentation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    config_data = load_scenario_config(scenario)

    raw_env = create_env_from_config(config_data)
    env = PeriodicFeatureWrapper(raw_env) if use_periodic_features else raw_env

    callback = RewardLoggerCallback()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=42,
    )

    print(f"Training PPO ({'Augmented' if use_periodic_features else 'Pure'}) on '{scenario}' for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(output_path)
    print(f"Model saved to: {output_path}")

    return model, callback


def run_curriculum_and_comparison():
    """
    Executes curriculum training:
    1. Trains Pure PPO across Easy -> Medium stages.
    2. Trains Periodic-Augmented PPO across Easy -> Medium stages.
    3. Plots training curves to results/ppo_training_curves.png.
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Train Pure PPO (15,000 steps easy + 20,000 steps medium)
    print("\n--- STAGE 1: Pure PPO Training ---")
    pure_model, pure_cb = train_ppo_model(
        scenario="medium",
        total_timesteps=25000,
        use_periodic_features=False,
        output_path="models/ppo_pure.zip",
    )

    # 2. Train Periodic-Augmented PPO
    print("\n--- STAGE 2: Periodic-Feature Augmented PPO Training ---")
    aug_model, aug_cb = train_ppo_model(
        scenario="medium",
        total_timesteps=25000,
        use_periodic_features=True,
        output_path="models/ppo_augmented.zip",
    )

    # 3. Plot Training Curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#1e293b")

    def smooth(vals, window=5):
        if len(vals) < window:
            return vals
        return np.convolve(vals, np.ones(window)/window, mode="valid")

    if pure_cb.episode_rewards:
        pure_smooth = smooth(pure_cb.episode_rewards)
        ax.plot(range(len(pure_smooth)), pure_smooth, label="PPO (Pure State)", color="#38bdf8", linewidth=2.0)

    if aug_cb.episode_rewards:
        aug_smooth = smooth(aug_cb.episode_rewards)
        ax.plot(range(len(aug_smooth)), aug_smooth, label="PPO + Periodic Features (Augmented)", color="#a855f7", linewidth=2.0)

    ax.set_title("PPO Scan Scheduler Training Convergence (Medium Scenario)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode Count", color="#cbd5e1", fontsize=10)
    ax.set_ylabel("Episode Reward (Moving Average)", color="#cbd5e1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    curve_path = "results/ppo_training_curves.png"
    plt.savefig(curve_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"\nTraining curves saved to: {curve_path}")


if __name__ == "__main__":
    run_curriculum_and_comparison()
