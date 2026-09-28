from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from rf_env.environment import EWScanEnv
from baselines.base import BaseScheduler
from metrics.evaluator import calculate_episode_metrics, EpisodeMetrics


@dataclass
class SchedulerSimulationResult:
    """Detailed step-by-step logs and episode metrics from running a scheduler."""
    scheduler_name: str
    actions: List[int]
    rewards: List[float]
    cumulative_rewards: List[float]
    detections: List[bool]
    true_detections: List[bool]
    false_alarms: List[bool]
    explanations: List[str]
    detected_emitters: List[List[str]]
    metrics: EpisodeMetrics
    ground_truth_matrix: np.ndarray


def run_single_episode_simulation(
    env: EWScanEnv,
    scheduler: BaseScheduler,
    seed: int,
) -> SchedulerSimulationResult:
    """
    Runs an episode with detailed per-step recording for live visualization.
    """
    obs, info = env.reset(seed=seed)
    scheduler.reset(seed=seed)

    actions: List[int] = []
    rewards: List[float] = []
    cumulative_rewards: List[float] = []
    detections: List[bool] = []
    true_detections: List[bool] = []
    false_alarms: List[bool] = []
    explanations: List[str] = []
    detected_emitters_per_step: List[List[str]] = []

    running_reward = 0.0
    done = False

    while not done:
        action = scheduler.act(obs, info)
        explanation = scheduler.explain(obs, action)

        obs, reward, terminated, truncated, step_info = env.step(action)
        scheduler.update(action, reward, obs, step_info)

        running_reward += reward
        actions.append(action)
        rewards.append(reward)
        cumulative_rewards.append(running_reward)
        detections.append(bool(step_info["detected"]))
        true_detections.append(step_info["true_detections"] > 0)
        false_alarms.append(step_info["false_alarms"] > 0)
        explanations.append(explanation)
        detected_emitters_per_step.append(step_info.get("detected_emitters", []))

        done = terminated or truncated

    ep_metrics = calculate_episode_metrics(env, running_reward)
    gt_matrix = (
        env.rf_world.ground_truth_matrix[: len(actions)].copy()
        if env.rf_world.ground_truth_matrix is not None
        else np.zeros((len(actions), env.num_bands), dtype=bool)
    )

    return SchedulerSimulationResult(
        scheduler_name=scheduler.name,
        actions=actions,
        rewards=rewards,
        cumulative_rewards=cumulative_rewards,
        detections=detections,
        true_detections=true_detections,
        false_alarms=false_alarms,
        explanations=explanations,
        detected_emitters=detected_emitters_per_step,
        metrics=ep_metrics,
        ground_truth_matrix=gt_matrix,
    )
