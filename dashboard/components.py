from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from drishti.env.environment import SpectrumScanEnv
from drishti.baselines.base import Scheduler
from drishti.metrics.evaluator import compute_episode_metrics, MetricsResult
from drishti.env.emitters import FixedEmitter, PeriodicBurstEmitter, FrequencyAgileEmitter


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
    metrics: MetricsResult
    ground_truth_matrix: np.ndarray


def run_single_episode_simulation(
    env: SpectrumScanEnv,
    scheduler: Scheduler,
    seed: int,
    shock_step: Optional[int] = None,
    shock_band: Optional[int] = None,
    shock_type: str = "agile",
) -> SchedulerSimulationResult:
    """
    Runs an episode with detailed per-step recording for live dashboard visualization.
    Supports dynamic mid-mission 'Scenario Shock' emitter injection.
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
    shock_applied = False

    while not done:
        t = env.current_slot

        # Apply mid-mission scenario shock if configured
        if shock_step is not None and t >= shock_step and not shock_applied:
            shock_target_band = shock_band if shock_band is not None else (env.num_bands // 2)
            if shock_type == "agile":
                hop_set = [(shock_target_band + i) % env.num_bands for i in range(3)]
                shock_emitter = FrequencyAgileEmitter(
                    emitter_id="SHOCK_AGILE_THREAT",
                    hop_bands=hop_set,
                    threat_weight=10.0,
                    power_dbm=40.0,
                )
            elif shock_type == "periodic":
                shock_emitter = PeriodicBurstEmitter(
                    emitter_id="SHOCK_PULSE_RADAR",
                    band=shock_target_band,
                    period=15,
                    on_time=3,
                    threat_weight=10.0,
                    power_dbm=40.0,
                )
            else:
                shock_emitter = FixedEmitter(
                    emitter_id="SHOCK_JAMMER",
                    band=shock_target_band,
                    threat_weight=8.0,
                    power_dbm=35.0,
                )
            shock_emitter.reset(env.np_random)
            env.emitters.append(shock_emitter)
            shock_applied = True

        action = scheduler.act(obs, info)
        exp_dict = scheduler.explain()
        explanation = exp_dict.get("reason", f"Selected band {action}")

        obs, reward, terminated, truncated, step_info = env.step(action)
        scheduler.update(obs=obs, action=action, reward=reward, info=step_info)

        running_reward += reward
        actions.append(action)
        rewards.append(reward)
        cumulative_rewards.append(running_reward)
        detections.append(bool(step_info.get("detected", False)))
        true_detections.append(bool(step_info.get("true_detections", 0) > 0))
        false_alarms.append(bool(step_info.get("false_alarms", 0) > 0))
        explanations.append(explanation)
        detected_emitters_per_step.append(step_info.get("detected_emitters", []))

        done = terminated or truncated

    metrics = compute_episode_metrics(env, running_reward)

    gt_mat = env.ground_truth_matrix if env.ground_truth_matrix is not None else np.zeros((env.current_slot, env.num_bands), dtype=bool)

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
        metrics=metrics,
        ground_truth_matrix=gt_mat[:len(actions)],
    )
