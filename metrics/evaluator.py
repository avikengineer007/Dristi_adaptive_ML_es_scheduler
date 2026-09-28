from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from scipy import stats

from rf_env.environment import EWScanEnv
from baselines.base import BaseScheduler


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single evaluation episode."""
    probability_of_detection: float
    false_alarm_rate: float
    interception_ratio: float
    average_intercept_time: float
    intercept_time_error: float
    total_reward: float
    total_dwells: int
    total_bursts: int
    intercepted_bursts: int


@dataclass
class AggregatedMetric:
    """Summary statistics (mean, std, 95% confidence interval) across multiple seeds."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float

    def __str__(self) -> str:
        half_ci = (self.ci_upper - self.ci_lower) / 2.0
        return f"{self.mean:.4f} +/- {half_ci:.4f}"


def calculate_episode_metrics(env: EWScanEnv, total_reward: float) -> EpisodeMetrics:
    """
    Computes rigorous radar/EW performance metrics from ground truth logs of an episode.
    """
    observations = env.episode_observations
    bursts = env.rf_world.burst_events

    # 1. Detection probability and False Alarm Rate
    opportunities_present = 0
    true_detections = 0
    opportunities_absent = 0
    false_alarms = 0

    for obs in observations:
        # Check ground truth for the visited band at time slot obs.time_slot
        t = obs.time_slot
        b = obs.band
        gt_active = False
        if env.rf_world.ground_truth_matrix is not None and t < env.rf_world.ground_truth_matrix.shape[0]:
            gt_active = bool(env.rf_world.ground_truth_matrix[t, b])

        if gt_active:
            opportunities_present += 1
            if obs.detected and obs.is_true_detection:
                true_detections += 1
        else:
            opportunities_absent += 1
            if obs.detected and obs.is_false_alarm:
                false_alarms += 1

    p_d = true_detections / opportunities_present if opportunities_present > 0 else 0.0
    far = false_alarms / opportunities_absent if opportunities_absent > 0 else 0.0

    # 2. Interception ratio and Intercept Time
    total_bursts = len(bursts)
    intercepted_bursts = [b for b in bursts if b.intercepted and b.first_intercept_time is not None]
    num_intercepted = len(intercepted_bursts)
    interception_ratio = num_intercepted / total_bursts if total_bursts > 0 else 0.0

    intercept_latencies: List[float] = [
        float(b.first_intercept_time - b.start_time)  # type: ignore
        for b in intercepted_bursts
    ]

    if intercept_latencies:
        avg_intercept_time = float(np.mean(intercept_latencies))
        intercept_time_error = float(np.std(intercept_latencies))
    else:
        # Penalty default if no bursts intercepted
        avg_intercept_time = float(env.max_steps)
        intercept_time_error = 0.0

    return EpisodeMetrics(
        probability_of_detection=p_d,
        false_alarm_rate=far,
        interception_ratio=interception_ratio,
        average_intercept_time=avg_intercept_time,
        intercept_time_error=intercept_time_error,
        total_reward=total_reward,
        total_dwells=len(observations),
        total_bursts=total_bursts,
        intercepted_bursts=num_intercepted,
    )


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> AggregatedMetric:
    """Computes sample mean, std, and two-sided Student's t confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return AggregatedMetric(mean=mean, std=0.0, ci_lower=mean, ci_upper=mean)

    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(n)
    h = sem * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return AggregatedMetric(mean=mean, std=std, ci_lower=mean - h, ci_upper=mean + h)


class MultiSeedEvaluator:
    """
    Evaluates schedulers across identical seeds and dwell parameters,
    producing publication-quality metric aggregates with confidence intervals.
    """

    def __init__(
        self,
        env_factory: Callable[[], EWScanEnv],
        seeds: List[int],
    ) -> None:
        self.env_factory = env_factory
        self.seeds = seeds

    def evaluate_scheduler(self, scheduler: BaseScheduler) -> Dict[str, AggregatedMetric]:
        """
        Runs the scheduler through all seeds and aggregates metrics.
        """
        p_d_list: List[float] = []
        far_list: List[float] = []
        ir_list: List[float] = []
        ait_list: List[float] = []
        ite_list: List[float] = []
        reward_list: List[float] = []

        env = self.env_factory()

        for seed in self.seeds:
            obs, info = env.reset(seed=seed)
            scheduler.reset(seed=seed)

            total_reward = 0.0
            done = False

            while not done:
                action = scheduler.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                scheduler.update(action, reward, obs, info)
                total_reward += reward
                done = terminated or truncated

            ep_metrics = calculate_episode_metrics(env, total_reward)

            p_d_list.append(ep_metrics.probability_of_detection)
            far_list.append(ep_metrics.false_alarm_rate)
            ir_list.append(ep_metrics.interception_ratio)
            ait_list.append(ep_metrics.average_intercept_time)
            ite_list.append(ep_metrics.intercept_time_error)
            reward_list.append(ep_metrics.total_reward)

        return {
            "Probability of Detection (Pd)": compute_confidence_interval(p_d_list),
            "False Alarm Rate (FAR)": compute_confidence_interval(far_list),
            "Interception Ratio (IR)": compute_confidence_interval(ir_list),
            "Avg Intercept Time (AIT)": compute_confidence_interval(ait_list),
            "Intercept Time Error (ITE)": compute_confidence_interval(ite_list),
            "Average Episode Reward": compute_confidence_interval(reward_list),
        }
