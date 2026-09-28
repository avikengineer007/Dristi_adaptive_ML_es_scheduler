from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from scipy import stats

from drishti.env.environment import SpectrumScanEnv
from drishti.baselines.base import Scheduler


@dataclass
class MetricsResult:
    """Rigorous Electronic Warfare metrics evaluated against ground truth."""
    probability_of_detection: float
    false_alarm_rate: float
    interception_ratio_count: float
    interception_ratio_time: float
    average_intercept_time: float
    censored_burst_count: int
    average_intercept_rate: float
    intercept_time_error: float
    cumulative_reward: float
    average_reward: float
    total_bursts: int
    intercepted_bursts: int


@dataclass
class StatisticalSummary:
    """Mean, standard deviation, and Student-t 95% confidence intervals."""
    mean: float
    std: float
    ci_half: float
    ci_lower: float
    ci_upper: float

    def __str__(self) -> str:
        return f"{self.mean:.4f} +/- {self.ci_half:.4f}"


def calculate_statistical_summary(values: List[float], confidence: float = 0.95) -> StatisticalSummary:
    """Computes sample mean, standard deviation, and two-sided Student's t CI."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return StatisticalSummary(mean=mean, std=0.0, ci_half=0.0, ci_lower=mean, ci_upper=mean)

    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(n)
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    ci_half = t_crit * sem
    return StatisticalSummary(
        mean=mean,
        std=std,
        ci_half=ci_half,
        ci_lower=mean - ci_half,
        ci_upper=mean + ci_half,
    )


def compute_episode_metrics(
    env: SpectrumScanEnv,
    cumulative_reward: float,
    predicted_intercept_times: Optional[Dict[str, float]] = None,
) -> MetricsResult:
    """
    Computes all standard DRISHTI metrics against the episode's physical ground truth.

    Formulas:
    ---------
    1. Probability of Detection (Pd):
       Pd = (True Detections) / (Total Dwell Opportunities where Emitter was Present)

    2. False Alarm Rate (FAR):
       FAR = (False Alarms) / (Total Dwell Opportunities where Emitter was Absent)

    3. Interception Ratio - Count (IR_count):
       IR_count = (Intercepted Distinct Bursts) / (Total Distinct Bursts)

    4. Interception Ratio - Time (IR_time):
       IR_time = (Total Intercepted Slots) / (Total Active Emitter Slots across all bands)

    5. Average Intercept Time (AIT):
       For intercepted bursts: delta_t = first_intercept_slot - start_slot.
       For unintercepted bursts (censored): penalty assigned as (max_steps - start_slot).
       AIT = mean(delta_t across intercepted bursts).
    """
    observations = env.episode_observations
    bursts = env.burst_records
    gt_matrix = env.ground_truth_matrix

    # 1. Detection Probability and False Alarm Rate
    opp_present = 0
    true_detections = 0
    opp_absent = 0
    false_alarms = 0

    for obs in observations:
        t = obs.time_slot
        b = obs.band
        gt_active = False
        if gt_matrix is not None and t < gt_matrix.shape[0]:
            gt_active = bool(gt_matrix[t, b])

        if gt_active:
            opp_present += 1
            if obs.detected and obs.is_true_detection:
                true_detections += 1
        else:
            opp_absent += 1
            if obs.detected and obs.is_false_alarm:
                false_alarms += 1

    pd = true_detections / opp_present if opp_present > 0 else 0.0
    far = false_alarms / opp_absent if opp_absent > 0 else 0.0

    # 2. Interception Ratios
    total_bursts = len(bursts)
    intercepted_bursts = [b for b in bursts if b.intercepted and b.first_intercept_slot is not None]
    num_intercepted = len(intercepted_bursts)
    num_censored = total_bursts - num_intercepted

    ir_count = num_intercepted / total_bursts if total_bursts > 0 else 0.0

    # Time-based IR
    total_active_slots = int(np.sum(gt_matrix[: env.current_slot])) if gt_matrix is not None else 1
    ir_time = true_detections / max(1, total_active_slots)

    # 3. Average Intercept Time (AIT)
    latencies = [float(b.first_intercept_slot - b.start_slot) for b in intercepted_bursts]  # type: ignore
    if latencies:
        avg_intercept_time = float(np.mean(latencies))
    else:
        avg_intercept_time = float(env.max_steps)

    # 4. Average Intercept Rate (bursts intercepted per 100 slots)
    avg_intercept_rate = (num_intercepted / max(1, env.current_slot)) * 100.0

    # 5. Intercept Time Error (ITE) for predicted vs actual
    ite_errors = []
    if predicted_intercept_times:
        for b in intercepted_bursts:
            if b.emitter_id in predicted_intercept_times:
                t_pred = predicted_intercept_times[b.emitter_id]
                t_act = float(b.first_intercept_slot - b.start_slot)  # type: ignore
                ite_errors.append((t_pred - t_act) ** 2)

    ite = float(np.sqrt(np.mean(ite_errors))) if ite_errors else 0.0

    return MetricsResult(
        probability_of_detection=pd,
        false_alarm_rate=far,
        interception_ratio_count=ir_count,
        interception_ratio_time=ir_time,
        average_intercept_time=avg_intercept_time,
        censored_burst_count=num_censored,
        average_intercept_rate=avg_intercept_rate,
        intercept_time_error=ite,
        cumulative_reward=cumulative_reward,
        average_reward=cumulative_reward / max(1, len(observations)),
        total_bursts=total_bursts,
        intercepted_bursts=num_intercepted,
    )


class MultiSeedEvaluator:
    """
    Evaluates schedulers across identical seeds and scenarios,
    aggregating statistics with Student-t 95% Confidence Intervals.
    """

    def __init__(
        self,
        env_factory: Callable[[], SpectrumScanEnv],
        seeds: List[int],
    ) -> None:
        self.env_factory = env_factory
        self.seeds = seeds

    def evaluate_scheduler(self, scheduler: Scheduler) -> Dict[str, StatisticalSummary]:
        """Runs scheduler across all seeds and returns aggregated statistical summaries."""
        pd_list: List[float] = []
        far_list: List[float] = []
        ir_cnt_list: List[float] = []
        ir_time_list: List[float] = []
        ait_list: List[float] = []
        rate_list: List[float] = []
        reward_list: List[float] = []

        env = self.env_factory()

        for seed in self.seeds:
            obs, info = env.reset(seed=seed)
            scheduler.reset(seed=seed)

            total_reward = 0.0
            done = False

            while not done:
                action = scheduler.act(obs, info)
                obs, reward, terminated, truncated, step_info = env.step(action)
                scheduler.update(obs, action, reward, step_info)
                total_reward += reward
                done = terminated or truncated

            m = compute_episode_metrics(env, total_reward)

            pd_list.append(m.probability_of_detection)
            far_list.append(m.false_alarm_rate)
            ir_cnt_list.append(m.interception_ratio_count)
            ir_time_list.append(m.interception_ratio_time)
            ait_list.append(m.average_intercept_time)
            rate_list.append(m.average_intercept_rate)
            reward_list.append(m.cumulative_reward)

        return {
            "Probability of Detection (Pd)": calculate_statistical_summary(pd_list),
            "False Alarm Rate (FAR)": calculate_statistical_summary(far_list),
            "Interception Ratio (Count)": calculate_statistical_summary(ir_cnt_list),
            "Interception Ratio (Time)": calculate_statistical_summary(ir_time_list),
            "Avg Intercept Time (AIT)": calculate_statistical_summary(ait_list),
            "Avg Intercept Rate (/100 slots)": calculate_statistical_summary(rate_list),
            "Cumulative Reward": calculate_statistical_summary(reward_list),
        }
