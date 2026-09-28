import argparse
import os
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from rf_env.environment import EWScanEnv
from baselines.sequential import SequentialSweep
from baselines.random_scan import RandomScan
from baselines.priority_sweep import PrioritySweep
from schedulers.bandits import SlidingWindowUCB, DiscountedThompsonSampling
from schedulers.periodic_tracker import PeriodicPredictiveScheduler
from schedulers.ppo_agent import PPOScheduler
from metrics.evaluator import MultiSeedEvaluator, AggregatedMetric


def run_full_benchmark(
    num_seeds: int = 30,
    num_bands: int = 16,
    episode_length: int = 500,
    output_dir: str = "artifacts",
) -> Dict[str, Dict[str, AggregatedMetric]]:
    """
    Executes the standard EW Smart Scan Strategy benchmark across N identical seeds.
    Evaluates:
      1. Sequential Sweep (Round-Robin Baseline)
      2. Uniform Random Scan (Baseline)
      3. Pre-Mission Priority Sweep (Baseline)
      4. Sliding-Window UCB (Non-Stationary Bandit)
      5. Discounted Thompson Sampling (Non-Stationary Bandit)
      6. Periodic-Predictive ML (Signal Processing Coherence Tracker + Bandit)
      7. PPO Deep RL (Proximal Policy Optimization Policy)

    Prints a formatted metrics table with 95% Confidence Intervals and saves comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    seeds = [1000 + i for i in range(num_seeds)]
    model_name = "ppo_ew_model.zip" if num_bands == 16 else f"ppo_ew_model_{num_bands}bands.zip"
    model_path = os.path.join(output_dir, model_name)

    print("\n" + "=" * 90)
    print(" SIH 2026 PS 26055: SMART SCAN STRATEGY FOR ELECTRONIC WARFARE")
    print(f" BENCHMARK: {num_seeds} Seeds | {num_bands} Channels | {episode_length} Slots/Episode | 95% Confidence Intervals")
    print("=" * 90 + "\n")

    def make_env():
        return EWScanEnv(num_bands=num_bands, max_steps=episode_length)

    # 1. Ensure PPO model is trained for this band count
    ppo_scheduler = PPOScheduler(num_bands=num_bands, model_path=model_path)
    if not os.path.exists(model_path) or ppo_scheduler.model is None:
        print(f"Pre-trained PPO model for {num_bands} bands not found. Training PPO Agent...")
        ppo_scheduler.train_agent(total_timesteps=20000, save_path=model_path, seed=42)
    else:
        print(f"Loaded compatible PPO model from: {model_path}")

    # 2. Configure baseline and ML schedulers dynamically
    b_scan = max(0, min(num_bands - 1, num_bands - 1))
    b_b1 = max(0, min(num_bands - 1, int(0.3 * num_bands)))
    b_b2 = max(0, min(num_bands - 1, int(0.7 * num_bands)))
    b_fixed = max(0, min(num_bands - 1, 1))
    prior_priorities = {b_scan: 10.0, b_b1: 5.0, b_b2: 6.0, b_fixed: 2.0}
    schedulers = [
        SequentialSweep(num_bands=num_bands),
        RandomScan(num_bands=num_bands),
        PrioritySweep(num_bands=num_bands, band_priorities=prior_priorities),
        SlidingWindowUCB(num_bands=num_bands, window_size=50, exploration_coef=1.5),
        DiscountedThompsonSampling(num_bands=num_bands, gamma=0.92),
        PeriodicPredictiveScheduler(num_bands=num_bands, window_size=40),
        ppo_scheduler,
    ]

    evaluator = MultiSeedEvaluator(env_factory=make_env, seeds=seeds)
    all_results: Dict[str, Dict[str, AggregatedMetric]] = {}

    for sched in schedulers:
        print(f"--> Evaluating {sched.name:<26} across {num_seeds} identical seeds...")
        res = evaluator.evaluate_scheduler(sched)
        all_results[sched.name] = res

    # 3. Print formatted ASCII table
    header = f"{'Scheduler':<26} | {'Pd':<14} | {'FAR':<14} | {'IR':<14} | {'AIT (slots)':<15} | {'ITE':<14} | {'Avg Reward':<16}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(header)
    print(divider)

    for name, res in all_results.items():
        pd_s = f"{res['Probability of Detection (Pd)'].mean:.3f} +/- {((res['Probability of Detection (Pd)'].ci_upper - res['Probability of Detection (Pd)'].ci_lower)/2):.3f}"
        far_s = f"{res['False Alarm Rate (FAR)'].mean:.3f} +/- {((res['False Alarm Rate (FAR)'].ci_upper - res['False Alarm Rate (FAR)'].ci_lower)/2):.3f}"
        ir_s = f"{res['Interception Ratio (IR)'].mean:.3f} +/- {((res['Interception Ratio (IR)'].ci_upper - res['Interception Ratio (IR)'].ci_lower)/2):.3f}"
        ait_s = f"{res['Avg Intercept Time (AIT)'].mean:.3f} +/- {((res['Avg Intercept Time (AIT)'].ci_upper - res['Avg Intercept Time (AIT)'].ci_lower)/2):.3f}"
        ite_s = f"{res['Intercept Time Error (ITE)'].mean:.3f} +/- {((res['Intercept Time Error (ITE)'].ci_upper - res['Intercept Time Error (ITE)'].ci_lower)/2):.3f}"
        rew_s = f"{res['Average Episode Reward'].mean:.2f} +/- {((res['Average Episode Reward'].ci_upper - res['Average Episode Reward'].ci_lower)/2):.2f}"

        print(f"{name:<26} | {pd_s:<14} | {far_s:<14} | {ir_s:<14} | {ait_s:<15} | {ite_s:<14} | {rew_s:<16}")
    print(divider + "\n")

    # 4. Generate comparison plots
    plot_metrics = [
        ("Interception Ratio (IR)", "Higher is better (Bursts Caught / Total)", True),
        ("Average Episode Reward", "Higher is better (Threat Detections - Costs)", True),
        ("Avg Intercept Time (AIT)", "Lower is better (latency in slots)", False),
        ("Probability of Detection (Pd)", "Receiver ROC Pd", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    colors = ["#7f7f7f", "#bcbd22", "#17becf", "#1f77b4", "#9467bd", "#e377c2", "#2ca02c"]
    sched_names = list(all_results.keys())

    for idx, (m_name, subtitle, higher_better) in enumerate(plot_metrics):
        ax = axes[idx]
        means = [all_results[s][m_name].mean for s in sched_names]
        ci_errs = [
            (all_results[s][m_name].ci_upper - all_results[s][m_name].ci_lower) / 2.0
            for s in sched_names
        ]

        bars = ax.bar(
            range(len(sched_names)),
            means,
            yerr=ci_errs,
            capsize=4,
            color=colors,
            alpha=0.88,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.set_title(f"{m_name}\n({subtitle})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(sched_names)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in sched_names], fontsize=8)

        for bar, mean_val in zip(bars, means):
            y_pos = bar.get_height()
            offset = max(means) * 0.03 if max(means) > 0 else 0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos + offset,
                f"{mean_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.suptitle("EW Smart Scan Strategy: Benchmark Summary (N=30 Seeds, 95% Confidence Intervals)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "benchmark_results.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Comparison plot saved to: {output_path}\n")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EW Smart Scan Strategy Benchmark")
    parser.add_argument("--seeds", type=int, default=30, help="Number of evaluation seeds (default: 30)")
    parser.add_argument("--bands", type=int, default=16, help="Number of frequency bands (default: 16)")
    parser.add_argument("--steps", type=int, default=500, help="Episode length in slots (default: 500)")
    args = parser.parse_args()

    run_full_benchmark(num_seeds=args.seeds, num_bands=args.bands, episode_length=args.steps)
