import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from rf_env.environment import EWScanEnv
from baselines.sequential import SequentialSweep
from baselines.random_scan import RandomScan
from baselines.priority_sweep import PrioritySweep
from schedulers.bandits import SlidingWindowUCB, DiscountedThompsonSampling
from metrics.evaluator import MultiSeedEvaluator, AggregatedMetric


def run_phase3_comparison(num_seeds: int = 30, episode_length: int = 500) -> Dict[str, Dict[str, AggregatedMetric]]:
    """
    Executes comprehensive benchmark comparing Baselines vs Bandit Schedulers
    across identical seeds, printing results and generating publication-grade plots.
    """
    num_bands = 16
    seeds = [1000 + i for i in range(num_seeds)]

    print(f"==========================================================================")
    print(f" EW SMART SCAN STRATEGY: PHASE 3 BANDIT BENCHMARK")
    print(f" Seeds: {num_seeds} | Bands: {num_bands} | Episode Length: {episode_length} slots")
    print(f"==========================================================================\n")

    def make_env():
        return EWScanEnv(num_bands=num_bands, max_steps=episode_length)

    prior_priorities = {15: 10.0, 5: 5.0, 11: 6.0, 1: 2.0}
    schedulers = [
        SequentialSweep(num_bands=num_bands),
        RandomScan(num_bands=num_bands),
        PrioritySweep(num_bands=num_bands, band_priorities=prior_priorities),
        SlidingWindowUCB(num_bands=num_bands, window_size=50, exploration_coef=1.5),
        DiscountedThompsonSampling(num_bands=num_bands, gamma=0.92),
    ]

    evaluator = MultiSeedEvaluator(env_factory=make_env, seeds=seeds)
    all_results: Dict[str, Dict[str, AggregatedMetric]] = {}

    for sched in schedulers:
        print(f"Evaluating {sched.name:<28} across {num_seeds} identical seeds...")
        res = evaluator.evaluate_scheduler(sched)
        all_results[sched.name] = res

    # 1. Print formatted results table
    header = f"{'Scheduler':<28} | {'Pd':<14} | {'FAR':<14} | {'IR':<14} | {'AIT (slots)':<15} | {'ITE':<14} | {'Avg Reward':<16}"
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

        print(f"{name:<28} | {pd_s:<14} | {far_s:<14} | {ir_s:<14} | {ait_s:<15} | {ite_s:<14} | {rew_s:<16}")
    print(divider + "\n")

    # 2. Generate publication-quality comparison plot
    os.makedirs("artifacts", exist_ok=True)
    plot_metrics = [
        ("Interception Ratio (IR)", "Higher is better (Bursts Caught / Total)", True),
        ("Average Episode Reward", "Higher is better (Threat Detections - Costs)", True),
        ("Probability of Detection (Pd)", "Receiver ROC Pd", True),
        ("Avg Intercept Time (AIT)", "Lower is better (latency in slots)", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ["#7f7f7f", "#bcbd22", "#17becf", "#1f77b4", "#2ca02c"]
    sched_names = list(all_results.keys())

    for idx, (m_name, subtitle, higher_better) in enumerate(plot_metrics):
        ax = axes[idx]
        means = [all_results[s][m_name].mean for s in sched_names]
        ci_errs = [
            (all_results[s][m_name].ci_upper - all_results[s][m_name].ci_lower) / 2.0
            for s in sched_names
        ]

        bars = ax.bar(sched_names, means, yerr=ci_errs, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1.0)
        ax.set_title(f"{m_name}\n({subtitle})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(sched_names)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in sched_names], fontsize=9)

        for bar, mean_val in zip(bars, means):
            y_pos = bar.get_height()
            offset = max(means) * 0.03 if max(means) > 0 else 0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos + offset,
                f"{mean_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle("Phase 3: Baselines vs Adaptive Bandits (N=30 Seeds, 95% CI)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join("artifacts", "bandit_vs_baseline_comparison.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Comparison plot successfully saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    run_phase3_comparison(num_seeds=30, episode_length=500)
