import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from rf_env.environment import EWScanEnv
from baselines.sequential import SequentialSweep
from baselines.random_scan import RandomScan
from baselines.priority_sweep import PrioritySweep
from metrics.evaluator import MultiSeedEvaluator, AggregatedMetric


def run_phase2_comparison(num_seeds: int = 30, episode_length: int = 500) -> None:
    """
    Executes baseline comparison benchmark across 30 identical seeds,
    prints formatted results table with 95% CIs, and saves comparison plots.
    """
    num_bands = 16
    seeds = [1000 + i for i in range(num_seeds)]

    print(f"==========================================================================")
    print(f" EW SMART SCAN STRATEGY: PHASE 2 BASELINE BENCHMARK")
    print(f" Seeds: {num_seeds} | Bands: {num_bands} | Episode Length: {episode_length} slots")
    print(f"==========================================================================\n")

    # Define standard environment factory
    def make_env():
        return EWScanEnv(num_bands=num_bands, max_steps=episode_length)

    # Instantiate baselines
    # Assign pre-mission knowledge: threat emitters are known on band 15 (scan radar) and band 5, 11 (bursts)
    prior_priorities = {15: 10.0, 5: 5.0, 11: 6.0, 1: 2.0}
    schedulers = [
        SequentialSweep(num_bands=num_bands),
        RandomScan(num_bands=num_bands),
        PrioritySweep(num_bands=num_bands, band_priorities=prior_priorities),
    ]

    evaluator = MultiSeedEvaluator(env_factory=make_env, seeds=seeds)

    all_results: Dict[str, Dict[str, AggregatedMetric]] = {}
    for sched in schedulers:
        print(f"Evaluating {sched.name} across {num_seeds} identical seeds...")
        res = evaluator.evaluate_scheduler(sched)
        all_results[sched.name] = res

    # 1. Print formatted results table
    metric_names = [
        "Probability of Detection (Pd)",
        "False Alarm Rate (FAR)",
        "Interception Ratio (IR)",
        "Avg Intercept Time (AIT)",
        "Intercept Time Error (ITE)",
        "Average Episode Reward",
    ]

    header = f"{'Scheduler':<18} | {'Pd':<14} | {'FAR':<14} | {'IR':<14} | {'AIT (slots)':<15} | {'ITE':<14} | {'Avg Reward':<16}"
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

        print(f"{name:<18} | {pd_s:<14} | {far_s:<14} | {ir_s:<14} | {ait_s:<15} | {ite_s:<14} | {rew_s:<16}")
    print(divider + "\n")

    # 2. Generate publication-quality comparison plot
    os.makedirs("artifacts", exist_ok=True)
    plot_metrics = [
        ("Probability of Detection (Pd)", "Higher is better", True),
        ("Interception Ratio (IR)", "Higher is better", True),
        ("Avg Intercept Time (AIT)", "Lower is better (slots)", False),
        ("Average Episode Reward", "Higher is better", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    sched_names = list(all_results.keys())

    for idx, (m_name, subtitle, higher_better) in enumerate(plot_metrics):
        ax = axes[idx]
        means = [all_results[s][m_name].mean for s in sched_names]
        ci_errs = [
            (all_results[s][m_name].ci_upper - all_results[s][m_name].ci_lower) / 2.0
            for s in sched_names
        ]

        bars = ax.bar(sched_names, means, yerr=ci_errs, capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
        ax.set_title(f"{m_name}\n({subtitle})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Add data value labels on top of bars
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

    plt.suptitle("Phase 2: Baseline Schedulers Multi-Seed Benchmark (N=30 Seeds, 95% CI)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join("artifacts", "baseline_comparison.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Comparison plot successfully saved to: {output_path}")


if __name__ == "__main__":
    run_phase2_comparison(num_seeds=30, episode_length=500)
