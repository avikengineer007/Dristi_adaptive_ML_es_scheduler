import argparse
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drishti.utils.config import load_scenario_config, create_env_from_config
from drishti.baselines.base import Scheduler
from drishti.baselines.sequential import SequentialSweep
from drishti.baselines.random_scan import RandomScan
from drishti.baselines.priority_sweep import PriorityPreMissionSweep
from drishti.metrics.evaluator import MultiSeedEvaluator, StatisticalSummary


from drishti.schedulers.bandit import SlidingWindowUCB, DiscountedThompson


def build_baseline_schedulers(num_bands: int, config_data: Dict[str, Any]) -> List[Scheduler]:
    """
    Builds baseline schedulers with appropriate pre-mission priority intelligence.
    """
    return build_schedulers(num_bands=num_bands, config_data=config_data, include_bandits=False)


def build_schedulers(
    num_bands: int,
    config_data: Dict[str, Any],
    include_bandits: bool = True,
) -> List[Scheduler]:
    """
    Builds schedulers including baselines and adaptive bandits with pre-mission threat intelligence.
    """
    emitter_specs = config_data.get("emitters", [])
    priorities: Dict[int, float] = {}
    for spec in emitter_specs:
        threat = float(spec.get("threat_weight", 1.0))
        if "band" in spec:
            b = int(spec["band"])
            priorities[b] = max(priorities.get(b, 0.0), threat)
        elif "bands" in spec:
            for b in spec["bands"]:
                priorities[int(b)] = max(priorities.get(int(b), 0.0), threat)
        elif "hop_bands" in spec:
            for b in spec["hop_bands"]:
                priorities[int(b)] = max(priorities.get(int(b), 0.0), threat * 0.75)
        elif "listening_bands" in spec:
            for b in spec["listening_bands"]:
                priorities[int(b)] = max(priorities.get(int(b), 0.0), threat)

    schedulers: List[Scheduler] = [
        SequentialSweep(num_bands=num_bands),
        RandomScan(num_bands=num_bands),
        PriorityPreMissionSweep(num_bands=num_bands, band_priorities=priorities),
    ]

    if include_bandits:
        schedulers.append(
            SlidingWindowUCB(
                num_bands=num_bands,
                window_size=120,
                exploration_coef=0.5,
                aoi_weight=0.5,
                switch_penalty_weight=0.2,
                prior_weights=priorities,
            )
        )
        schedulers.append(
            DiscountedThompson(
                num_bands=num_bands,
                gamma=0.995,
                aoi_weight=2.0,
                switch_penalty_weight=0.2,
                prior_weights=priorities,
            )
        )

    return schedulers


def run_benchmark(
    config_name: str = "medium",
    num_seeds: int = 30,
    include_bandits: bool = True,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Runs multi-seed benchmark across schedulers on specified scenario,
    prints comparison table with 95% CIs, and exports CSV, Markdown, and plot artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "summaries"), exist_ok=True)

    config_data = load_scenario_config(config_name)
    num_bands = config_data.get("scenario", {}).get("num_bands", 16)
    episode_length = config_data.get("scenario", {}).get("max_steps", 500)
    seeds = [1000 + i for i in range(num_seeds)]

    print("\n" + "=" * 95)
    print(f" DRISHTI EW SCAN BENCHMARK | Scenario: '{config_name.upper()}'")
    print(f" Seeds: {num_seeds} | Channels: {num_bands} | Episode Length: {episode_length} slots | 95% Confidence Intervals")
    print("=" * 95 + "\n")

    def make_env():
        return create_env_from_config(config_data)

    schedulers = build_schedulers(num_bands, config_data, include_bandits=include_bandits)
    evaluator = MultiSeedEvaluator(env_factory=make_env, seeds=seeds)

    all_results: Dict[str, Dict[str, StatisticalSummary]] = {}
    for sched in schedulers:
        print(f"--> Evaluating {sched.name:<28} across {num_seeds} identical seeds...")
        res = evaluator.evaluate_scheduler(sched)
        all_results[sched.name] = res

    # 1. Build DataFrame & Formatted Tables
    metric_keys = [
        "Probability of Detection (Pd)",
        "False Alarm Rate (FAR)",
        "Interception Ratio (Count)",
        "Interception Ratio (Time)",
        "Avg Intercept Time (AIT)",
        "Cumulative Reward",
    ]

    table_rows = []
    export_rows = []

    for name, res in all_results.items():
        row_dict = {"Scheduler": name}
        export_dict = {"Scheduler": name}
        for k in metric_keys:
            stat = res[k]
            row_dict[k] = f"{stat.mean:.3f} +/- {stat.ci_half:.3f}"
            export_dict[f"{k}_mean"] = stat.mean
            export_dict[f"{k}_std"] = stat.std
            export_dict[f"{k}_ci95"] = stat.ci_half
        table_rows.append(row_dict)
        export_rows.append(export_dict)

    df_display = pd.DataFrame(table_rows)
    df_export = pd.DataFrame(export_rows)

    # Print ASCII table
    header = f"{'Scheduler':<28} | {'Pd':<14} | {'FAR':<14} | {'IR (Count)':<15} | {'IR (Time)':<14} | {'AIT (slots)':<15} | {'Reward':<16}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(header)
    print(divider)
    for _, r in df_display.iterrows():
        print(
            f"{r['Scheduler']:<28} | {r['Probability of Detection (Pd)']:<14} | {r['False Alarm Rate (FAR)']:<14} | "
            f"{r['Interception Ratio (Count)']:<15} | {r['Interception Ratio (Time)']:<14} | "
            f"{r['Avg Intercept Time (AIT)']:<15} | {r['Cumulative Reward']:<16}"
        )
    print(divider + "\n")

    # 2. Save CSV and Markdown summaries
    csv_path = os.path.join(output_dir, "summaries", f"benchmark_{config_name}.csv")
    md_path = os.path.join(output_dir, "summaries", f"benchmark_{config_name}.md")
    df_export.to_csv(csv_path, index=False)

    # Clean markdown table writer without external dependency
    md_headers = list(df_display.columns)
    md_lines = ["| " + " | ".join(md_headers) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(md_headers)) + " |")
    for _, r in df_display.iterrows():
        md_lines.append("| " + " | ".join(str(r[h]) for h in md_headers) + " |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Results summary saved to: {csv_path} and {md_path}")

    # 3. Generate comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#0b0f19")
    axes = axes.flatten()

    plot_configs = [
        ("Cumulative Reward", "Threat Detections - Penalties (Higher is better)", True),
        ("Interception Ratio (Count)", "Fraction of Bursts Caught (Higher is better)", True),
        ("Avg Intercept Time (AIT)", "Mean Intercept Latency in slots (Lower is better)", False),
        ("Probability of Detection (Pd)", "Receiver ROC Sensitivity (Higher is better)", True),
    ]

    sched_names = list(all_results.keys())
    colors = ["#64748b", "#94a3b8", "#3b82f6", "#10b981", "#8b5cf6"]

    for idx, (m_key, subtitle, higher_better) in enumerate(plot_configs):
        ax = axes[idx]
        ax.set_facecolor("#1e293b")
        means = [all_results[s][m_key].mean for s in sched_names]
        ci_halfs = [all_results[s][m_key].ci_half for s in sched_names]

        bars = ax.bar(
            range(len(sched_names)),
            means,
            yerr=ci_halfs,
            capsize=4,
            color=colors[: len(sched_names)],
            alpha=0.88,
            edgecolor="#ffffff",
            linewidth=0.8,
        )
        ax.set_title(f"{m_key}\n({subtitle})", color="#f8fafc", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#64748b")
        ax.set_xticks(range(len(sched_names)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in sched_names], color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#94a3b8")

        for bar, m_val in zip(bars, means):
            y_pos = bar.get_height()
            offset = max(means) * 0.03 if max(means) > 0 else 0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos + offset,
                f"{m_val:.2f}",
                ha="center",
                va="bottom",
                color="#f8fafc",
                fontsize=8,
                fontweight="bold",
            )

    plt.suptitle(
        f"DRISHTI Scan Strategy Benchmark: {config_name.upper()} Scenario (N={num_seeds} Seeds, 95% CI)",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"benchmark_{config_name}.png")
    plt.savefig(plot_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    # Also save as baseline_comparison_<config>.png for compatibility
    plt.savefig(os.path.join(output_dir, f"baseline_comparison_{config_name}.png"), dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Comparison plot saved to: {plot_path}\n")

    return df_display


def main():
    parser = argparse.ArgumentParser(description="Run DRISHTI multi-seed scan scheduler benchmark")
    parser.add_argument("--config", type=str, default="medium", help="Scenario config (easy, medium, hard, nonstationary)")
    parser.add_argument("--seeds", type=int, default=30, help="Number of evaluation seeds (default: 30)")
    parser.add_argument("--no-bandits", action="store_true", help="Exclude bandit schedulers")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    run_benchmark(
        config_name=args.config,
        num_seeds=args.seeds,
        include_bandits=not args.no_bandits,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
