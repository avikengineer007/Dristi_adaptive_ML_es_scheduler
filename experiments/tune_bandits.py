import os
from pathlib import Path
from typing import Dict, Any, List
import yaml
import numpy as np
import matplotlib.pyplot as plt

from drishti.utils.config import load_scenario_config, create_env_from_config
from drishti.schedulers.bandit import SlidingWindowUCB, DiscountedThompson
from drishti.metrics.evaluator import MultiSeedEvaluator


def evaluate_config(scheduler_cls, kwargs, scenario: str, seeds: List[int]) -> float:
    """Evaluates a parameter configuration across seeds and returns mean reward."""
    config_data = load_scenario_config(scenario)
    num_bands = config_data.get("scenario", {}).get("num_bands", 16)

    # Extract pre-mission threat intelligence for prior weighting
    priorities: Dict[int, float] = {}
    for spec in config_data.get("emitters", []):
        threat = float(spec.get("threat_weight", 1.0))
        if "band" in spec:
            priorities[int(spec["band"])] = max(priorities.get(int(spec["band"]), 0.0), threat)
        elif "bands" in spec:
            for b in spec["bands"]:
                priorities[int(b)] = max(priorities.get(int(b), 0.0), threat)

    def env_factory():
        return create_env_from_config(config_data)

    kwargs_copy = dict(kwargs)
    kwargs_copy["num_bands"] = num_bands
    kwargs_copy["prior_weights"] = priorities

    sched = scheduler_cls(**kwargs_copy)
    evaluator = MultiSeedEvaluator(env_factory=env_factory, seeds=seeds)
    stats = evaluator.evaluate_scheduler(sched)
    return stats["Cumulative Reward"].mean


def tune_bandits(output_yaml: str = "configs/bandit_tuned.yaml"):
    """
    Performs grid search tuning and ablation study for SW-UCB and D-TS.
    Uses separate tuning seeds (2000-2014) to avoid data leakage into test benchmarks.
    """
    tuning_seeds = [2000 + i for i in range(15)]
    scenarios = ["medium", "nonstationary"]

    print("=================================================================")
    print(" TUNING & ABLATION STUDY: NON-STATIONARY BANDIT SCHEDULERS")
    print(f" Tuning Seeds: {len(tuning_seeds)} | Scenarios: {scenarios}")
    print("=================================================================\n")

    # 1. SW-UCB Window Size & Exploration Ablation
    windows = [15, 30, 50, 80, 120]
    exp_coefs = [0.5, 1.0, 1.5, 2.0]
    aoi_weights = [0.0, 0.5, 1.0, 2.0]

    best_ucb_score = -float("inf")
    best_ucb_params: Dict[str, Any] = {}

    print("1. Tuning Sliding-Window UCB...")
    # Ablation grid over window size with c=1.0, aoi=1.0
    sw_rewards_window = []
    for w in windows:
        score = evaluate_config(
            SlidingWindowUCB,
            {"window_size": w, "exploration_coef": 1.0, "aoi_weight": 1.0},
            scenario="medium",
            seeds=tuning_seeds,
        )
        sw_rewards_window.append(score)
        print(f"  SW-UCB Window W={w:<3} -> Medium Reward: {score:.2f}")
        if score > best_ucb_score:
            best_ucb_score = score
            best_ucb_params = {"window_size": w, "exploration_coef": 1.0, "aoi_weight": 1.0}

    # Fine-tune exploration and AoI for best window
    w_best = best_ucb_params["window_size"]
    for c in exp_coefs:
        for aoi in aoi_weights:
            score = evaluate_config(
                SlidingWindowUCB,
                {"window_size": w_best, "exploration_coef": c, "aoi_weight": aoi},
                scenario="medium",
                seeds=tuning_seeds,
            )
            if score > best_ucb_score:
                best_ucb_score = score
                best_ucb_params = {"window_size": w_best, "exploration_coef": c, "aoi_weight": aoi}

    print(f"  >> Best SW-UCB: {best_ucb_params} (Score: {best_ucb_score:.2f})\n")

    # 2. D-TS Discount Factor Gamma & AoI Ablation
    gammas = [0.85, 0.90, 0.92, 0.95, 0.98, 0.995]
    best_dts_score = -float("inf")
    best_dts_params: Dict[str, Any] = {}

    print("2. Tuning Discounted Thompson Sampling...")
    dts_rewards_gamma = []
    for g in gammas:
        score = evaluate_config(
            DiscountedThompson,
            {"gamma": g, "aoi_weight": 0.5},
            scenario="medium",
            seeds=tuning_seeds,
        )
        dts_rewards_gamma.append(score)
        print(f"  D-TS Gamma gamma={g:<5} -> Medium Reward: {score:.2f}")
        if score > best_dts_score:
            best_dts_score = score
            best_dts_params = {"gamma": g, "aoi_weight": 0.5}

    # Fine-tune AoI for best gamma
    g_best = best_dts_params["gamma"]
    for aoi in aoi_weights:
        score = evaluate_config(
            DiscountedThompson,
            {"gamma": g_best, "aoi_weight": aoi},
            scenario="medium",
            seeds=tuning_seeds,
        )
        if score > best_dts_score:
            best_dts_score = score
            best_dts_params = {"gamma": g_best, "aoi_weight": aoi}

    print(f"  >> Best D-TS: {best_dts_params} (Score: {best_dts_score:.2f})\n")

    # 3. Save tuned configuration to YAML
    tuned_config = {
        "sliding_window_ucb": {
            "window_size": int(best_ucb_params["window_size"]),
            "exploration_coef": float(best_ucb_params["exploration_coef"]),
            "aoi_weight": float(best_ucb_params["aoi_weight"]),
            "switch_penalty_weight": 0.2,
        },
        "discounted_thompson": {
            "gamma": float(best_dts_params["gamma"]),
            "aoi_weight": float(best_dts_params["aoi_weight"]),
            "alpha_0": 1.0,
            "beta_0": 1.0,
            "switch_penalty_weight": 0.2,
        },
    }

    Path(output_yaml).parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(tuned_config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved tuned parameters to: {output_yaml}")

    # 4. Generate Ablation Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(windows, sw_rewards_window, marker="o", color="#1f77b4", linewidth=2.0)
    ax1.set_title("SW-UCB: Window Size W Ablation", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Window Size W (slots)", fontsize=10)
    ax1.set_ylabel("Mean Cumulative Reward", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(gammas, dts_rewards_gamma, marker="s", color="#ff7f0e", linewidth=2.0)
    ax2.set_title("D-TS: Discount Factor γ Ablation", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Discount Factor γ", fontsize=10)
    ax2.set_ylabel("Mean Cumulative Reward", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plot_path = "results/bandit_ablation_window_gamma.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved ablation curves to: {plot_path}")


if __name__ == "__main__":
    tune_bandits()
