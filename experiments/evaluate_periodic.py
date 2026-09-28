import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from drishti.models.periodicity import CircularPhaseCoherenceEstimator
from drishti.models.receiver_model import ReceiverPredictiveModel
from drishti.utils.config import load_scenario_config, create_env_from_config
from drishti.metrics.evaluator import MultiSeedEvaluator
from drishti.schedulers.periodic_aware import PeriodicAwareScheduler
from drishti.schedulers.bandit.sliding_window_ucb import SlidingWindowUCB
from drishti.baselines.priority_sweep import PriorityPreMissionSweep


def evaluate_period_estimation_error(
    true_periods: List[int] = [15, 25, 40, 60],
    num_trials: int = 50,
    output_path: str = "results/period_estimation_error.png",
):
    """
    Evaluates estimation error |T_est - T_true| as a function of the number of
    sparse observations K, generating a publication-grade error analysis plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    k_values = [3, 5, 8, 12, 18, 25]
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#1e293b")

    colors = ["#38bdf8", "#34d399", "#fbbf24", "#f43f5e"]

    for idx, true_T in enumerate(true_periods):
        mean_errors = []
        for K in k_values:
            errors = []
            for _ in range(num_trials):
                est = CircularPhaseCoherenceEstimator(min_period=5, max_period=100, coherence_threshold=0.65)
                est.reset()
                phase = int(rng.integers(0, true_T))
                # Generate K randomly observed pulses with missing slots
                sample_indices = sorted(rng.choice(range(K * 2), size=K, replace=False))
                for s in sample_indices:
                    slot = phase + s * true_T
                    est.add_observation(slot)

                t_est = est.estimated_period if est.estimated_period is not None else 0
                errors.append(abs(t_est - true_T))
            mean_errors.append(float(np.mean(errors)))

        ax.plot(
            k_values,
            mean_errors,
            marker="o",
            linewidth=2.2,
            label=f"Period T={true_T} slots",
            color=colors[idx % len(colors)],
        )

    ax.set_title(
        "Circular Phase Coherence: Period Estimation Error vs Observations K",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Sparse Detections (K)", color="#cbd5e1", fontsize=10)
    ax.set_ylabel("Mean Absolute Period Error |T_est - T_true| (slots)", color="#cbd5e1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Period estimation error curve saved to: {output_path}")


def evaluate_receiver_calibration(
    num_samples: int = 500,
    output_path: str = "results/receiver_calibration.png",
):
    """
    Evaluates learned receiver model calibration curve and Brier reliability score.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.default_rng(123)
    model = ReceiverPredictiveModel(num_bands=16)
    model.reset()

    for _ in range(num_samples):
        hit_ratio = float(rng.uniform(0.0, 1.0))
        tau_norm = float(rng.uniform(0.0, 1.0))
        last_seen = float(rng.choice([0.0, 1.0]))
        periodic_conf = float(rng.uniform(0.0, 1.0))

        pred = model.predict_hit_probability(0, hit_ratio, tau_norm, last_seen, periodic_conf)
        # Ground truth simulated with realistic logistic sensor noise
        true_prob = 1.0 / (1.0 + np.exp(-(-1.5 + 2.2 * hit_ratio + 1.1 * tau_norm + 1.2 * last_seen + 3.0 * periodic_conf)))
        actual = bool(rng.random() < true_prob)
        model.record_outcome(pred, actual)

    calib = model.compute_calibration_curve(num_bins=8)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#1e293b")

    # Perfectly calibrated diagonal line
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Perfect Calibration")
    if calib["bin_centers"]:
        ax.plot(
            calib["bin_centers"],
            calib["empirical_freqs"],
            marker="s",
            color="#38bdf8",
            linewidth=2.2,
            label=f"Receiver Model (Brier: {calib['brier_score']:.3f})",
        )

    ax.set_title("Receiver Detection Probability Calibration Curve", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", color="#cbd5e1", fontsize=10)
    ax.set_ylabel("Empirical Detection Frequency", color="#cbd5e1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Receiver calibration plot saved to: {output_path}")


def run_phase4_periodic_benchmark(
    scenario_name: str = "medium",
    num_seeds: int = 30,
):
    """
    Evaluates PeriodicAwareScheduler alongside SW-UCB and Priority Sweep
    across 30 identical seeds on the specified scenario.
    """
    config_data = load_scenario_config(scenario_name)
    num_bands = config_data.get("scenario", {}).get("num_bands", 16)
    seeds = [1000 + i for i in range(num_seeds)]

    def make_env():
        return create_env_from_config(config_data)

    emitter_specs = config_data.get("emitters", [])
    priorities = {}
    for spec in emitter_specs:
        threat = float(spec.get("threat_weight", 1.0))
        if "band" in spec:
            priorities[int(spec["band"])] = max(priorities.get(int(spec["band"]), 0.0), threat)
        elif "bands" in spec:
            for b in spec["bands"]:
                priorities[int(b)] = max(priorities.get(int(b), 0.0), threat)

    schedulers = [
        PriorityPreMissionSweep(num_bands=num_bands, band_priorities=priorities),
        SlidingWindowUCB(num_bands=num_bands, window_size=120, exploration_coef=0.5, aoi_weight=0.5, prior_weights=priorities),
        PeriodicAwareScheduler(num_bands=num_bands, window_size=120, exploration_coef=0.5, aoi_weight=0.5, prior_weights=priorities),
    ]

    evaluator = MultiSeedEvaluator(env_factory=make_env, seeds=seeds)
    print(f"\nEvaluating Periodic Synchronization on '{scenario_name.upper()}' (N={num_seeds} seeds)...")

    for s in schedulers:
        res = evaluator.evaluate_scheduler(s)
        reward_stat = res["Cumulative Reward"]
        cnt_ir_stat = res["Interception Ratio (Count)"]
        time_ir_stat = res["Interception Ratio (Time)"]
        ait_stat = res["Avg Intercept Time (AIT)"]
        print(f"  {s.name:<26} | Reward: {reward_stat.mean:>7.2f} +/- {reward_stat.ci_half:>5.2f} | Count IR: {cnt_ir_stat.mean:.3f} | Time IR: {time_ir_stat.mean:.3f} | AIT: {ait_stat.mean:.2f} slots")


if __name__ == "__main__":
    evaluate_period_estimation_error()
    evaluate_receiver_calibration()
    run_phase4_periodic_benchmark("medium", num_seeds=30)
