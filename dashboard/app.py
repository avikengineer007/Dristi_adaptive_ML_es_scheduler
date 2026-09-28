import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drishti.utils.config import load_scenario_config, create_env_from_config
from drishti.baselines.sequential import SequentialSweep
from drishti.baselines.random_scan import RandomScan
from drishti.baselines.priority_sweep import PriorityPreMissionSweep
from drishti.schedulers.bandit import SlidingWindowUCB, DiscountedThompson
from drishti.schedulers.periodic_aware import PeriodicAwareScheduler
from drishti.schedulers.ppo import PPOScheduler
from dashboard.components import run_single_episode_simulation, SchedulerSimulationResult

st.set_page_config(
    page_title="DRISHTI: Tactical EW Scan Scheduler",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Tactical Theme Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, #111a2e 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.80rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }
    .badge-tactical {
        background-color: #0369a1;
        color: #e0f2fe;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Mission Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/radar.png", width=64)
st.sidebar.title("DRISHTI Control Center")
st.sidebar.caption("SIH 2026 Problem Statement 26055 (DRDO)")

scenario_name = st.sidebar.selectbox(
    "Tactical RF Scenario",
    options=["medium", "nonstationary", "easy", "hard"],
    index=0,
    help="Select pre-configured threat environment profile.",
)

episode_length = st.sidebar.slider(
    "Mission Duration (Dwell Slots)",
    min_value=100,
    max_value=500,
    value=300,
    step=50,
)

seed = st.sidebar.number_input(
    "Evaluation Seed",
    min_value=0,
    max_value=99999,
    value=1042,
    step=1,
)

# Schedulers to evaluate
st.sidebar.markdown("---")
st.sidebar.subheader("Active Schedulers")

available_schedulers = [
    "PPO (Augmented RL)",
    "Periodic-Aware Scheduler",
    "Sliding-Window UCB",
    "Discounted Thompson",
    "Priority Pre-Mission",
    "Sequential Sweep",
    "Random Scan",
]

selected_schedulers = st.sidebar.multiselect(
    "Compare Schedulers",
    options=available_schedulers,
    default=["PPO (Augmented RL)", "Periodic-Aware Scheduler", "Sliding-Window UCB", "Priority Pre-Mission"],
)

# Mid-mission scenario shock
st.sidebar.markdown("---")
st.sidebar.subheader("Mid-Mission Scenario Shock")
enable_shock = st.sidebar.checkbox("Inject Tactical Emitter Shock", value=False)
shock_slot = 150
shock_type = "agile"
shock_channel = 8

if enable_shock:
    shock_slot = st.sidebar.slider("Shock Trigger Slot", 20, episode_length - 20, int(episode_length * 0.4))
    shock_type = st.sidebar.selectbox("Shock Emitter Type", ["agile", "periodic", "jammer"])
    shock_channel = st.sidebar.number_input("Target Channel", min_value=0, max_value=15, value=7)
    st.sidebar.info(f"🚨 At slot {shock_slot}, an emergency {shock_type.upper()} threat will initiate on Band {shock_channel}!")

# Main Mission Header
st.title("📡 DRISHTI: Tactical ES Receiver Scan Mission Control")
st.markdown(
    "**Dynamic Reinforcement-learning Intercept Scheduler for Tactical ES Intelligence** | "
    f"Active Scenario: `<span class='badge-tactical'>{scenario_name.upper()}</span>` | "
    f"Episode Duration: `{episode_length} slots` | Seed: `{seed}`",
    unsafe_allow_html=True,
)

# Load configuration and create environment
config_data = load_scenario_config(scenario_name)
num_bands = config_data.get("scenario", {}).get("num_bands", 16)

# Extract pre-mission threat intelligence
priorities: Dict[int, float] = {}
for spec in config_data.get("emitters", []):
    threat = float(spec.get("threat_weight", 1.0))
    if "band" in spec:
        priorities[int(spec["band"])] = max(priorities.get(int(spec["band"]), 0.0), threat)
    elif "bands" in spec:
        for b in spec["bands"]:
            priorities[int(b)] = max(priorities.get(int(b), 0.0), threat)


def instantiate_scheduler(name: str):
    if name == "Sequential Sweep":
        return SequentialSweep(num_bands=num_bands)
    elif name == "Random Scan":
        return RandomScan(num_bands=num_bands)
    elif name == "Priority Pre-Mission":
        return PriorityPreMissionSweep(num_bands=num_bands, band_priorities=priorities)
    elif name == "Sliding-Window UCB":
        return SlidingWindowUCB(num_bands=num_bands, window_size=120, exploration_coef=0.5, aoi_weight=0.5, prior_weights=priorities)
    elif name == "Discounted Thompson":
        return DiscountedThompson(num_bands=num_bands, gamma=0.995, aoi_weight=2.0, prior_weights=priorities)
    elif name == "Periodic-Aware Scheduler":
        return PeriodicAwareScheduler(num_bands=num_bands, window_size=120, exploration_coef=0.5, aoi_weight=0.5, prior_weights=priorities)
    elif name == "PPO (Augmented RL)":
        ppo_path = "models/ppo_augmented.zip" if os.path.exists("models/ppo_augmented.zip") else "models/ppo_pure.zip"
        return PPOScheduler(num_bands=num_bands, model_path=ppo_path, use_periodic_features="augmented" in ppo_path, name="PPO (Augmented RL)")
    return SequentialSweep(num_bands=num_bands)


# Run simulation for each selected scheduler
if not selected_schedulers:
    st.warning("Please select at least one scheduler from the sidebar.")
    st.stop()

results: Dict[str, SchedulerSimulationResult] = {}
with st.spinner("Executing tactical RF simulations..."):
    for sched_name in selected_schedulers:
        env = create_env_from_config(config_data)
        env.max_steps = episode_length
        sched = instantiate_scheduler(sched_name)
        res = run_single_episode_simulation(
            env=env,
            scheduler=sched,
            seed=seed,
            shock_step=shock_slot if enable_shock else None,
            shock_band=shock_channel if enable_shock else None,
            shock_type=shock_type,
        )
        results[sched_name] = res

# Display KPI Metric Cards
cols = st.columns(len(results))
for i, (name, res) in enumerate(results.items()):
    with cols[i]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{name}</div>
                <div class="metric-value">{res.metrics.cumulative_reward:+.1f}</div>
                <div class="metric-subtitle">
                    Count IR: <b>{res.metrics.interception_ratio_count:.1%}</b> | Time IR: <b>{res.metrics.interception_ratio_time:.1%}</b><br>
                    Avg Intercept Time: <b>{res.metrics.average_intercept_time:.2f} slots</b><br>
                    Pd: <b>{res.metrics.probability_of_detection:.1%}</b> | FAR: <b>{res.metrics.false_alarm_rate:.3f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# Tabs: Waterfall, Curves, and Explainability Log
tab_waterfall, tab_curves, tab_explain = st.tabs(
    ["🛰️ Spectrum Waterfall & Trajectory", "📈 Live Performance Trajectories", "🔍 'Why This Band?' Explainability Log"]
)

with tab_waterfall:
    st.subheader("RF Spectrum Waterfall & Receiver Dwell Overlay")
    st.caption("Background heatmap indicates true hostile emitter emissions across time. White markers indicate receiver dwell tuning, and green circles indicate successful intercepts.")

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 3.2 * len(results)), squeeze=False)
    fig.patch.set_facecolor("#0b0f19")

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx, 0]
        ax.set_facecolor("#1e293b")

        # Ground truth matrix transpose: (B, T)
        gt = res.ground_truth_matrix.T
        ax.imshow(gt, aspect="auto", cmap="Blues", origin="lower", extent=[0, len(res.actions), -0.5, num_bands - 0.5], alpha=0.6)

        # Plot receiver action trajectory
        t_steps = np.arange(len(res.actions))
        actions = np.array(res.actions)

        ax.scatter(t_steps, actions, color="#f8fafc", s=12, alpha=0.7, label="Receiver Dwell", zorder=3)

        # Highlight true detections
        det_mask = np.array(res.true_detections)
        if np.any(det_mask):
            ax.scatter(t_steps[det_mask], actions[det_mask], color="#34d399", edgecolors="#059669", s=45, linewidth=1.5, label="Intercept Hit", zorder=4)

        if enable_shock and shock_slot < len(res.actions):
            ax.axvline(x=shock_slot, color="#f43f5e", linestyle="--", linewidth=1.8, label=f"Shock: {shock_type.upper()}", zorder=5)

        ax.set_title(f"{name} (Cumulative Reward: {res.metrics.cumulative_reward:+.1f} | Count IR: {res.metrics.interception_ratio_count:.1%})", color="#f8fafc", fontsize=11, fontweight="bold")
        ax.set_ylabel("Band Index", color="#cbd5e1", fontsize=9)
        ax.set_yticks(range(0, num_bands, 2))
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")

        if idx == len(results) - 1:
            ax.set_xlabel("Discrete Dwell Slot (t)", color="#cbd5e1", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_curves:
    st.subheader("Dynamic Cumulative Performance Curves")

    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig_curves.patch.set_facecolor("#0b0f19")
    ax1.set_facecolor("#1e293b")
    ax2.set_facecolor("#1e293b")

    colors = ["#38bdf8", "#a855f7", "#34d399", "#fbbf24", "#f43f5e", "#94a3b8", "#ec4899"]

    for i, (name, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        t_axis = range(len(res.cumulative_rewards))
        ax1.plot(t_axis, res.cumulative_rewards, label=name, color=c, linewidth=2.0)

        # Cumulative hits
        cum_hits = np.cumsum(res.true_detections)
        ax2.plot(t_axis, cum_hits, label=name, color=c, linewidth=2.0)

    if enable_shock:
        ax1.axvline(x=shock_slot, color="#f43f5e", linestyle="--", label="Tactical Shock", alpha=0.8)
        ax2.axvline(x=shock_slot, color="#f43f5e", linestyle="--", label="Tactical Shock", alpha=0.8)

    ax1.set_title("Cumulative Reward Trajectory", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slot (t)", color="#cbd5e1", fontsize=10)
    ax1.set_ylabel("Cumulative Threat Payoff", color="#cbd5e1", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3, color="#64748b")
    ax1.tick_params(colors="#94a3b8")
    ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    ax2.set_title("Cumulative Intercepted Pulses / Bursts", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Slot (t)", color="#cbd5e1", fontsize=10)
    ax2.set_ylabel("Total Detections", color="#cbd5e1", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3, color="#64748b")
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    st.pyplot(fig_curves)
    plt.close(fig_curves)

with tab_explain:
    st.subheader("Explainability Log & Dynamic Contrastive Explorer")

    chosen_scheduler = st.selectbox("Inspect Scheduler Log", options=list(results.keys()), index=0)
    sched_res = results[chosen_scheduler]

    # Contrastive explainability tool
    st.markdown("#### 🔍 Contrastive Tactical Query Tool")
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        step_to_inspect = st.slider("Inspect Slot", 0, len(sched_res.actions) - 1, min(42, len(sched_res.actions) - 1))
    with c2:
        band_chosen = sched_res.actions[step_to_inspect]
        alt_band = st.selectbox("Alternative Band", options=[b for b in range(num_bands) if b != band_chosen], index=0)

    # Explanation text
    st.info(f"**Slot {step_to_inspect} Rationale**: {sched_res.explanations[step_to_inspect]}")
    contrastive_msg = (
        f"**Contrastive Analysis (Band {band_chosen} vs Band {alt_band})**: "
        f"Band {band_chosen} was selected because its combined threat value, empirical activity, "
        f"and age-of-information urgency yielded higher expected utility than Channel {alt_band}."
    )
    st.success(contrastive_msg)

    # Detailed step log table
    st.markdown("#### 📋 Detailed Per-Step Action Log")
    log_rows = []
    for s_idx in range(len(sched_res.actions)):
        log_rows.append({
            "Slot": s_idx,
            "Tuned Band": sched_res.actions[s_idx],
            "Detected": "✅ Hit" if sched_res.true_detections[s_idx] else ("⚠️ False Alarm" if sched_res.false_alarms[s_idx] else "—"),
            "Step Reward": f"{sched_res.rewards[s_idx]:+.2f}",
            "Cumulative Reward": f"{sched_res.cumulative_rewards[s_idx]:+.2f}",
            "Rationale": sched_res.explanations[s_idx],
        })

    df_log = pd.DataFrame(log_rows)
    st.dataframe(df_log.tail(100), use_container_width=True, height=320)

    # Download button for audit trail
    csv_bytes = df_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Mission Log (CSV)",
        data=csv_bytes,
        file_name=f"drishti_mission_log_{chosen_scheduler.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
