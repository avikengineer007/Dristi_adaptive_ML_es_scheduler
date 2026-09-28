import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path so rf_env, baselines, and schedulers can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rf_env.environment import EWScanEnv
from baselines.sequential import SequentialSweep
from baselines.random_scan import RandomScan
from baselines.priority_sweep import PrioritySweep
from schedulers.bandits import SlidingWindowUCB, DiscountedThompsonSampling
from schedulers.periodic_tracker import PeriodicPredictiveScheduler
from schedulers.ppo_agent import PPOScheduler
from dashboard.components import run_single_episode_simulation, SchedulerSimulationResult

st.set_page_config(
    page_title="EW Smart Scan Strategy | SIH 2026",
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
        padding: 16px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
    }
    .stSelectbox label, .stSlider label, .stMultiSelect label {
        font-weight: 600;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📡 Smart Scan Strategy for Electronic Warfare (ES Receiver)")
st.markdown(
    "**SIH 2026 PS 26055 Prototype**: Adaptive reinforcement learning & non-stationary bandit frequency scheduling "
    "to maximize interception ratio and minimize intercept latency against frequency-agile and periodic pulse radars."
)

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.header("⚙️ Scenario & Receiver Parameters")

seed = st.sidebar.number_input("Random Scenario Seed", min_value=1, max_value=99999, value=1042, step=1)
num_bands = st.sidebar.slider("Number of Frequency Bands (N)", min_value=8, max_value=32, value=16, step=2)
episode_length = st.sidebar.slider("Episode Length (Time Slots)", min_value=50, max_value=1000, value=250, step=25)
dwell_time = st.sidebar.slider("Receiver Dwell Time (Slots/Step)", min_value=1, max_value=3, value=1, step=1)

with st.sidebar.expander("Receiver ROC & Noise Parameters", expanded=False):
    p_md = st.slider("Missed Detection Prob (P_md)", 0.0, 0.20, 0.05, 0.01)
    p_fa = st.slider("False Alarm Prob (P_fa)", 0.0, 0.10, 0.02, 0.005)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Schedulers to Compare")

available_schedulers = [
    "Sequential Sweep",
    "Random Scan",
    "Priority Sweep",
    "Sliding-Window UCB",
    "Discounted Thompson Sampling",
    "Periodic-Predictive ML",
    "PPO Deep RL",
]

selected_schedulers = st.sidebar.multiselect(
    "Select Schedulers to Run Side-by-Side:",
    options=available_schedulers,
    default=[
        "Sequential Sweep",
        "Sliding-Window UCB",
        "Periodic-Predictive ML",
        "PPO Deep RL",
    ],
)

run_button = st.sidebar.button("🚀 Run Live Mission Simulation", type="primary")

# ----------------- SIMULATION LOGIC -----------------
if not selected_schedulers:
    st.warning("Please select at least one scheduler from the sidebar.")
    st.stop()


def build_scheduler_instances(names: List[str], n_bands: int) -> List[Any]:
    schedulers = []
    b_scan = max(0, min(n_bands - 1, n_bands - 1))
    b_b1 = max(0, min(n_bands - 1, int(0.3 * n_bands)))
    b_b2 = max(0, min(n_bands - 1, int(0.7 * n_bands)))
    b_fixed = max(0, min(n_bands - 1, 1))
    prior_priorities = {b_scan: 10.0, b_b1: 5.0, b_b2: 6.0, b_fixed: 2.0}

    model_name = "ppo_ew_model.zip" if n_bands == 16 else f"ppo_ew_model_{n_bands}bands.zip"
    model_path = os.path.join("artifacts", model_name)

    for name in names:
        if name == "Sequential Sweep":
            schedulers.append(SequentialSweep(num_bands=n_bands))
        elif name == "Random Scan":
            schedulers.append(RandomScan(num_bands=n_bands))
        elif name == "Priority Sweep":
            schedulers.append(PrioritySweep(num_bands=n_bands, band_priorities=prior_priorities))
        elif name == "Sliding-Window UCB":
            schedulers.append(SlidingWindowUCB(num_bands=n_bands, window_size=50, exploration_coef=1.5))
        elif name == "Discounted Thompson Sampling":
            schedulers.append(DiscountedThompsonSampling(num_bands=n_bands, gamma=0.92))
        elif name == "Periodic-Predictive ML":
            schedulers.append(PeriodicPredictiveScheduler(num_bands=n_bands, window_size=40))
        elif name == "PPO Deep RL":
            ppo = PPOScheduler(num_bands=n_bands, model_path=model_path if os.path.exists(model_path) else None)
            if ppo.model is None:
                ppo.train_agent(total_timesteps=15000, save_path=model_path)
            schedulers.append(ppo)
    return schedulers


@st.cache_data(show_spinner=False)
def run_all_selected(
    _sched_names: List[str],
    s_seed: int,
    n_bands: int,
    ep_len: int,
    dwell: int,
    miss_prob: float,
    fa_prob: float,
) -> Dict[str, SchedulerSimulationResult]:
    env = EWScanEnv(
        num_bands=n_bands,
        max_steps=ep_len,
        dwell_time=dwell,
        p_md=miss_prob,
        p_fa=fa_prob,
    )
    schedulers = build_scheduler_instances(_sched_names, n_bands)
    results = {}
    for s in schedulers:
        res = run_single_episode_simulation(env=env, scheduler=s, seed=s_seed)
        results[s.name] = res
    return results


with st.spinner("Executing synchronous mission simulation across all chosen schedulers..."):
    results_map = run_all_selected(
        selected_schedulers,
        seed,
        num_bands,
        episode_length,
        dwell_time,
        p_md,
        p_fa,
    )

# ----------------- SECTION 1: TOP KPI CARDS -----------------
st.subheader("📊 Mission Key Performance Indicators (KPIs)")
cols = st.columns(4)

# Determine best performers
best_reward_name = max(results_map.keys(), key=lambda k: results_map[k].metrics.total_reward)
best_ir_name = max(results_map.keys(), key=lambda k: results_map[k].metrics.interception_ratio)
best_ait_name = min(results_map.keys(), key=lambda k: results_map[k].metrics.average_intercept_time)
max_bursts = max(r.metrics.total_bursts for r in results_map.values())

with cols[0]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Highest Mission Reward</div>
            <div class="metric-value">{results_map[best_reward_name].metrics.total_reward:.1f}</div>
            <div class="metric-subtitle">Leader: <strong>{best_reward_name}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[1]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Max Interception Ratio</div>
            <div class="metric-value">{results_map[best_ir_name].metrics.interception_ratio:.1%}</div>
            <div class="metric-subtitle">Leader: <strong>{best_ir_name}</strong> ({results_map[best_ir_name].metrics.intercepted_bursts}/{max_bursts} bursts)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[2]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Lowest Intercept Latency</div>
            <div class="metric-value">{results_map[best_ait_name].metrics.average_intercept_time:.2f} <span style="font-size:1rem;">slots</span></div>
            <div class="metric-subtitle">Leader: <strong>{best_ait_name}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with cols[3]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Scenario Bursts</div>
            <div class="metric-value">{max_bursts}</div>
            <div class="metric-subtitle">Agile + Periodic + Scan Emitters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ----------------- SECTION 2: LIVE CUMULATIVE REWARD & COMPARISON TABLE -----------------
col_chart, col_table = st.columns([1.4, 1.0])

with col_chart:
    st.markdown("### 📈 Cumulative Mission Reward Curves")
    fig_reward, ax_reward = plt.subplots(figsize=(8, 4.2))
    fig_reward.patch.set_facecolor("#0f172a")
    ax_reward.set_facecolor("#1e293b")

    for name, res in results_map.items():
        steps = list(range(len(res.cumulative_rewards)))
        ax_reward.plot(steps, res.cumulative_rewards, label=name, linewidth=2.0)

    ax_reward.set_xlabel("Time Slot (t)", color="#94a3b8", fontsize=10)
    ax_reward.set_ylabel("Cumulative Threat Reward", color="#94a3b8", fontsize=10)
    ax_reward.tick_params(colors="#94a3b8")
    ax_reward.grid(True, linestyle="--", alpha=0.3, color="#475569")
    ax_reward.legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    st.pyplot(fig_reward)
    plt.close(fig_reward)

with col_table:
    st.markdown("### 📋 Episode Metrics Table")
    rows = []
    for name, res in results_map.items():
        m = res.metrics
        rows.append(
            {
                "Scheduler": name,
                "Reward": f"{m.total_reward:.1f}",
                "IR": f"{m.interception_ratio:.1%}",
                "AIT": f"{m.average_intercept_time:.2f}",
                "Pd": f"{m.probability_of_detection:.2%}",
                "FAR": f"{m.false_alarm_rate:.2%}",
            }
        )
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, hide_index=True)
    st.info(
        "💡 **Key Insight**: Open-loop sweeps miss periodic and agile emitters due to asynchronous blind spots. "
        "ML schedulers estimate burst periodicity and dynamically re-tune to achieve 2x-3x higher interception ratio."
    )

st.write("---")

# ----------------- SECTION 3: SPECTRUM WATERFALL & DWELL TRAJECTORY -----------------
st.markdown("### 🛰️ Spectrum Waterfall & Receiver Scan Trajectory")
st.markdown(
    "Compare where each receiver dwells across the frequency channels over time. "
    "Background heatmap shows ground truth transmissions (grey = silent, gold = active emission). "
    "Overlaid markers show receiver dwells: **Green** = True Intercept, **Red** = False Alarm, **Cyan** = Dwell on Silent Band."
)

inspect_sched = st.selectbox(
    "Choose Scheduler to Inspect on Waterfall Display:",
    options=list(results_map.keys()),
    index=0,
)

res_inspect = results_map[inspect_sched]
gt_matrix = res_inspect.ground_truth_matrix
T_display = min(150, gt_matrix.shape[0])  # Display first 150 time slots for clear visibility

fig_wf, ax_wf = plt.subplots(figsize=(14, 5.5))
fig_wf.patch.set_facecolor("#0f172a")
ax_wf.set_facecolor("#0f172a")

# Plot Ground Truth Spectrogram (Transpose so Y is Band, X is Time)
ax_wf.imshow(
    gt_matrix[:T_display, :].T,
    aspect="auto",
    origin="lower",
    cmap="cividis",
    extent=[0, T_display, -0.5, num_bands - 0.5],
    alpha=0.65,
)

# Overlay scan trajectory
time_steps = list(range(T_display))
dwell_bands = res_inspect.actions[:T_display]

# Connect dwells with subtle line
ax_wf.plot(time_steps, dwell_bands, color="#38bdf8", alpha=0.35, linewidth=1.0, linestyle="--")

# Categorize scatter points
true_hits_t = [t for t in range(T_display) if res_inspect.true_detections[t]]
true_hits_b = [res_inspect.actions[t] for t in true_hits_t]

false_alarms_t = [t for t in range(T_display) if res_inspect.false_alarms[t]]
false_alarms_b = [res_inspect.actions[t] for t in false_alarms_t]

misses_t = [t for t in range(T_display) if not res_inspect.detections[t]]
misses_b = [res_inspect.actions[t] for t in misses_t]

ax_wf.scatter(misses_t, misses_b, color="#0284c7", s=25, alpha=0.6, label="Dwell (Silent / Miss)")
if false_alarms_t:
    ax_wf.scatter(false_alarms_t, false_alarms_b, color="#ef4444", s=55, marker="x", label="False Alarm")
if true_hits_t:
    ax_wf.scatter(true_hits_t, true_hits_b, color="#22c55e", s=65, marker="o", edgecolors="#ffffff", linewidths=1.2, label="True Intercept")

ax_wf.set_title(f"Receiver Scan Path: {inspect_sched} vs Spectrum Ground Truth (Seed {seed})", color="#f8fafc", fontsize=12, fontweight="bold")
ax_wf.set_xlabel("Time Slot (t)", color="#94a3b8", fontsize=10)
ax_wf.set_ylabel("Frequency Band Channel", color="#94a3b8", fontsize=10)
ax_wf.set_yticks(range(num_bands))
ax_wf.tick_params(colors="#94a3b8")
ax_wf.grid(True, linestyle=":", alpha=0.2, color="#94a3b8")
ax_wf.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

st.pyplot(fig_wf)
plt.close(fig_wf)

st.write("---")

# ----------------- SECTION 4: 'WHY THIS BAND?' EXPLANATION LOG -----------------
st.markdown("### 🧠 \"Why This Band?\" Operational Explanation Log")
st.markdown(
    "Inspect the real-time reasoning and internal decision variables behind every single band selection made by the agent."
)

col_log_ctrl, col_log_view = st.columns([1.0, 2.0])

with col_log_ctrl:
    inspect_step = st.slider(
        "Select Time Slot (t) to Inspect Decision:",
        min_value=0,
        max_value=len(res_inspect.actions) - 1,
        value=min(25, len(res_inspect.actions) - 1),
        step=1,
    )
    st.markdown(f"**Inspecting Time Slot**: `t = {inspect_step}`")
    st.markdown(f"**Ground Truth Active Channels**: `{np.where(gt_matrix[inspect_step])[0].tolist()}`")

with col_log_view:
    st.markdown("#### Real-time Agent Reasoning Comparison at `t = {}`".format(inspect_step))
    for s_name, res in results_map.items():
        band_chosen = res.actions[inspect_step]
        reason = res.explanations[inspect_step]
        detected = res.detections[inspect_step]
        emitters = res.detected_emitters[inspect_step]
        det_badge = "🟢 HIT" if detected else "⚪ SILENT"

        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#1e293b; border-left:4px solid #38bdf8; padding:10px 14px; border-radius:4px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#f8fafc;">{s_name}</span>
                        <span style="font-size:0.85rem; color:#94a3b8;">Tuned: <strong>Band {band_chosen}</strong> | {det_badge}</span>
                    </div>
                    <div style="font-size:0.88rem; color:#cbd5e1; margin-top:6px; font-family:monospace;">
                        {reason}
                    </div>
                    {f'<div style="font-size:0.78rem; color:#86efac; margin-top:4px;">Intercepted Emitters: {", ".join(emitters)}</div>' if emitters else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

with st.expander("🔍 Browse Full Step-by-Step Mission Event Log (Searchable Table)"):
    log_rows = []
    for t_idx in range(len(res_inspect.actions)):
        log_rows.append(
            {
                "Slot": t_idx,
                "Band": res_inspect.actions[t_idx],
                "Outcome": "Intercept" if res_inspect.true_detections[t_idx] else ("False Alarm" if res_inspect.false_alarms[t_idx] else "Silent"),
                "Step Reward": f"{res_inspect.rewards[t_idx]:.1f}",
                "Cumulative": f"{res_inspect.cumulative_rewards[t_idx]:.1f}",
                "Agent Explanation": res_inspect.explanations[t_idx],
            }
        )
    st.dataframe(pd.DataFrame(log_rows))
