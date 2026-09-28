# Smart Scan Strategy for Electronic Warfare (ES Receiver)
### Smart India Hackathon (SIH 2026) — Problem Statement 26055 Prototype

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Tactical--Dashboard-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-brightgreen.svg)]()

An intelligent, adaptive frequency scan scheduling engine for Electronic Support (ES) receivers operating under instantaneous bandwidth bottlenecks against frequency-agile and periodically scanning radar threats.

---

## 🎯 Executive Summary & Problem Context

In modern contested electromagnetic environments, an ES receiver's **instantaneous bandwidth is significantly narrower than the total RF spectrum** it must monitor. The receiver must make sequential, discrete-time tuning decisions: *which frequency band to dwell on next*.

Conventional open-loop sweeps (sequential round-robin or static priority tables):
- Waste valuable dwell capacity on silent or ambient channels.
- Suffer severe asynchronous blind spots, missing short-duration radar pulses.
- Fail against **frequency-agile (hopping)** countermeasures and **periodic scanning beams**.

This repository implements an **Adaptive ML & Signal-Processing Scan Scheduler** that:
1. **Doubles the Interception Ratio ($IR$)** against bursty, hopping, and rotating threats.
2. **Reduces Average Intercept Latency ($AIT$) to near-zero ($0.034$ time slots)**.
3. **Multiplies cumulative threat reward by $\approx 10\times$** compared to traditional sweeps.

---

## 🏗️ System Architecture

```text
                               +-------------------------------------+
                               |         RF World Simulator          |
                               |  - Fixed Emitters                   |
                               |  - Periodic Pulse Radars            |
                               |  - Frequency-Agile Hoppers          |
                               |  - Rotating Periodic-Scan Beams     |
                               +------------------+------------------+
                                                  | Ground Truth
                                                  v
+------------------------+            +------------------------------+
|     Agent Policy       |            |    Gymnasium EWScanEnv       |
| - Sequential / Random  | --Action-> |  - ROC Detection Physics     |
| - Priority Sweep       |    (Band)  |    (P_fa, P_md, Noise Floor) |
| - Sliding-Window UCB   |            |  - Feature Engineering       |
| - Discounted Thompson  | <-State--- |    (tau_b, hit_ema, last_seen|
| - Periodic-Predictive  |  & Reward  |  - Threat-Weighted Reward    |
| - PPO Deep RL Policy   |            +--------------+---------------+
+------------------------+                           |
                                                     v
                               +-------------------------------------+
                               |          Evaluation & Metrics       |
                               |  - Probability of Detection (Pd)    |
                               |  - False Alarm Rate (FAR)           |
                               |  - Interception Ratio (IR)          |
                               |  - Avg Intercept Time (AIT)         |
                               |  - 30-Seed 95% Confidence Intervals |
                               +-------------------------------------+
```

---

## 📊 Comprehensive 30-Seed Benchmark Results

Every scheduler was evaluated on **30 identical random seeds** ($N_{\text{seeds}} = 30$, $N_{\text{bands}} = 16$, $T = 500$ slots/episode) with $95\%$ Student-$t$ Confidence Intervals:

| Scheduler | Architecture / Paradigm | $P_d$ (Detection Prob) | $FAR$ (False Alarm Rate) | $IR$ (Interception Ratio) $\uparrow$ | $AIT$ (Avg Intercept Time, slots) $\downarrow$ | $ITE$ (Jitter) $\downarrow$ | **Avg Episode Reward** $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential Sweep** | Round-robin baseline | $0.950 \pm 0.008$ | $0.022 \pm 0.003$ | $0.129 \pm 0.005$ | $0.606 \pm 0.074$ | $0.858 \pm 0.196$ | $92.97 \pm 14.95$ |
| **Random Scan** | Uniform stochastic baseline | $0.954 \pm 0.009$ | $0.019 \pm 0.002$ | $0.118 \pm 0.007$ | $1.072 \pm 0.236$ | $2.688 \pm 1.046$ | $81.73 \pm 18.24$ |
| **Priority Sweep** | Static pre-mission intelligence | $0.956 \pm 0.008$ | $0.021 \pm 0.002$ | $0.105 \pm 0.005$ | $1.308 \pm 0.073$ | $2.657 \pm 0.092$ | $71.73 \pm 12.38$ |
| **Sliding-Window UCB** | Non-stationary MAB ($W=50$) | $0.949 \pm 0.007$ | $0.018 \pm 0.003$ | $0.113 \pm 0.009$ | $0.672 \pm 0.075$ | $0.954 \pm 0.373$ | **$276.67 \pm 20.24$** |
| **Discounted Thompson** | Recency discounted $\beta$-Bernoulli | $0.947 \pm 0.006$ | $0.018 \pm 0.003$ | $0.101 \pm 0.006$ | $0.980 \pm 0.129$ | $2.077 \pm 0.652$ | **$241.57 \pm 15.06$** |
| **PPO Deep RL** | Actor-Critic MLP Policy ($[64, 64]$) | $0.955 \pm 0.006$ | $0.020 \pm 0.002$ | **$0.260 \pm 0.002$** | **$0.034 \pm 0.007$** | **$0.172 \pm 0.020$** | **$830.97 \pm 15.81$** |
| **Periodic-Predictive ML** | Circular Coherence + Adaptive Bandit | $0.953 \pm 0.006$ | $0.019 \pm 0.002$ | **$0.297 \pm 0.043$** | **$0.375 \pm 0.080$** | **$0.693 \pm 0.257$** | **$911.23 \pm 159.38$** |

### Key Takeaways
1. **$9.8\times$ Threat Reward Boost**: ML schedulers aggressively dwell on confirmed threat emissions, avoiding empty spectrum.
2. **Interception Ratio Doubled**: Interception ratio increases from $12.9\%$ to **$29.7\%$**, successfully capturing high-threat agile bursts that baseline sweeps miss.
3. **Zero Latency Interception**: PPO reduces intercept latency to **$0.034$ time slots** ($17\times$ faster than sequential sweep), locking onto active radar transmissions almost instantaneously upon emission.

---

## 🔬 Mathematical Formulations

### 1. RF Environment & Threat-Weighted Reward
At step $t$, receiver chooses frequency band $b \in \{0, \dots, N-1\}$. Ground truth activity is $G[t, b] \in \{0, 1\}$.
- **Receiver Operating Characteristics (ROC)**:
  - If $G[t, b] = 1$: True Detection with probability $P_d = 1 - P_{md}$ ($P_{md} = 0.05$).
  - If $G[t, b] = 0$: False Alarm with probability $P_{fa} = 0.02$.
- **Reward Function**:
  $$R_t = \sum_{e \in \text{detected}(b, t)} W_e - C_{\text{dwell}} \cdot \Delta t_{\text{dwell}} - C_{fa} \cdot \mathbb{I}(\text{False Alarm})$$
  where $W_e \in [1, 10]$ is the threat priority weight of emitter $e$.

### 2. Signal Processing Periodic-Emitter Module (Circular Coherence)
For sparse detection timestamps $t_1, t_2, \dots, t_K$ recorded on band $b$, candidate period $T$ is evaluated using Epoch Folding circular phase coherence:
$$R(T) = \left| \frac{1}{K} \sum_{k=1}^K \exp\left(i \frac{2\pi (t_k \pmod{T})}{T}\right) \right| \in [0, 1]$$
When $R(T) > 0.82$, phase $\hat{\phi}$ and period $\hat{T}$ are locked. Dwells are scheduled preemptively at $t_{\text{pred}} = \hat{\phi} + k \hat{T}$ to catch pulses exactly as they illuminate.

### 3. Proximal Policy Optimization (PPO)
State vector $s_t \in \mathbb{R}^{3N}$:
$$s_t = \left[ \frac{\min(\tau_b, \tau_{\max})}{\tau_{\max}}, \quad \text{EMA\_Hit}_b, \quad \text{Last\_Seen}_b \right]_{b=0}^{N-1}$$
Trained via clipped surrogate objective $\mathcal{L}^{CLIP}(\theta)$ with generalized advantage estimation (GAE-$\lambda$).

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone repository and install requirements:
```bash
git clone https://github.com/example/Dristi_freq.git
cd Dristi_freq
pip install -r requirements.txt
```

### 2. Run All Unit Tests
Run the 24-test verification suite covering determinism, Gymnasium compliance, baselines, bandits, period estimation, and dashboard components:
```bash
python -m pytest tests/ -v
```

### 3. Run Multi-Seed Benchmark
Run the single-command 30-seed benchmark to print the metrics table and generate publication plots:
```bash
python benchmark.py --seeds 30 --bands 16 --steps 500
```
Plot artifact is saved to: `artifacts/benchmark_results.png`.

### 4. Launch Interactive Tactical Dashboard
Launch the Streamlit tactical radar visualizer:
```bash
streamlit run app.py
```
*Features*:
- Live Spectrum Waterfall Heatmap with overlaid receiver dwell path.
- Cumulative Mission Reward progression curves.
- **"Why This Band?" Decision Log**: Scrub through any time slot $t$ to inspect the real-time reasoning and internal variables of every agent.

---

## 📁 Repository Directory Structure

```text
Dristi_freq/
├── app.py                      # Root Streamlit entrypoint
├── benchmark.py                # Top-level single-command benchmark runner
├── requirements.txt            # Pinned dependencies
├── README.md                   # Full documentation & benchmark report
├── rf_env/                     # Gymnasium spectrum environment
│   ├── emitters.py             # Fixed, PeriodicBurst, FrequencyAgile, PeriodicScan emitters
│   ├── rf_world.py             # Physical spectrum simulator & ground truth event tracker
│   ├── environment.py          # EWScanEnv Gymnasium implementation
│   └── wrappers.py             # State transformation wrappers
├── baselines/                  # Operational baseline scan policies
│   ├── base.py                 # Abstract BaseScheduler with .act() and .explain()
│   ├── sequential.py           # Sequential round-robin sweep
│   ├── random_scan.py          # Uniform random scanner
│   └── priority_sweep.py       # Deterministic threat-weighted sweep
├── schedulers/                 # Machine learning & signal processing schedulers
│   ├── bandits.py              # Sliding-Window UCB & Discounted Thompson Sampling
│   ├── periodic_tracker.py     # Circular phase coherence tracker & hybrid scheduler
│   └── ppo_agent.py            # Stable-Baselines3 PPO wrapper & policy network
├── metrics/                    # EW evaluation engine
│   └── evaluator.py            # Pd, FAR, IR, AIT, ITE & Student-t 95% CIs
├── dashboard/                  # Streamlit tactical GUI
│   ├── app.py                  # Tactical mission control & waterfall interface
│   └── components.py           # Synchronous simulation engine
├── tests/                      # Automated test suite (24 unit tests)
└── artifacts/                  # Benchmark plots & pre-trained model weights
```
