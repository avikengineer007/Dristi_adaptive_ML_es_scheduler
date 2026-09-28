# Model Card: DRISHTI Tactical Scan Scheduler
**SIH 2026 Problem Statement 26055: Smart Scan Strategy for Electronic Warfare**

---

## 1. Model Details

- **Model Name**: DRISHTI (*Dynamic Reinforcement-learning Intercept Scheduler for Tactical ES Intelligence*)
- **Version**: 0.1.0-tactical
- **Model Type**: Multi-Tier Adaptive Decision Architecture (Multi-Armed Bandits + Circular Phase Coherence Tracking + Proximal Policy Optimization Reinforcement Learning)
- **Primary Authors**: DRISHTI Engineering Team (SIH 2026 DRDO PS 26055)
- **License**: Apache 2.0 / Open Defense Research
- **Frameworks**: Python 3.12, PyTorch 2.13, Stable-Baselines3 2.9, Gymnasium 1.3, TorchScript JIT

---

## 2. Intended Use & Tactical Domain

- **Primary Application**: Autonomous frequency channel selection for an Electronic Support (ES) receiver operating under instantaneous receiver bandwidth bottlenecks ($B$ discrete candidate channels, 1 channel monitored per dwell slot $\Delta t = 1.0\text{ ms}$).
- **Operational Target Profiles**:
  - Fixed frequency radars and communications emitters
  - Periodic pulsed radars (`PeriodicBurstEmitter`)
  - Frequency-hopping and agile transmitters (`FrequencyAgileEmitter`)
  - Mechanically and electronically rotating search radars (`ScanningEmitter`)
  - Periodic scanning surveillance receivers (`PeriodicScanReceiverTarget`)
- **Deployment Environments**:
  - Embedded avionics mission computers (LibTorch C++ / TorchScript runtime)
  - Ground-based mobile ELINT / COMINT interception units
  - Real-time FPGA/SDR host processing controllers

---

## 3. Observation Inputs & Feature Representations

The model processes a normalized continuous observation vector from the RF receiver:

| Feature Dimension | Description | Normalization Range |
| :--- | :--- | :--- |
| $\text{obs}[0 : B]$ | Channel Age-of-Information ($\tau_b / \tau_{\max}$) | $[0.0, 1.0]$ |
| $\text{obs}[B : 2B]$ | Historical Channel Hit Ratio ($h_b / v_b$) | $[0.0, 1.0]$ |
| $\text{obs}[2B : 3B]$ | Historical Channel Miss Ratio ($m_b / v_b$) | $[0.0, 1.0]$ |
| $\text{obs}[3B : 4B]$ | Binary Last Seen Indicator ($y_b$) | $\{0.0, 1.0\}$ |
| $\text{obs}[4B]$ | Current Receiver Tuned Band ($b_{t-1} / (B-1)$) | $[0.0, 1.0]$ |
| $\text{obs}[4B+1 : 5B+1]$ | Estimated Pulse Repetition Period ($\hat{T}_b / T_{\max}$) | $[0.0, 1.0]$ |
| $\text{obs}[5B+1 : 6B+1]$ | Forecasted Time-to-Next-Pulse ($\Delta \hat{t}_b / T_{\max}$) | $[0.0, 1.0]$ |
| $\text{obs}[6B+1 : 7B+1]$ | Circular Phase Coherence Rayleigh Score ($R_b$) | $[0.0, 1.0]$ |

---

## 4. Quantitative Performance Summary (30-Seed Benchmark)

Evaluated across **30 identical seeds** on the standard `medium` EW scenario ($B=16$, $T=500$ slots):

| Scheduler | Reward (95% CI) | Count $IR$ | Time $IR$ | Censored $AIT$ | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential Sweep** | -194.37 ± 12.79 | 0.046 | 0.040 | 1.18 slots | $0.2\ \mu\text{s}$ |
| **Random Scan** | +48.16 ± 16.90 | 0.132 | 0.065 | 1.00 slots | $0.5\ \mu\text{s}$ |
| **Priority Pre-Mission** | +268.00 ± 9.43 | 0.234 | 0.081 | 1.08 slots | $0.8\ \mu\text{s}$ |
| **Sliding-Window UCB** | +576.94 ± 4.06 | 0.076 | 0.283 | 0.59 slots | $8.2\ \mu\text{s}$ |
| **Discounted Thompson** | +340.45 ± 29.44 | 0.166 | 0.071 | 2.71 slots | $9.5\ \mu\text{s}$ |
| **Periodic-Aware Scheduler** | +608.11 ± 43.72 | **0.204** | 0.121 | **0.22 slots** | $18.4\ \mu\text{s}$ |
| **PPO (Augmented RL)** | **+753.87 ± 0.67** | 0.003 | **0.441** | **0.00 slots** | **15.55 $\mu\text{s}$** |

---

## 5. Real-Time Hardware Execution & Latency Guarantees

Tested over 1,000 single-step decision cycles on an Intel CPU using TorchScript JIT (`results/onnx_latency_benchmark.json`):
- **Mean Single-Step Latency**: **$15.55\ \mu\text{s}$** ($0.0156\text{ ms}$)
- **95th Percentile Latency**: **$16.81\ \mu\text{s}$**
- **99th Percentile Latency**: **$49.91\ \mu\text{s}$**
- **Max Jitter Peak**: **$88.40\ \mu\text{s}$**
- **EW Time-Slot Budget**: **$1000\ \mu\text{s}$ ($1.0\text{ ms}$)**
- **Real-Time Margin**: **$98.4\%$ headroom** remaining for RF front-end synthesizer settling and digital signal processing.

---

## 6. Explainability & Human-Machine Teaming

Every scheduling decision produces a structured decision record containing:
- **Rationale**: Natural-language reason (e.g. *"Predictive Synchronization: Periodic emission forecasted on band 7 (T=40, confidence=0.88)"*).
- **Component Attribution**: Detailed decomposition into empirical payoff, exploration bonus, Age-of-Information urgency, and switching penalties.
- **Contrastive Queries**: The `explain_contrastive(chosen_band, alternative_band)` method dynamically answers operator queries regarding channel priority trade-offs.
- **Forensic Audit**: Comprehensive logging to newline-delimited JSON (`.jsonl`) for post-mission debriefing.

---

## 7. Limitations & Operating Envelopes

1. **Instantaneous Bandwidth Limitation**: The model assumes an instantaneous receiver bandwidth covering a single channel. Wideband multi-channel receivers should deploy parallel instances of the policy across instantaneous IF segments.
2. **Thermal Noise & Low SNR**: At extreme negative SNRs ($\text{SNR} < -10\text{ dB}$), detection probability drops below $0.50$, degrading period estimation coherence. Longer observation windows ($K \ge 15$) are required in severe noise.
3. **Synthesizer Settling Time**: Hardware switching penalties are modeled as an additive penalty ($C_{\text{switch}} = 0.2$). Ultra-slow synthesizers ($> 500\ \mu\text{s}$ retuning time) may require configuring dead-time slots.
