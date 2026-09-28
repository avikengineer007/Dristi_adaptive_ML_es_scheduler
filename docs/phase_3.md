# Phase 3: Non-Stationary Multi-Armed Bandit Schedulers
**DRISHTI: Dynamic Reinforcement-learning Intercept Scheduler for Tactical ES Intelligence**
*SIH 2026 Problem Statement 26055 (DRDO)*

---

## 1. Executive Summary & Core Advancements

In Phase 2, we established that open-loop baseline sweeps (sequential, random, and priority pre-mission) suffer from an **interception gap**, missing over 75% of RF emitter bursts because they allocate receiver dwell slots blindly without closed-loop feedback.

In **Phase 3**, DRISHTI introduces **closed-loop adaptive scan schedulers** based on non-stationary multi-armed bandit (MAB) theory:
1. **Sliding-Window Upper Confidence Bound (`SlidingWindowUCB`)**:
   - Maintains a moving observation window $\mathcal{H}_W$ of past receiver outcomes to discard stale history.
   - Balances empirical threat payoff against UCB confidence bounds and Age-of-Information (AoI) revisit urgency.
2. **Discounted Thompson Sampling (`DiscountedThompson`)**:
   - Maintains a Beta-Bernoulli conjugate model per channel with geometric recency discounting $\gamma \in (0, 1)$.
   - Naturally adapts to time-varying emitter duty cycles and beam rotation intervals.
3. **Age-of-Information (AoI) Feature Weighting**:
   - Explicitly models channel uncertainty: bands left uninspected accumulate an AoI urgency bonus $\alpha_{\text{aoi}} \cdot \tau_b / \tau_{\max}$, preventing the scheduler from permanently starving silent or periodic channels.
4. **Switching Penalty Mitigation**:
   - Factors the receiver hardware retuning penalty $C_{\text{switch}}$ into arm selection to prevent high-frequency jitter.
5. **Human-Readable Explainability**:
   - Every selection outputs a structured rationale (e.g., *"AoI Urgency"*, *"High payoff exploitation"*, *"UCB exploration"*) with component attribution.

---

## 2. Mathematical Formulations

### 2.1 Sliding-Window UCB (SW-UCB)

Let $B$ be the number of instantaneous frequency bands. At step $t$, the scheduler considers observations within the trailing window of size $W$:
$$\mathcal{H}_W(t) = \{(s, b_s, r_s, y_s) \mid \max(0, t - W) \le s < t\}$$

For each band $b \in \{0, \dots, B-1\}$:
- **Window Pull Count**: $N_b(t, W) = \sum_{s \in \mathcal{H}_W(t)} \mathbb{I}(b_s = b)$
- **Window Empirical Mean**:
  $$\hat{\mu}_b(t, W) = \frac{1}{N_b(t, W)} \sum_{s \in \mathcal{H}_W(t), b_s = b} r_s \quad (\text{if } N_b(t, W) > 0)$$
- **Exploration Confidence Term**:
  $$U_b(t, W) = c \cdot \sqrt{\frac{2 \ln(\min(t, W))}{N_b(t, W)}}$$
- **Age-of-Information (AoI) Urgency**:
  $$A_b(t) = \alpha_{\text{aoi}} \cdot \frac{\tau_b(t)}{\tau_{\max}} \cdot \bar{w}_b^{\text{prior}}$$
  where $\tau_b(t)$ is the elapsed slots since band $b$ was last visited, and $\bar{w}_b^{\text{prior}}$ is normalized pre-mission threat intelligence.
- **Switching Cost Mitigation**:
  $$S_b(t) = - \beta_{\text{switch}} \cdot \mathbb{I}(b \neq b_{t-1})$$

**Arm Selection Policy**:
If any arm has $N_b(t, W) = 0$, forced exploration selects the unvisited arm with the highest prior threat and AoI. Otherwise:
$$b_t = \arg\max_{b \in \{0, \dots, B-1\}} \left[ \hat{\mu}_b(t, W) + U_b(t, W) + A_b(t) + S_b(t) \right]$$

---

### 2.2 Discounted Thompson Sampling (D-TS)

Maintains Beta parameters $(\alpha_b, \beta_b)$ for each channel $b$. At each step $t$:
1. **Geometric Discounting (Decaying Stale Evidence)**:
   $$\alpha_b \leftarrow \gamma \alpha_b + (1 - \gamma) \alpha_0, \quad \beta_b \leftarrow \gamma \beta_b + (1 - \gamma) \beta_0 \quad \forall b$$
   with uninformative prior $\alpha_0 = 1.0, \beta_0 = 1.0$.
2. **Reward Update for Chosen Band $a = b_t$**:
   $$\begin{cases} \alpha_a \leftarrow \alpha_a + \max(1.0, R_{\text{threat}}) & \text{if detection occurs } (y_t = 1) \\ \beta_a \leftarrow \beta_a + 1.0 & \text{if no detection } (y_t = 0) \end{cases}$$
3. **Posterior Sampling & Decision**:
   $$\theta_b \sim \text{Beta}(\alpha_b, \beta_b)$$
   $$V_b(t) = \theta_b \cdot \bar{w}_b^{\text{prior}} + \alpha_{\text{aoi}} \cdot \frac{\tau_b(t)}{\tau_{\max}} - \beta_{\text{switch}} \cdot \mathbb{I}(b \neq b_{t-1})$$
   $$b_t = \arg\max_b V_b(t)$$

---

## 3. Hyperparameter Tuning & Ablation Study

Tuning was conducted using separate seeds ($2000 - 2014$) to avoid data leakage into the evaluation benchmarks.
Optimal parameters discovered and saved to `configs/bandit_tuned.yaml`:

```yaml
sliding_window_ucb:
  window_size: 120
  exploration_coef: 0.5
  aoi_weight: 0.5
  switch_penalty_weight: 0.2
discounted_thompson:
  gamma: 0.995
  aoi_weight: 2.0
  alpha_0: 1.0
  beta_0: 1.0
  switch_penalty_weight: 0.2
```

### Key Ablation Insights (`results/bandit_ablation_window_gamma.png`):
- **Window Size $W$**: A short window ($W=15$) discards history too rapidly, causing the scheduler to oscillate excessively (Reward: $24.13$). As $W$ expands to $120$ slots, the scheduler maintains sufficient memory of periodic emitter burst cycles while remaining nimble to frequency changes, increasing reward to **$584.99$**.
- **Discount Factor $\gamma$**: Lower discount factors ($\gamma < 0.90$) lose track of longer periodic emitters. A high discount factor $\gamma = 0.995$ combined with strong AoI weighting ($\alpha_{\text{aoi}} = 2.0$) provides the best balance between tracking active radars and periodically scanning silent bands.

---

## 4. 30-Seed Benchmark Results Across 4 Scenarios

All experiments evaluated across **30 identical seeds** ($T = 500$ slots, $B = 16$ channels).
Mean values reported with Student-$t$ **95% Confidence Intervals**:

### Scenario: `medium` (Fixed, Periodic-Burst, Agile, & Scanning Emitters)
| Scheduler | $P_d$ | $FAR$ | Count $IR$ | Time $IR$ | Censored $AIT$ (slots) | Cumulative Reward |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential Sweep** | 0.999 ± 0.002 | 0.020 ± 0.003 | 0.046 ± 0.004 | 0.040 ± 0.001 | 1.181 ± 0.116 | -194.37 ± 12.79 |
| **Random Scan** | 0.998 ± 0.002 | 0.020 ± 0.003 | 0.132 ± 0.007 | 0.065 ± 0.002 | 1.002 ± 0.202 | 48.16 ± 16.90 |
| **Priority Pre-Mission**| 0.999 ± 0.001 | 0.020 ± 0.002 | 0.234 ± 0.004 | 0.081 ± 0.001 | 1.085 ± 0.018 | 268.00 ± 9.43 |
| **Discounted Thompson** | 0.999 ± 0.001 | 0.022 ± 0.003 | 0.166 ± 0.010 | 0.071 ± 0.004 | 2.714 ± 0.454 | 340.45 ± 29.44 |
| **Sliding-Window UCB** | **0.999 ± 0.001** | **0.020 ± 0.005** | 0.076 ± 0.008 | **0.283 ± 0.012** | **0.589 ± 0.085** | **576.94 ± 4.06** |

### Scenario: `nonstationary` (Dynamic Emitter Emergence & Frequency Switches)
| Scheduler | $P_d$ | $FAR$ | Count $IR$ | Time $IR$ | Censored $AIT$ (slots) | Cumulative Reward |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential Sweep** | 1.000 ± 0.001 | 0.019 ± 0.002 | 0.121 ± 0.022 | 0.062 ± 0.011 | 0.372 ± 0.099 | -72.83 ± 52.39 |
| **Random Scan** | 0.998 ± 0.003 | 0.019 ± 0.002 | 0.116 ± 0.008 | 0.063 ± 0.004 | 0.549 ± 0.054 | -66.57 ± 19.67 |
| **Priority Pre-Mission**| 1.000 ± 0.001 | 0.020 ± 0.003 | 0.241 ± 0.006 | 0.127 ± 0.003 | 0.515 ± 0.020 | 221.37 ± 11.98 |
| **Discounted Thompson** | 0.999 ± 0.001 | 0.021 ± 0.003 | 0.181 ± 0.011 | 0.145 ± 0.008 | 0.333 ± 0.040 | 371.02 ± 30.56 |
| **Sliding-Window UCB** | **0.999 ± 0.001** | **0.020 ± 0.003** | 0.177 ± 0.008 | **0.157 ± 0.005** | **0.071 ± 0.014** | **435.79 ± 26.76** |

### Scenario: `easy` (3 Static/Predictable Emitters)
- **Priority Pre-Mission**: Reward: $42.43 \pm 5.45$, Time $IR$: $0.119$
- **Discounted Thompson**: Reward: $724.54 \pm 55.43$, Time $IR$: $0.278$
- **Sliding-Window UCB**: Reward: **$883.78 \pm 9.58$**, Time $IR$: **$0.370$**, $AIT$: **$0.81\text{ slots}$**

### Scenario: `hard` (10 Dense Bursty & Agile Emitters)
- **Priority Pre-Mission**: Reward: $560.28 \pm 29.90$, Time $IR$: $0.054$
- **Discounted Thompson**: Reward: $1187.74 \pm 49.42$, Time $IR$: $0.080$
- **Sliding-Window UCB**: Reward: **$1374.57 \pm 63.23$**, Time $IR$: **$0.096$**, $AIT$: **$0.76\text{ slots}$**

---

## 5. Electronic Warfare Tactical Analysis

1. **Massive Payoff Dominance**:
   - `SlidingWindowUCB` achieves **$2.15\times$ higher reward** than the best pre-mission baseline on `medium` ($576.94$ vs $268.00$) and **$1.97\times$ higher reward** on `nonstationary` ($435.79$ vs $221.37$).
   - On `easy`, SW-UCB achieves a **$20.8\times$ improvement** ($883.78$ vs $42.43$).
2. **Sub-Slot Interception Latency**:
   - On `nonstationary`, SW-UCB achieves an $AIT$ of **$0.071\text{ slots}$** (down from $0.515\text{ slots}$ for Priority Sweep). It catches hostile transmissions almost the instant they appear.
3. **Continuous Dwell Coverage ($IR_{\text{time}}$)**:
   - On `medium`, Time $IR$ jumped from $0.081$ to **$0.283$** — a **$3.5\times$ improvement in continuous signal interception**.
4. **The Next Frontier: Periodic Pulse Alignment**:
   - While bandits excel at tracking channel occupancy probabilities, they lack a dedicated **temporal phase estimator** to synchronize dwell windows with the exact periodic recurrence of rotating radar beams or target surveillance receivers.
   - This directly motivates **Phase 4**: Circular phase coherence / epoch folding period estimation and lookahead dwell synchronization.
