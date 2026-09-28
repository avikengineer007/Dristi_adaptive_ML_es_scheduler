# DRISHTI: System Design Document
**Dynamic Reinforcement-learning Intercept Scheduler for Tactical ES Intelligence**  
**SIH 2026 Problem Statement 26055: Smart Scan Strategy for Electronic Warfare (DRDO)**

---

## 1. Formal Problem Definition

In an Electronic Warfare (EW) theater, an Electronic Support (ES) receiver monitors a broad RF frequency spectrum $F = [f_{\min}, f_{\max}]$ partitioned into $B$ discrete contiguous frequency channels:
$$\mathcal{B} = \{0, 1, 2, \dots, B - 1\}$$

Time advances in discrete slots $t \in \{0, 1, \dots, T - 1\}$ of duration $\Delta t$ (default $\Delta t = 1.0\text{ ms}$).

### 1.1 Receiver Dwell & Tuning Constraint
- **Instantaneous Bandwidth Bottleneck**: The ES receiver has high sensitivity but narrow instantaneous bandwidth. At step $k$, it selects a single frequency channel $b_k \in \mathcal{B}$ (or $K \ll B$ channels) to monitor for dwell duration $D_k \ge 1$ slots.
- **Switching Latency**: Re-tuning the local oscillator (LO) from band $b_{k-1}$ to band $b_k \ne b_{k-1}$ incurs a tuning penalty $C_{\text{switch}}$ (modeled as a reward cost and/or dead-time slot).

### 1.2 Ground Truth & Signal Interception Physics
At any time slot $t$, the true physical spectrum occupancy is represented by a binary matrix $G \in \{0, 1\}^{T \times B}$:
$$G[t, b] = \begin{cases} 1 & \text{if at least one emitter is actively transmitting on band } b \text{ at time } t \\ 0 & \text{otherwise} \end{cases}$$

When the ES receiver dwells on band $b$ at time $t$:
1. **True Intercept (Hit)**: An emitter is active ($G[t, b] = 1$) and detected by the receiver detector with probability $P_d(SNR) = 1 - P_{md}$.
2. **Missed Detection (Miss)**: An emitter is active ($G[t, b] = 1$) but receiver noise/fading causes the detector to fail to cross threshold with probability $P_{md}$.
3. **False Alarm (FA)**: No emitter is active ($G[t, b] = 0$), but ambient thermal noise fluctuations exceed the detection threshold with probability $P_{fa}$.
4. **Silent Channel (True Negative)**: No emitter is active ($G[t, b] = 0$) and no false alarm occurs with probability $1 - P_{fa}$.

### 1.3 Emitter Hierarchy
All emitters implement a common interface `Emitter` with seed-deterministic behavior:
1. **FixedEmitter**: Continuously active or follows a fixed duty cycle on an assigned channel.
2. **PeriodicBurstEmitter**: Transmits for $T_{\text{on}}$ slots every period $T_{\text{period}}$, with initial phase $\phi_0$ and optional pulse-to-pulse jitter $\delta \sim \mathcal{N}(0, \sigma_{\text{jitter}}^2)$.
3. **FrequencyAgileEmitter**: Hops between an assigned subset of bands according to a hopping pattern (uniform pseudorandom, cyclic, or Markov transition matrix) with hop dwell $T_{\text{hop}}$.
4. **ScanningEmitter**: Models a rotating spatial radar antenna illuminating the receiver with beamwidth $\theta_{\text{beam}}$ and mechanical/electronic scan period $T_{\text{scan}}$, optionally combined with frequency agility.
5. **PeriodicScanReceiverTarget**: Models the specific problem statement case where the entity being monitored is itself a scanning receiver whose listening/receptive window cycles periodically across bands. Interception requires a predictive *rendezvous strategy*.

---

## 2. POMDP Formulation (State, Action, Observation, Reward)

Because the receiver cannot observe unmonitored channels, the frequency scheduling problem is framed as a **Partially Observable Markov Decision Process (POMDP)**: $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$.

### 2.1 State Space ($\mathcal{S}$)
The true environment state $s_t \in \mathcal{S}$ is unobservable to the agent:
$$s_t = \Big( G[t, \cdot], \{ \phi_e(t), b_e(t) \}_{e=1}^E, \text{receiver\_tuned\_band} \Big)$$

### 2.2 Action Space ($\mathcal{A}$)
Discrete choice of channel to monitor next:
$$\mathcal{A} = \{0, 1, \dots, B - 1\}$$

### 2.3 Observation Space ($\Omega$)
The agent receives a normalized, strictly partially-observable feature vector $o_t \in \mathbb{R}^{4B + 1}$:
1. **Normalized Age of Information (Time Since Last Visit)**:
   $$\tilde{\tau}_b = \frac{\min(\tau_b, \tau_{\max})}{\tau_{\max}} \in [0, 1], \quad \forall b \in \mathcal{B}$$
2. **Running Hit Count Ratio**:
   $$h_b = \frac{\text{hits}_b}{\max(1, \text{visits}_b)} \in [0, 1], \quad \forall b \in \mathcal{B}$$
3. **Running Miss Count Ratio**:
   $$m_b = \frac{\text{misses}_b}{\max(1, \text{visits}_b)} \in [0, 1], \quad \forall b \in \mathcal{B}$$
4. **Last Observed Status**:
   $$y_b \in \{0, 1\}, \quad \forall b \in \mathcal{B} \quad (\text{1 if signal detected on last visit, 0 otherwise})$$
5. **Current Tuned Band**:
   $$\tilde{b}_{\text{current}} = \frac{b_{\text{current}}}{B - 1} \in [0, 1]$$

*Ground truth $G[t, b]$ is never exposed in the observation vector.*

### 2.4 Reward Function ($\mathcal{R}$)
$$R_t = \sum_{e \in \text{detected}(b_t, t)} W_e + R_{\text{first}}(e) - C_{\text{dwell}} \cdot D - C_{\text{switch}} \cdot \mathbb{I}(b_t \ne b_{t-1}) - C_{fa} \cdot \mathbb{I}(\text{False Alarm})$$
- $W_e \in [1, 10]$: Relative threat weight of emitter $e$ (e.g. fire control radar > surveillance radar > comms).
- $R_{\text{first}}(e)$: Bonus rewarded on the first intercept of an uncatalogued emitter in the episode.
- $C_{\text{dwell}}$: Energy / time expenditure per dwell slot.
- $C_{\text{switch}}$: Local oscillator re-tuning penalty when switching bands.
- $C_{fa}$: Penalty for reporting a false detection triggered by receiver noise.

---

## 3. Mathematical Metric Definitions

All metrics are evaluated against the true physical ground truth array $G[t, b]$ and logged burst events:

### 3.1 Probability of Detection ($P_d$)
Evaluated across all dwell opportunities where an active emitter was physically present on the monitored channel:
$$P_d = \frac{\sum_{t} \mathbb{I}(\text{True Detection at } t \mid G[t, b_t] = 1)}{\sum_{t} \mathbb{I}(G[t, b_t] = 1)}$$

### 3.2 False Alarm Rate ($FAR$)
Evaluated across all dwell opportunities where no emitter was transmitting on the monitored channel:
$$FAR = \frac{\sum_{t} \mathbb{I}(\text{Detection at } t \mid G[t, b_t] = 0)}{\sum_{t} \mathbb{I}(G[t, b_t] = 0)}$$

### 3.3 Interception Ratio ($IR$)
1. **Count-Based Interception Ratio ($IR_{\text{count}}$)**:
   Fraction of distinct emitter burst events $E_i$ intercepted at least once:
   $$IR_{\text{count}} = \frac{\sum_{i=1}^M \mathbb{I}(E_i \text{ intercepted})}{\text{Total emitter bursts } M}$$
2. **Time-Based Interception Ratio ($IR_{\text{time}}$)**:
   Fraction of total active emitter transmission time slots successfully monitored:
   $$IR_{\text{time}} = \frac{\sum_{t=0}^{T-1} \mathbb{I}(\text{True Detection at } t)}{\sum_{t=0}^{T-1} \sum_{b=0}^{B-1} G[t, b]}$$

### 3.4 Average Intercept Time ($AIT$) with Right-Censoring
For each emitter burst $i$, let $t_{\text{start}, i}$ be its emission onset and $t_{\text{first}, i}$ be the receiver's first intercept.
- For intercepted bursts: $\Delta t_i = t_{\text{first}, i} - t_{\text{start}, i}$.
- For never-intercepted bursts (right-censored): $\Delta t_i$ is recorded as censored with penalty duration $T_{\text{episode}} - t_{\text{start}, i}$.
$$\overline{AIT} = \frac{1}{|\text{Intercepted}|} \sum_{i \in \text{Intercepted}} (t_{\text{first}, i} - t_{\text{start}, i})$$

### 3.5 Intercept Time Error ($ITE$)
Evaluates prediction fidelity of the periodicity tracker / receiver model:
$$ITE = \sqrt{\frac{1}{K} \sum_{k=1}^K \left( t_{\text{predicted}, k} - t_{\text{actual}, k} \right)^2}$$

### 3.6 Confidence Intervals
Reported across $N \ge 30$ identical seeds using two-sided Student-$t$ distribution:
$$\mu \pm t_{1 - \alpha/2, N-1} \cdot \frac{s}{\sqrt{N}}$$

---

## 4. Default Scenario Parameters & Justifications

| Parameter | Value | Technical Justification |
| :--- | :--- | :--- |
| **Number of Bands ($B$)** | `16` | Enforces severe bandwidth bottleneck ($6.25\%$ instantaneous coverage) typical of tactical EW receivers. |
| **Slot Duration ($\Delta t$)** | `1.0 ms` | Natural resolution for radar pulse repetition intervals (PRI) and dwell times. |
| **Episode Length ($T$)** | `500 slots` | Captures multiple complete cycles of pulse bursts and antenna rotations. |
| **Receiver Dwell ($D$)** | `1 slot` | Standard single-dwell observation quantum. |
| **Switching Penalty ($C_{\text{switch}}$)** | `0.2` | Models synthesizer tuning latency. |
| **Miss Probability ($P_{md}$)** | `0.05` | Reflects $95\%$ sensitivity at nominal target SNR. |
| **False Alarm Prob ($P_{fa}$)** | `0.02` | Controlled false alarm baseline for Neyman-Pearson receiver detectors. |

---

## 5. Assumptions & Simplifications

1. **Quantized Channelization**: The spectrum is divided into non-overlapping contiguous frequency bins. Emitter transmissions fall squarely within channels or are mapped to discrete bins.
2. **Independent Receiver Noise**: Thermal noise and false alarms are independent across time slots and frequency bins.
3. **Deterministic Seed Control**: Synthetic scenario generators, emitter start phases, hopping sequences, and receiver noise draws are strictly seeded using separate NumPy random generators.
4. **Data Source Portability**: Real-world recorded spectrum waterfalls (CSV / binary IQ energy) can be ingested through `CSVSource` without modifying the environment or scheduler interface.
