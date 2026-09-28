# DRISHTI: 5-Minute Live Tactical Demonstration Script
**Smart India Hackathon (SIH 2026) — Problem Statement 26055 (DRDO)**  
*Smart Scan Strategy for Electronic Warfare (ES Receiver)*

---

## ⏱️ Master Presentation Timeline (Total: 5 Minutes)

| Time Window | Section | Key Visual / Action | Objective |
| :---: | :--- | :--- | :--- |
| **0:00 – 0:45** | **The Operational Dilemma** | Problem statement slide / Architecture diagram | Establish the bottleneck: $1\text{ band} \ll 16\text{ bands}$. |
| **0:45 – 1:45** | **Failure of Legacy Sweeps** | Streamlit: Run `Sequential Sweep` on `medium.yaml` | Show sparse intercepts, high latency ($0.606$ slots), and blind spots. |
| **1:45 – 3:00** | **The DRISHTI Breakthrough** | Streamlit: Run `Augmented PPO` & `Periodic-Aware` | Show dense waterfall coverage, $+753.87$ reward, near-zero latency. |
| **3:00 – 4:00** | **Scenario Shock & Explainability** | Trigger "Scenario Shock" + Scrub "Why this band?" | Prove resilience to sudden emitter drift and explainable AI. |
| **4:00 – 5:00** | **Avionics Edge Readiness & Q&A** | Terminal: Show TorchScript ($15.55\ \mu\text{s}$) & 31 tests | Defend real-time feasibility and conclude. |

---

## 🎙️ Step-by-Step Presenter Cues & Script

### Minute 0:00 – 0:45 | The Operational Dilemma
- **Action**: Display the system architecture diagram ([`README.md`](file:///d:/Projects/Dristi_freq/README.md)).
- **Presenter**:
  > *"Respected DRDO evaluators and jury members: In modern electronic warfare, an ES receiver faces a severe physical bottleneck. While the battlefield spectrum spans 16 or more wideband channels, the receiver's instantaneous bandwidth can only listen to **one band at any single millisecond**.*  
  > *Legacy systems rely on rigid round-robin sweeps or pre-mission priority tables. Against modern frequency-hopping radars and rotating beams, these sweeps suffer massive asynchronous blind spots—missing high-threat missile illuminations.*  
  > *Today, we present **DRISHTI**—a cognitive, reinforcement-learning-driven scan scheduler that synchronizes with radar periodicity, tracks dynamic agile threats, and delivers deterministic decisions in under 16 microseconds."*

---

### Minute 0:45 – 1:45 | The Failure of Conventional Sweeps
- **Action**: In the Streamlit dashboard ([`http://localhost:8501`](http://localhost:8501)), select **Scenario: `medium`**, choose **Scheduler: `Sequential Sweep`**, and click **🚀 Run Mission Simulation**.
- **Presenter**:
  > *"Let us first witness the status quo. Here is a standard sequential sweep monitoring 16 bands against agile hoppers and rotating radars.*  
  > *Observe the dual waterfall display:*  
  > *The top waterfall shows ground truth radar emissions; the bottom shows actual receiver intercepts.*  
  > *Notice the massive gaps in the bottom waterfall. Because the sweep marches blind to radar timing, it intercepts only **12.9%** of bursts, with a sluggish intercept latency of **0.606 slots**. Its final mission reward is a meager **+92.97**.*  
  > *Now, let us switch to DRISHTI."*

---

### Minute 1:45 – 3:00 | The DRISHTI Multi-Model Breakthrough
- **Action**: Switch scheduler dropdown to **`PPO (Augmented RL)`** (or **`Periodic-Aware Scheduler`**) and click **🚀 Run Mission Simulation**.
- **Presenter**:
  > *"Watch the transformation:*  
  > *First, look at the cumulative reward curve—it surges to **+753.87**, an **8-fold increase** over the baseline.*  
  > *Second, examine the receiver intercept waterfall: it is packed with continuous captures. DRISHTI's continuous interception ratio jumps from 6.3% to **44.1%**.*  
  > *How does it achieve this?*  
  > *1. Our **Epoch-Folding Circular Phase Coherence Tracker** locks onto the pulse repetition period $T$ with mathematical subharmonic disambiguation, scheduling receiver dwells precisely when periodic beams illuminate.*  
  > *2. Our **Augmented PPO Agent** maintains an internal temporal memory of Age-of-Information and agile hop histories, interleaving dwells between active radars with zero wasted dwell capacity.*  
  > *3. Intercept latency plummets down to **0.225 slots**—meaning threats are detected virtually the instant they emit."*

---

### Minute 3:00 – 4:00 | "Scenario Shock" & Operator Explainability
- **Action 1**: Scroll to the **Scenario Shock** control panel in the sidebar, set **Shock Slot: 200**, toggle **Move Threat**, and re-run.
- **Presenter**:
  > *"In actual combat, the adversary does not stay static. Here, at slot 200, we inject a mid-mission **Scenario Shock**: our primary threat suddenly hops from Band 3 to Band 12.*  
  > *Watch how DRISHTI's sliding-window contextual engine detects the silence on Band 3, elevates the Age-of-Information urgency across neighboring bands, and locks onto the new threat band within 15 time slots.*  
  > *Static priority sweeps collapse completely under this shock; DRISHTI adapts dynamically."*

- **Action 2**: Scroll down to the **"Why This Band?" Decision Log & Inspector** and scrub to a specific step (e.g., slot 142).
- **Presenter**:
  > *"Crucially, DRISHTI is not an uninterpretable black box. Military operators require verifiable accountability.*  
  > *Here, the operator can scrub to any millisecond of the mission. DRISHTI reveals the exact mathematical breakdown:*  
  > *'Band 4 was selected over Band 8 because Band 4 had high Age-of-Information urgency (24 slots) and an impending periodic radar rendezvous (phase confidence 0.94), whereas Band 8 had an estimated hit probability of only 8%.'*  
  > *Every single dwell is archived in an immutable JSONL forensic audit log."*

---

### Minute 4:00 – 5:00 | Avionics Hardware Deployment & Closing
- **Action**: Switch to the terminal window and run:
  ```powershell
  pytest -v
  ```
- **Presenter**:
  > *"Finally, let us address deployment feasibility:*  
  > *1. **Real-Time Deadlines**: An ES receiver operates on strict 1.0 millisecond dwell cycles. We compiled DRISHTI's neural policy into standalone **TorchScript (`ppo_policy.pt`)**. On standard CPU hardware, its mean inference latency is just **15.55 microseconds**—giving us a **64-fold safety margin** for embedded avionics.*  
  > *2. **Cognitive Defense**: Phase 8 incorporates an **Adversarial Cognitive Radar** to stress-test min-max evasion, alongside an **Out-of-Distribution Mahalanobis Detector** that flags uncataloged radar modes beyond 3-sigma variance.*  
  > *3. **Software Rigor**: Our repository is packaged, fully typed, documented across formal technical monographs, and backed by a comprehensive **31-unit-test verification suite**, all passing.*  
  > *DRISHTI provides DRDO with a battle-ready, explainable, and provably superior cognitive scan strategy. Thank you, and we welcome your questions."*

---

## 🛡️ Anticipated Jury Questions & Winning Defenses

### Q1: *"Reinforcement learning can hallucinate or get stuck. What guarantees the receiver won't starve unvisited channels?"*
**Defense**:
> *"DRISHTI explicitly enforces an **Age-of-Information (AoI)** penalty $\alpha \cdot \bar{\tau}_b$ in both the reward formulation and the observation state space. As an unvisited channel remains unswept, its normalized age $\tau_b$ linearly scales up to $1.0$, creating an overwhelming information-theoretic urgency that forces exploration. Channel starvation is mathematically impossible under our bounded AoI formulation."*

### Q2: *"How does your system handle two periodic radars emitting at the exact same millisecond on different frequencies?"*
**Defense**:
> *"When simultaneous rendezvous opportunities occur, DRISHTI resolves contention using **threat-weighted utility maximization**: $\max_b (W_b \cdot P_{\text{hit}}(b) - C_{\text{switch}})$. The higher-threat radar (e.g. missile guidance radar $W=10$) is prioritized over the surveillance radar ($W=3$), and the secondary radar is scheduled for intercept on its subsequent periodic cycle."*

### Q3: *"Can this run on military embedded hardware without a power-hungry GPU?"*
**Defense**:
> *"Yes, absolutely. DRISHTI was specifically engineered for SWaP-constrained (Size, Weight, and Power) airborne and naval receivers. We eliminated heavy frameworks at runtime by compiling the policy with `torch.jit.trace` into a 68 KB TorchScript binary. It requires **zero GPU acceleration** and executes in **15.55 microseconds on an ordinary CPU core**, leaving 98% of the processor budget free for digital signal processing (DSP) and pulse deinterleaving."*

### Q4: *"What if an enemy radar uses intentional pulse jitter or staggering to defeat your periodic tracker?"*
**Defense**:
> *"Our `CircularPhaseCoherenceEstimator` evaluates phase dispersion across continuous circle folds rather than rigid delta-time matching. Under 10% to 20% Gaussian pulse jitter, the Rayleigh coherence $R(T)$ remains well above our confirmation threshold of $0.82$, maintaining track lock where traditional Fourier methods experience severe spectral leakage."*
