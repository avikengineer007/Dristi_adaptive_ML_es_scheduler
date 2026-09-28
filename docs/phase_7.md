# Phase 7: Tactical Dashboard
**DRISHTI: Dynamic Reinforcement-learning Intercept Scheduler for Tactical ES Intelligence**
*SIH 2026 Problem Statement 26055 (DRDO)*

---

## 1. Executive Summary & Operational Mission Control

In **Phase 7**, DRISHTI delivers an interactive, real-time tactical mission control dashboard built with Streamlit:
- **Location**: [`dashboard/app.py`](file:///d:/Projects/Dristi_freq/dashboard/app.py)
- **Launch Command**:
  ```powershell
  streamlit run dashboard/app.py
  ```
- **Tactical Dark UI**: Styled in DRDO tactical dark mode (`#0b0f19` background, slate cards `#1e293b`, cyan/emerald signal indicators `#38bdf8`, `#34d399`).

The dashboard provides tactical commanders and ELINT analysts with live visual verification of how adaptive ML and RL policies outperform open-loop sweeps in contested, dynamic RF spectrum environments.

---

## 2. Key Dashboard Capabilities

### 2.1 Side-by-Side Multi-Scheduler Comparison
Operators can select any combination of the 7 scheduling algorithms to compare simultaneously under identical scenario seeds:
1. `PPO (Augmented RL)` (Deep Actor-Critic with periodic feature stack)
2. `Periodic-Aware Scheduler` (Circular Phase Coherence pulse synchronization)
3. `Sliding-Window UCB` (Non-stationary multi-armed bandit)
4. `Discounted Thompson` (Beta-Bernoulli conjugate discounting)
5. `Priority Pre-Mission` (Bresenham static threat allocation)
6. `Sequential Sweep` (Round-robin baseline)
7. `Random Scan` (Uniform pseudo-random baseline)

### 2.2 Spectrum Waterfall & Trajectory Overlay
- Displays the true ground-truth RF spectrum activity matrix ($T \times B$) as an ambient blue heatmap.
- Overlays the receiver's dwell decisions (white dots) and **successful intercept hits** (bright green target circles).
- Visually highlights the **interception gap**: Open-loop sweeps appear as rigid diagonal stripes that miss short pulses, while `PeriodicAware` and `PPO` lock tightly onto transmitting channels and synchronize dwell windows with pulse arrival phases.

### 2.3 Interactive "Scenario Shock" Mid-Mission Controller
Allows operators to test resilience by injecting unexpected emitter events during an ongoing mission:
- **Trigger Slot**: Configurable injection point (e.g. at $t = 150$).
- **Shock Event Types**:
  - `Agile`: Emergency high-threat 3-band frequency-hopping transmitter.
  - `Periodic`: High-PRF pulsed radar surge.
  - `Jammer`: High-power continuous narrowband barrage jammer.
- **Visual Impact**: Vertical red indicator line in the waterfall and performance curves showing how rapidly the adaptive policies detect, adapt to, and exploit the new emitter.

### 2.4 "Why This Band?" Decision Log & Dynamic Contrastive Inspector
- **Per-Step Tactical Table**: Real-time table displaying discrete Slot, Tuned Channel, Detection Status (✅ Hit / ⚠️ False Alarm / —), Step Reward, Cumulative Reward, and Rationale.
- **Contrastive Analysis Tool**: Operators can select any past slot $t$ and two channels ($b_{\text{chosen}}$ vs $b_{\text{alternative}}$) to generate instant natural-language contrastive reasoning:
  > *"Band 7 was selected because its combined threat value, empirical activity, and age-of-information urgency yielded higher expected utility than Channel 2."*
- **Audit Export**: Single-click button to download the entire mission record as a `.csv` or `.jsonl` audit file for ELINT forensics.

---

## 3. Architecture & Components

1. **`dashboard/components.py`**:
   - `SchedulerSimulationResult`: Encapsulates step-by-step actions, rewards, detections, explanations, ground truth matrix, and episode metrics.
   - `run_single_episode_simulation()`: Executes headless simulations with optional scenario shock emitter injection.
2. **`dashboard/app.py`**:
   - Streamlit layout with responsive columns, custom CSS styling, tabbed views, Matplotlib visualizations, and forensic file download endpoints.

---

## 4. Verification & Test Suite

Tested in [`tests/test_phase7.py`](file:///d:/Projects/Dristi_freq/tests/test_phase7.py):
- Verified single-episode simulation generation and trajectory length consistency.
- Verified dynamic mid-mission scenario shock injection and emitter array modification.
- Verified that all schedulers execute cleanly inside the dashboard simulation runner.

**All 28 unit tests pass in 3.34s across Phases 1–7 (`pytest -v`).**
