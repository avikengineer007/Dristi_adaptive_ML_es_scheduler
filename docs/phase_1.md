# DRISHTI: Phase 1 Summary Report
**RF Environment Simulator & Scenario Framework**  
**SIH 2026 Problem Statement 26055 (DRDO)**

---

## 1. Overview of Phase 1 Deliverables

In Phase 1, we implemented the high-fidelity, deterministic RF spectrum environment simulator for tactical Electronic Support (ES) receiver operations.

### 1.1 Core Modules Implemented
1. **Gymnasium Environment (`drishti/env/environment.py`)**:
   - `SpectrumScanEnv`: Discrete spectrum of $B$ channels with discrete time advancement $t \in [0, T-1]$.
   - Strict POMDP observation space $\Omega = [0, 1]^{4B + 1}$:
     - `time_since_last_visit` ($\tilde{\tau}_b = \min(\tau_b, \tau_{\max}) / \tau_{\max}$)
     - `running_hit_ratio` ($h_b = \text{hits}_b / \max(1, \text{visits}_b)$)
     - `running_miss_ratio` ($m_b = \text{misses}_b / \max(1, \text{visits}_b)$)
     - `last_seen_status` ($y_b \in \{0, 1\}$)
     - `current_tuned_band` ($\tilde{b}_{\text{current}} = b / (B - 1)$)
   - Reward structure incorporating threat priorities ($W_e \in [1, 10]$), one-time new emitter first-intercept bonuses ($R_{\text{first}} = 5.0$), linear dwell costs ($C_{\text{dwell}} \cdot D$), local oscillator re-tuning switching penalties ($C_{\text{switch}} = 0.2$), and false alarm penalties ($C_{fa} = 1.0$).
   - Full episode ground truth matrix $G \in \{0, 1\}^{T \times B}$ and continuous burst event tracking (`BurstRecord`).

2. **Emitter Hierarchy (`drishti/env/emitters.py`)**:
   - `Emitter`: Abstract base class with seed-deterministic RNG reset and emission generation.
   - `FixedEmitter`: Fixed frequency continuous or steady-duty cycle radar.
   - `PeriodicBurstEmitter`: Pulsed radar with burst duration $T_{\text{on}}$, repetition period $T_{\text{period}}$, initial phase $\phi_0$, and Gaussian pulse jitter $\delta$.
   - `FrequencyAgileEmitter`: Agile hopper switching bands across candidate channels via `cyclic`, `random`, or `markov` transition models with dwell $T_{\text{hop}}$.
   - `ScanningEmitter`: Rotating radar antenna illuminating the ES receiver direction once per $T_{\text{scan}}$ for $T_{\text{beam}}$ slots, with carrier agility.
   - `PeriodicScanReceiverTarget`: Models the specific target case in PS 26055—a rotating scanning receiver target whose listening window moves across channels, requiring a predictive rendezvous strategy.

3. **Receiver Channel Physics (`drishti/env/receiver.py`)**:
   - `ReceiverModel`: Simulates realistic detector operating characteristics (ROC):
     - Effective SNR with dwell integration gain $\Delta SNR = 5 \log_{10}(D)$.
     - Sigmoidal detection probability $P_d(SNR_{\text{eff}})$.
     - Controlled thermal noise false alarm probability $P_{fa}$ per slot per channel.

4. **Data Source Adapter (`drishti/data/adapter.py`)**:
   - `DataSource`: Abstract base class.
   - `SyntheticSource`: Procedurally instantiates emitters from YAML configs.
   - `CSVSource`: Ingests offline real-world/synthetic recordings (`time_slot`, `band`, `power_dbm`, `emitter_id`, `threat_weight`).

5. **Scenario Configurations (`configs/`)**:
   - `configs/easy.yaml`: 3 fixed/slow periodic emitters.
   - `configs/medium.yaml`: 5 emitters (fixed, periodic bursts, agile hopper, scanning radar).
   - `configs/hard.yaml`: 6 high-density emitters including Markov hopper, agile search radar, and `PeriodicScanReceiverTarget`.
   - `configs/nonstationary.yaml`: Emitters alter hopping sets and schedules mid-mission at step $t=250$.

6. **Visualization Script (`experiments/render_episode.py`)**:
   - Generates publication-quality spectrum-time ground truth heatmaps.
   - Executable via: `python -m experiments.render_episode --config hard --seed 1`.
   - Output saved to: `results/spectrum_heatmap_hard_seed1.png`.

---

## 2. Verification & Test Suite

All 4 specialized Phase 1 tests pass in **0.49s** (`tests/test_phase1.py`), bringing the overall test suite to **28 passing tests** (2.01s):
- `test_seed_determinism_identical_trajectories`: Verifies identical seeds produce bitwise-identical ground truth matrices and observations.
- `test_ground_truth_matches_emitter_definitions`: Confirms zero-jitter periodic emitters match mathematical timing.
- `test_observations_never_leak_ground_truth`: Proves the agent observation vector exposes only past dwell history.
- `test_reward_components_sum_correctly_on_mini_scenario`: Validates threat reward, first-intercept bonus, dwell cost, and switch penalty arithmetic.

---

## 3. Acceptance Criteria Satisfied

- [x] All 28 unit tests pass (`python -m pytest`).
- [x] Command `python -m experiments.render_episode --config hard --seed 1` successfully renders and saves the ground truth heatmap.
- [x] 4 scenario YAML configurations verified.
- [x] Data adapter documentation and stub created.
