# DRISHTI: Project Master Plan & Phase Checklist
**Dynamic Reinforcement-learning Intercept Scheduler for Tactical ES Intelligence**
**SIH 2026 Problem Statement 26055 (DRDO)**

---

## 📋 Phase Roadmap & Status Checklist

- [x] **Phase 0: Setup and Design Doc**
  - [x] Create standardized repo skeleton (`drishti/`, `configs/`, `experiments/`, `dashboard/`, `docs/`, `results/`)
  - [x] Create `pyproject.toml` with editable install support (`pip install -e .`)
  - [x] Update `requirements.txt`, `.gitignore`, and pytest config
  - [x] Author comprehensive formal design document (`docs/design.md`)
  - [x] Create `PLAN.md` and `CLAUDE.md`
  - [x] Initial Phase 0 git commit and user review

- [x] **Phase 1: RF Environment Simulator**
  - [x] Implement `drishti/env/` Gymnasium environment (`SpectrumScanEnv`)
  - [x] Emitter hierarchy (`FixedEmitter`, `PeriodicBurstEmitter`, `FrequencyAgileEmitter`, `ScanningEmitter`, `PeriodicScanReceiverTarget`)
  - [x] Receiver sensor model (ROC curve, $P_d(SNR)$, $P_{fa}$, thermal noise floor)
  - [x] Strict POMDP observation space (normalized age of information, running hits/misses, last seen status)
  - [x] Ground truth logging array ($T \times B$) & continuous burst event tracking
  - [x] 4 YAML scenario configs (`configs/easy.yaml`, `configs/medium.yaml`, `configs/hard.yaml`, `configs/nonstationary.yaml`)
  - [x] `DataSource` adapter (`SyntheticSource` and extensible `CSVSource` stub)
  - [x] Script `experiments/render_episode.py` saving spectrum-time heatmap to `results/`
  - [x] Tests and `docs/phase_1.md`

- [x] **Phase 2: Baselines and Metrics Harness**
  - [x] Abstract `Scheduler` interface (`reset`, `choose_action`, `update`, `explain`)
  - [x] Baselines in `drishti/baselines/` (`SequentialSweep`, `RandomScan`, `PriorityPreMissionSweep`)
  - [x] Formal metric harness in `drishti/metrics/` ($P_d$, $FAR$, count & time $IR$, censored $AIT$, $ITE$, Reward)
  - [x] Multi-seed evaluation runner `experiments/benchmark.py` over 30 identical seeds with 95% CIs
  - [x] Generate baseline results table across all 4 scenarios
  - [x] Tests and `docs/phase_2.md`

- [x] **Phase 3: Non-Stationary Bandit Scheduler**
  - [x] Implement `SlidingWindowUCB` and `DiscountedThompson` in `drishti/schedulers/bandit/`
  - [x] Contextual age-of-information feature weighting
  - [x] Decision attribution in `explain()`
  - [x] Hyperparameter tuning script with train/validation seed split (`configs/bandit_tuned.yaml`)
  - [x] Ablation study (window size, discount factor $\gamma$)
  - [x] Full benchmark demonstrating bandit beats priority sweep on medium & nonstationary scenarios
  - [x] Tests and `docs/phase_3.md`

- [x] **Phase 4: Periodic-Emitter Module and Predictive Models**
  - [x] Periodicity estimation in `drishti/models/periodicity.py` (Circular Phase Coherence / Epoch Folding)
  - [x] `PeriodicAwareScheduler` with lookahead dwell synchronization
  - [x] Dedicated rendezvous strategy against `PeriodicScanReceiverTarget`
  - [x] Learned receiver system model in `drishti/models/receiver_model.py` predicting hit probability and intercept time
  - [x] Estimation error and calibration plots
  - [x] Tests and `docs/phase_4.md`

- [x] **Phase 5: Reinforcement Learning Scheduler**
  - [x] Train PPO agent on `SpectrumScanEnv` with temporal feature representation / frame stacking
  - [x] Curriculum learning protocol (easy -> medium -> hard -> nonstationary)
  - [x] Pure PPO vs Periodic-Feature Augmented PPO comparison
  - [x] Export best policy to ONNX with CPU latency benchmark
  - [x] Tests, training curves, and `docs/phase_5.md`

- [x] **Phase 6: Explainability and the Scheduler Service**
  - [x] Unified decision logging in `drishti/explain/` with JSONL export
  - [x] Standalone `ScanScheduler` service class
  - [x] Formal Model Card in `docs/model_card.md`
  - [x] Tests and `docs/phase_6.md`

- [x] **Phase 7: Tactical Dashboard**
  - [x] Streamlit mission control in `dashboard/app.py`
  - [x] Live waterfall heatmap, cumulative curves, and "Why this band?" log
  - [x] Interactive "scenario shock" mid-mission emitter alteration
  - [x] Tests and `docs/phase_7.md`

- [ ] **Phase 8: Differentiators (Post-Core)**
  - [ ] 8A: Adversarial self-play evasion emitter (`drishti/adversary/`)
  - [ ] 8B: Out-of-distribution (OOD) novelty detector (`drishti/novelty/`)
  - [ ] 8C: Hierarchical coarse-to-fine scanning
  - [ ] 8D: SDR hardware streaming adapter (optional)

- [ ] **Phase 9: Final Packaging & Submission**
  - [ ] Comprehensive `README.md` with Mermaid architecture and headline table
  - [ ] 4-page `docs/technical_report.md`
  - [ ] 5-minute walkthrough script `docs/demo_script.md`
  - [ ] Full reproducibility check on clean virtual environment

---

## 🔒 Approved Assumptions & Parameter Baselines

1. **Spectrum Grid**: Default $B = 16$ frequency bands, time slotted at $\Delta t = 1.0\text{ ms}$, episode length $T = 500\text{ slots}$.
2. **Receiver ROC**: $P_{md} = 0.05 \implies P_d = 0.95$ at nominal SNR; $P_{fa} = 0.02$ per slot per band.
3. **Band Switching Penalty**: Modeled primarily as a small reward penalty ($C_{\text{switch}} = 0.2$) with configurable dead-time slot option.
4. **Censoring Rule**: Never-intercepted bursts are counted as misses for $IR$; $AIT$ evaluates latency over intercepted bursts with full penalty imputation for complete misses.
5. **Statistical Rigor**: All benchmarks evaluate across at least 30 identical seeds reporting mean $\pm 95\%$ Student-$t$ confidence intervals.
