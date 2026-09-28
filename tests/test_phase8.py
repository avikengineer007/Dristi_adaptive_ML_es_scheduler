import pytest
import numpy as np

from drishti.adversary.evasion_emitter import AdversarialEvasionEmitter
from drishti.novelty.detector import NoveltyDetector
from drishti.schedulers.hierarchical import HierarchicalScanScheduler
from drishti.env import SpectrumScanEnv


def test_adversarial_evasion_emitter_evades_receiver():
    """Verify that AdversarialEvasionEmitter avoids channels frequently scanned by the receiver."""
    rng = np.random.default_rng(42)
    emitter = AdversarialEvasionEmitter(
        emitter_id="cog_radar_1",
        hop_bands=[2, 8],
        window_size=20,
        evasion_greediness=1.0,  # Deterministic evasion
    )
    emitter.reset(rng)

    # Receiver scans band 2 repeatedly
    for _ in range(15):
        emitter.observe_receiver_action(receiver_band=2)

    # Emitter should choose band 8 because band 2 is heavily monitored
    chosen_band = emitter.get_emission(t=100, rng=rng)
    assert chosen_band == 8, f"Adversary should have evaded to band 8, but emitted on {chosen_band}"


def test_novelty_detector_mahalanobis():
    """Verify NoveltyDetector flags anomalous waveforms and accepts nominal waveforms."""
    detector = NoveltyDetector(threshold=3.0)

    # Synthetic reference library: 50 nominal radar profiles
    # Features: [period, duty_cycle, power_dbm, hop_diversity]
    rng = np.random.default_rng(42)
    nominal_data = np.zeros((60, 4))
    nominal_data[:, 0] = rng.normal(loc=20.0, scale=1.5, size=60)   # period ~20 slots
    nominal_data[:, 1] = rng.normal(loc=0.10, scale=0.01, size=60)  # duty cycle ~10%
    nominal_data[:, 2] = rng.normal(loc=35.0, scale=2.0, size=60)   # power ~35 dBm
    nominal_data[:, 3] = rng.normal(loc=2.0, scale=0.2, size=60)    # hop diversity ~2 bands

    detector.fit_reference_library(nominal_data)
    assert detector.fitted

    # Test nominal sample
    nominal_sample = np.array([20.5, 0.10, 36.0, 2.1])
    d_nom, is_novel_nom = detector.score_novelty(nominal_sample)
    assert not is_novel_nom
    assert d_nom < 3.0
    nom_exp = detector.explain_novelty(nominal_sample)
    assert "Nominal" in nom_exp

    # Test anomalous / OOD sample (extreme PRF/period = 95 slots)
    ood_sample = np.array([95.0, 0.10, 35.0, 2.0])
    d_ood, is_novel_ood = detector.score_novelty(ood_sample)
    assert is_novel_ood
    assert d_ood > 3.0
    ood_exp = detector.explain_novelty(ood_sample)
    assert "NOVEL THREAT DETECTED" in ood_exp
    assert "period" in ood_exp


from drishti.utils.config import load_scenario_config, create_env_from_config


def test_hierarchical_scan_scheduler_simulation_loop():
    """Verify HierarchicalScanScheduler executes coarse-to-fine selection in an RF environment."""
    config = load_scenario_config("easy")
    env = create_env_from_config(config)
    scheduler = HierarchicalScanScheduler(
        num_bands=env.num_bands,
        bands_per_sector=2,
        window_size=30,
        name="Hierarchical Test",
    )

    scheduler.reset(seed=42)

    obs, _ = env.reset(seed=42)
    total_reward = 0.0

    for step in range(30):
        action = scheduler.choose_action(obs)
        assert 0 <= action < env.num_bands

        next_obs, reward, terminated, truncated, info = env.step(action)
        scheduler.update(obs=obs, action=action, reward=reward, info=info)

        total_reward += reward
        obs = next_obs
        if terminated or truncated:
            break

    assert scheduler.last_explanation is not None
    assert "chosen_sector" in scheduler.last_explanation
    assert scheduler.last_action is not None
