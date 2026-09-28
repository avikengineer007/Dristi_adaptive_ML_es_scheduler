import json
import os
import tempfile
import pytest
import numpy as np

from drishti.explain.logger import ExplanationLogger, DecisionRecord
from drishti.service import ScanScheduler
from drishti.env.environment import SpectrumScanEnv


def test_explanation_logger_buffering_and_export():
    """Verify ExplanationLogger buffers records and exports valid JSONL."""
    logger = ExplanationLogger(max_records=10)

    # Log 15 records (should keep last 10)
    for t in range(15):
        logger.log(
            slot=t,
            chosen_band=t % 4,
            scheduler_name="TestScheduler",
            reason=f"Slot {t} decision",
            scores={0: 1.0, 1: 2.0},
        )

    assert len(logger.records) == 10
    recent = logger.get_recent(5)
    assert len(recent) == 5
    assert recent[0]["slot"] == 14

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        exported = logger.export_jsonl(tmp_path)
        assert os.path.exists(exported)
        with open(exported, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 10
        assert lines[-1]["slot"] == 14
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_contrastive_explanation_generation():
    """Verify natural-language contrastive explanation logic."""
    logger = ExplanationLogger()
    rec = logger.log(
        slot=10,
        chosen_band=3,
        scheduler_name="SW-UCB",
        reason="Exploiting active channel",
        scores={1: 2.5, 3: 5.8},
        components={
            "aoi_bonuses": {1: 0.2, 3: 1.5},
            "empirical_means": {1: 0.1, 3: 3.2},
        },
    )

    contrast = logger.explain_contrastive(band_chosen=3, band_alternative=1, record=rec)
    assert "Band 3 was selected over Band 1" in contrast
    assert "+3.300" in contrast
    assert "Age-of-Information" in contrast
    assert "Empirical reward" in contrast


def test_scan_scheduler_service_modes_and_switching():
    """Verify ScanScheduler service initializes, switches modes, and executes episode."""
    service = ScanScheduler(num_bands=8, mode="auto")
    assert service.mode == "auto"

    # Test mode switching
    service.set_mode("bandit_ucb")
    assert service.mode == "bandit_ucb"

    service.set_mode("periodic")
    assert service.mode == "periodic"

    with pytest.raises(ValueError):
        service.set_mode("invalid_mode")

    # Step through simple episode
    env = SpectrumScanEnv(num_bands=8, max_steps=20)
    obs, info = env.reset(seed=42)
    service.reset(seed=42)

    done = False
    step_count = 0
    while not done:
        action = service.step(obs, info)
        assert 0 <= action < 8
        obs, reward, term, trunc, step_info = env.step(action)
        service.update(obs, action, reward, step_info)
        done = term or trunc
        step_count += 1

    assert step_count == 20
    assert len(service.logger.records) == 20
    exp = service.explain()
    assert "chosen_band" in exp
