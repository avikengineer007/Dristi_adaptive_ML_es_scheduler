import numpy as np
import pytest
from rf_env.emitters import (
    FixedEmitter,
    PeriodicBurstEmitter,
    FrequencyAgileEmitter,
    PeriodicScanEmitter,
)


def test_fixed_emitter():
    rng = np.random.default_rng(42)
    emitter = FixedEmitter("test_fixed", band=3, active_ratio=1.0)
    emitter.reset(rng)

    for t in range(50):
        band = emitter.get_emission(t, rng)
        assert band == 3


def test_periodic_burst_emitter():
    rng = np.random.default_rng(42)
    period = 10
    burst_dur = 3
    phase = 2
    emitter = PeriodicBurstEmitter(
        "test_burst", band=4, period=period, burst_duration=burst_dur, phase=phase
    )
    emitter.reset(rng)

    for t in range(50):
        expected_active = ((t + phase) % period) < burst_dur
        band = emitter.get_emission(t, rng)
        if expected_active:
            assert band == 4
        else:
            assert band is None


def test_frequency_agile_emitter():
    rng = np.random.default_rng(123)
    hop_bands = [1, 5, 8, 12]
    dwell = 2
    emitter = FrequencyAgileEmitter(
        "test_agile", hop_bands=hop_bands, hop_dwell=dwell, active_ratio=1.0
    )
    emitter.reset(rng)

    seen_bands = set()
    for t in range(20):
        band = emitter.get_emission(t, rng)
        assert band in hop_bands
        seen_bands.add(band)

    # Check that it hopped across multiple bands in the hop set
    assert len(seen_bands) > 1


def test_periodic_scan_emitter():
    rng = np.random.default_rng(999)
    scan_period = 25
    beam_width = 4
    phase = 0
    emitter = PeriodicScanEmitter(
        "test_scan", band=15, scan_period=scan_period, beam_width=beam_width, phase=phase
    )
    emitter.reset(rng)

    active_count = 0
    for t in range(scan_period):
        band = emitter.get_emission(t, rng)
        if band is not None:
            assert band == 15
            active_count += 1

    assert active_count == beam_width
