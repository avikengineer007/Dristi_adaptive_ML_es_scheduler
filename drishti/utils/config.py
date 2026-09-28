import os
from typing import Dict, Any, Union
import yaml

from drishti.env.environment import SpectrumScanEnv
from drishti.env.receiver import ReceiverModel
from drishti.data.adapter import SyntheticSource


def load_scenario_config(config_input: str) -> Dict[str, Any]:
    """
    Loads scenario configuration from file path or scenario name ('easy', 'medium', etc.).
    """
    if os.path.exists(config_input):
        file_path = config_input
    else:
        # Check configs/ directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
        candidate = os.path.join(base_dir, f"{config_input}.yaml")
        if os.path.exists(candidate):
            file_path = candidate
        else:
            raise FileNotFoundError(f"Configuration not found for '{config_input}' at {candidate}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def create_env_from_config(config_data: Dict[str, Any]) -> SpectrumScanEnv:
    """
    Instantiates a configured SpectrumScanEnv from parsed YAML configuration.
    """
    sc_conf = config_data.get("scenario", {})
    rx_conf = config_data.get("receiver", {})
    em_conf = config_data.get("emitters", [])

    receiver = ReceiverModel(
        noise_floor_dbm=rx_conf.get("noise_floor_dbm", -95.0),
        snr_threshold_db=rx_conf.get("snr_threshold_db", 10.0),
        p_fa=rx_conf.get("p_fa", 0.02),
        noise_std_db=rx_conf.get("noise_std_db", 1.0),
    )

    data_source = SyntheticSource(emitter_specs=em_conf)

    env = SpectrumScanEnv(
        num_bands=sc_conf.get("num_bands", 16),
        max_steps=sc_conf.get("max_steps", 500),
        dwell_time=sc_conf.get("dwell_time", 1),
        dwell_cost=sc_conf.get("dwell_cost", 0.5),
        switch_cost=sc_conf.get("switch_cost", 0.2),
        false_alarm_penalty=sc_conf.get("false_alarm_penalty", 1.0),
        first_intercept_bonus=sc_conf.get("first_intercept_bonus", 5.0),
        nonstationary_step=sc_conf.get("nonstationary_step", None),
        receiver_model=receiver,
        data_source=data_source,
    )
    return env
