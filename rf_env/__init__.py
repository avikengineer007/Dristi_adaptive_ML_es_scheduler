from gymnasium.envs.registration import register
from rf_env.emitters import (
    BaseEmitter,
    FixedEmitter,
    PeriodicBurstEmitter,
    FrequencyAgileEmitter,
    PeriodicScanEmitter,
)
from rf_env.rf_world import RFWorld, BurstEvent, ChannelObservation
from rf_env.environment import EWScanEnv, create_default_emitters

# Register with Gymnasium registry for standard gym.make() access
register(
    id="EWScan-v0",
    entry_point="rf_env.environment:EWScanEnv",
    max_episode_steps=1000,
)

__all__ = [
    "BaseEmitter",
    "FixedEmitter",
    "PeriodicBurstEmitter",
    "FrequencyAgileEmitter",
    "PeriodicScanEmitter",
    "RFWorld",
    "BurstEvent",
    "ChannelObservation",
    "EWScanEnv",
    "create_default_emitters",
]
