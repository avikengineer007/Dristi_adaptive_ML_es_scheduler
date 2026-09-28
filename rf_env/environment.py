import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from rf_env.emitters import (
    BaseEmitter,
    FixedEmitter,
    PeriodicBurstEmitter,
    FrequencyAgileEmitter,
    PeriodicScanEmitter,
)
from rf_env.rf_world import RFWorld, ChannelObservation


def create_default_emitters(num_bands: int = 16) -> List[BaseEmitter]:
    """
    Creates a standard representative EW emitter scenario:
    - 1 Fixed emitter (surveillance radar / fixed channel)
    - 2 Periodic burst emitters with different periods and burst widths
    - 1 Frequency-agile emitter hopping over bands
    - 1 Periodic-scan rotating radar with mainlobe illumination
    """
    b_fixed = max(0, min(num_bands - 1, 1))
    b_burst1 = max(0, min(num_bands - 1, int(0.3 * num_bands)))
    b_burst2 = max(0, min(num_bands - 1, int(0.7 * num_bands)))
    hop_candidates = [
        int(0.4 * num_bands),
        int(0.5 * num_bands),
        int(0.6 * num_bands),
    ]
    hop_bands = sorted(list(set(max(0, min(num_bands - 1, b)) for b in hop_candidates)))
    b_scan = max(0, min(num_bands - 1, num_bands - 1))

    emitters: List[BaseEmitter] = [
        FixedEmitter(
            emitter_id="FIXED_SURVEILLANCE",
            band=b_fixed,
            threat_weight=2.0,
            active_ratio=1.0,
        ),
        PeriodicBurstEmitter(
            emitter_id="BURST_RADAR_1",
            band=b_burst1,
            period=15,
            burst_duration=3,
            phase=None,  # Randomized per seed on reset
            threat_weight=5.0,
        ),
        PeriodicBurstEmitter(
            emitter_id="BURST_RADAR_2",
            band=b_burst2,
            period=28,
            burst_duration=4,
            phase=None,  # Randomized per seed on reset
            threat_weight=6.0,
        ),
        FrequencyAgileEmitter(
            emitter_id="AGILE_HOPPER",
            hop_bands=hop_bands,
            hop_dwell=2,
            active_ratio=0.85,
            threat_weight=8.0,
        ),
        PeriodicScanEmitter(
            emitter_id="SCAN_SEARCH_RADAR",
            band=b_scan,
            scan_period=40,
            beam_width=3,
            phase=None,  # Randomized per seed on reset
            threat_weight=10.0,
        ),
    ]
    return emitters


class EWScanEnv(gym.Env):
    """
    Gymnasium environment for Electronic Warfare (EW) ES receiver band scheduling.

    Observation space:
        Box(shape=(3 * num_bands,), dtype=np.float32)
        - [0 : N]: Time since last visit to band b (normalized by max_tau)
        - [N : 2N]: Exponential moving average (EMA) hit/miss history for band b [0, 1]
        - [2N : 3N]: Last-seen detection flag on band b (1.0 if detected, 0.0 otherwise)

    Action space:
        Discrete(num_bands): Index of frequency band to monitor next.

    Reward:
        Threat-weighted detections minus dwell cost and false alarm penalty.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        num_bands: int = 16,
        max_steps: int = 1000,
        dwell_time: int = 1,
        dwell_cost: float = 0.5,
        false_alarm_penalty: float = 1.0,
        emitters: Optional[List[BaseEmitter]] = None,
        noise_floor_dbm: float = -95.0,
        p_fa: float = 0.02,
        p_md: float = 0.05,
        max_tau: float = 100.0,
        history_alpha: float = 0.2,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            num_bands: Total frequency channels (N).
            max_steps: Maximum episode length in time slots.
            dwell_time: Number of time slots the receiver monitors the chosen band.
            dwell_cost: Cost subtracted per dwell slot.
            false_alarm_penalty: Cost penalty per false alarm occurrence.
            emitters: Optional custom emitter list; defaults to standard EW scenario.
            noise_floor_dbm: Receiver thermal noise floor.
            p_fa: Probability of false alarm per slot.
            p_md: Probability of missed detection per slot.
            max_tau: Normalization constant for time-since-last-visit.
            history_alpha: EMA smoothing factor for hit/miss history.
            render_mode: Gymnasium render mode ("human" or None).
        """
        super().__init__()
        self.num_bands = num_bands
        self.max_steps = max_steps
        self.dwell_time = dwell_time
        self.dwell_cost = dwell_cost
        self.false_alarm_penalty = false_alarm_penalty
        self.max_tau = max_tau
        self.history_alpha = history_alpha
        self.render_mode = render_mode

        self._emitter_templates = emitters

        # Internal state tracking
        self.time_step = 0
        self.time_since_visit = np.zeros(num_bands, dtype=np.float32)
        self.hit_history = np.zeros(num_bands, dtype=np.float32)
        self.last_seen = np.zeros(num_bands, dtype=np.float32)
        self.visit_counts = np.zeros(num_bands, dtype=np.int64)

        # RF World simulator
        self.rf_world = RFWorld(
            num_bands=num_bands,
            emitters=None,
            noise_floor_dbm=noise_floor_dbm,
            p_fa=p_fa,
            p_md=p_md,
        )

        # Gymnasium spaces
        self.action_space = spaces.Discrete(num_bands)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 * num_bands,),
            dtype=np.float32,
        )

        # Episode logging
        self.episode_observations: List[ChannelObservation] = []
        self.episode_actions: List[int] = []

    def _get_obs(self) -> np.ndarray:
        """Construct normalized observation vector."""
        norm_tau = np.clip(self.time_since_visit / self.max_tau, 0.0, 1.0).astype(np.float32)
        return np.concatenate([norm_tau, self.hit_history, self.last_seen], dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset timers and tracking features
        self.time_step = 0
        self.time_since_visit.fill(0.0)
        self.hit_history.fill(0.0)
        self.last_seen.fill(0.0)
        self.visit_counts.fill(0)
        self.episode_observations.clear()
        self.episode_actions.clear()

        # Re-initialize emitters for RF world
        if self._emitter_templates is not None:
            active_emitters = [
                emitter
                for emitter in self._emitter_templates
            ]
        else:
            active_emitters = create_default_emitters(self.num_bands)

        self.rf_world.emitters = active_emitters
        self.rf_world.reset(self.np_random, max_steps=self.max_steps + self.dwell_time + 10)

        info = {
            "num_bands": self.num_bands,
            "emitter_count": len(self.rf_world.emitters),
            "dwell_time": self.dwell_time,
        }
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute receiver dwell on the chosen frequency band for dwell_time slots.

        Args:
            action: Frequency band index (0 to num_bands - 1).

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        band = int(action)
        self.episode_actions.append(band)
        self.visit_counts[band] += 1

        step_reward = 0.0
        dwell_detected = False
        dwell_false_alarms = 0
        dwell_true_detections = 0
        dwell_threat_reward = 0.0
        detected_emitters: List[str] = []

        # Advance RF world by dwell_time slots
        for _ in range(self.dwell_time):
            current_t = self.time_step
            # Step RF world physical state
            active_on_bands = self.rf_world.step_world(current_t, self.np_random)

            # Observe target band
            obs_result = self.rf_world.observe_band(
                band=band,
                t=current_t,
                active_on_bands=active_on_bands,
                rng=self.np_random,
            )
            self.episode_observations.append(obs_result)

            if obs_result.detected:
                dwell_detected = True
                if obs_result.is_true_detection:
                    dwell_true_detections += 1
                    dwell_threat_reward += obs_result.threat_value
                    detected_emitters.extend(obs_result.detected_emitters)
                elif obs_result.is_false_alarm:
                    dwell_false_alarms += 1

            self.time_step += 1

        # Reward = Threat-weighted detection minus dwell cost and false alarm penalty
        step_reward = (
            dwell_threat_reward
            - (self.dwell_cost * self.dwell_time)
            - (self.false_alarm_penalty * dwell_false_alarms)
        )

        # Update per-band feature state
        # 1. Update time since visit
        self.time_since_visit += self.dwell_time
        self.time_since_visit[band] = 0.0

        # 2. Update hit history (EMA)
        hit_val = 1.0 if dwell_detected else 0.0
        self.hit_history[band] = (
            (1.0 - self.history_alpha) * self.hit_history[band]
            + self.history_alpha * hit_val
        )

        # 3. Update last seen status
        self.last_seen[band] = hit_val

        # Termination & truncation
        terminated = False
        truncated = self.time_step >= self.max_steps

        info = {
            "band": band,
            "time_step": self.time_step,
            "detected": dwell_detected,
            "true_detections": dwell_true_detections,
            "false_alarms": dwell_false_alarms,
            "threat_reward": dwell_threat_reward,
            "detected_emitters": list(set(detected_emitters)),
        }

        return self._get_obs(), step_reward, terminated, truncated, info

    def render(self) -> None:
        """Render current receiver band state to console."""
        print(f"Time: {self.time_step}/{self.max_steps} | Last seen: {self.last_seen}")
