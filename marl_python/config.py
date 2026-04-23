"""
S-MAS Phase 2/3 — Centralised Hyperparameters & Configuration.

All tunable constants live here so that train.py stays clean
and experiments can be tracked by diffing this single file.
"""
from dataclasses import dataclass, field
from pathlib import Path
import os

# ── Resolve project root (two levels up from this file) ────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent


@dataclass
class EnvConfig:
    """Parameters for the C++ physics engine wrapper."""
    data_dir: str = str(PROJECT_ROOT / "preprocessed-data")
    dll_path: str = str(PROJECT_ROOT / "backend_cpp" / "build" / "smas_engine.dll")
    seed: int = 42
    dt: float = 5.0                     # immutable physics step (seconds)
    max_steps_per_episode: int = 17_280  # ~1 day  (86 400 / 5)
    num_envs: int = 4                   # parallel environments for rollout
    density_multiplier: float = 0.01    # calibrated for multi-year realism (PROBA-1: 24yr at 600km)


@dataclass
class ObsConfig:
    """Observation space dimensions and normalisation."""
    # StatePacket fields selected for the observation vector
    # (raw_dim will be computed automatically in observation.py)
    orbit_features: int = 6            # alt, lat, lon, |v|, vx_norm, vy_norm
    power_features: int = 4            # soc, capacity_frac, solar_w, draw_w
    env_features:   int = 5            # rho_log, flux10_log, flux30_log, eclipse, saa
    comm_features:  int = 2            # gs_visible_any, time_since_contact_norm
    fdir_features:  int = 4            # one-hot [NOM, DEG, SAFE, REC]
    degrad_features: int = 3           # panel_eff, cd_norm, cycles_norm
    seu_features:   int = 1            # seu_active
    # look-ahead / look-back lag features (placeholder count)
    lag_features: int = 4              # kp_3h, f107_3h, kp_6h, f107_6h

    @property
    def obs_dim(self) -> int:
        return (self.orbit_features + self.power_features +
                self.env_features + self.comm_features +
                self.fdir_features + self.degrad_features +
                self.seu_features + self.lag_features)


@dataclass
class ActionConfig:
    """Action spaces for the three agents (Phase 2 + Phase 3)."""
    # Navigation Agent  (continuous)
    nav_dim: int = 4                   # [thrust_x, thrust_y, thrust_z, throttle]
    # Resource Agent    (discrete binary)
    bus_dim: int = 1                   # [deep_sleep]
    # Mission Agent     (discrete binary, Phase 3)
    mission_dim: int = 1               # [payload_on]


@dataclass
class RewardConfig:
    """Explicit reward weights (from pipeline doc §3.2.2)."""
    w_alive: float = 1.0               # +1 per step survived
    w_fuel:  float = 5.0               # penalty per unit ΔV used
    w_dod:   float = 2.0               # penalty for Depth of Discharge
    w_fdir:  float = 100.0             # penalty when FDIR intervenes
    w_fatal: float = 1000.0            # massive penalty on terminal failure
    w_alt:   float = 10.0              # penalty for altitude deviation
    target_alt_km: float = 600.0       # nominal target altitude
    alt_deadband_km: float = 10.0      # tolerance band (no penalty within ±10km)


@dataclass
class MissionRewardConfig:
    """Phase 3: Mission-layer reward weights (from pipeline doc §3.3.2)."""
    w_valid_target: float = 50.0       # +50 for valid target imaged
    w_saa_penalty: float = 500.0       # -500 for payload ON inside SAA
    w_idle_power: float = 5.0          # -5 for payload ON when not over target
    # Valid imaging criteria
    target_lat_min: float = -60.0      # min latitude for valid target
    target_lat_max: float = 60.0       # max latitude for valid target
    target_min_solar_w: float = 10.0   # need sunlight for optical imaging


@dataclass
class MAPPOConfig:
    """MAPPO algorithm hyperparameters."""
    # ── Network ──
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "tanh"           # tanh | relu

    # ── PPO ──
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # ── Training ──
    lr: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 4                # PPO update epochs per rollout
    rollout_steps: int = 1176          # ≈ 1 orbit at dt=5 s

    # ── Scaling ──
    num_agents: int = 1                # start with 1 for validation
    shared_policy: bool = True         # shared parameters across agents


@dataclass
class TrainConfig:
    """Top-level training run settings."""
    total_timesteps: int = 500_000
    log_interval: int = 10             # episodes between metric prints
    save_interval: int = 50            # episodes between checkpoint saves
    eval_episodes: int = 5
    checkpoint_dir: str = str(PROJECT_ROOT / "marl_python" / "checkpoints")
    log_dir: str = str(PROJECT_ROOT / "marl_python" / "logs")
    seed: int = 42
    device: str = "cuda"                # "cpu" or "cuda"
