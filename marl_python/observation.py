"""
S-MAS Phase 2 — Task 2.2
Observation Space Construction & Normalisation.

Converts the raw StatePacket from C++ into a normalised float
tensor suitable for neural network input.

Normalisation strategy (from pipeline §2.2.4):
  - Bounded variables  → Min-Max scaling to [0, 1]
  - Unbounded variables → Robust Scaling (clip + scale)
  - Categorical (FDIR) → One-hot encoding
"""
import numpy as np
from typing import Optional
from config import ObsConfig
from env_wrapper import StatePacket


# ═══════════════════════════════════════════════════════════════════
#  Normalisation helpers
# ═══════════════════════════════════════════════════════════════════

def _minmax(val: float, lo: float, hi: float) -> float:
    """Scale val from [lo, hi] to [0, 1], clipped."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _robust(val: float, median: float, iqr: float,
            clip_range: float = 5.0) -> float:
    """Robust scaling: (val - median) / IQR, clipped to ±clip_range."""
    if iqr <= 0:
        return 0.0
    scaled = (val - median) / iqr
    return max(-clip_range, min(clip_range, scaled))


def _log_safe(val: float, floor: float = 1e-20) -> float:
    """log10 of a non-negative value (clamped to floor)."""
    return np.log10(max(val, floor))


def _one_hot(idx: int, n: int) -> list:
    """Return a list of length n with idx-th element = 1."""
    vec = [0.0] * n
    if 0 <= idx < n:
        vec[idx] = 1.0
    return vec


# ═══════════════════════════════════════════════════════════════════
#  State → Observation vector
# ═══════════════════════════════════════════════════════════════════

class ObservationBuilder:
    """
    Converts a raw StatePacket into a normalised numpy vector.

    The observation order is deterministic and documented:
      [orbit(6) | power(4) | env(5) | comm(2) | fdir(4) | degrad(3) | seu(1) | lag(4)]
    Total = obs_dim (from ObsConfig, default 29).
    """

    def __init__(self, cfg: Optional[ObsConfig] = None):
        self.cfg = cfg or ObsConfig()
        self._prev_weather = None   # for lag features (simplified)

    @property
    def obs_dim(self) -> int:
        return self.cfg.obs_dim

    def build(self, s: StatePacket,
              weather_lag: Optional[dict] = None) -> np.ndarray:
        """
        Build a flat observation vector from a StatePacket.

        Parameters
        ----------
        s : StatePacket   — raw state from C++ engine
        weather_lag : dict (optional) — pre-computed lag features

        Returns
        -------
        np.ndarray of shape (obs_dim,), dtype float32
        """
        obs = []

        # ── 1. Orbit features (6) ─────────────────────────────────
        obs.append(_minmax(s.altitude_km, 200.0, 700.0))
        obs.append(_minmax(s.latitude_deg, -90.0, 90.0))
        obs.append(_minmax(s.longitude_deg, -180.0, 180.0))
        # velocity magnitude (m/s) — ~7600 at 600 km
        v_mag = np.sqrt(s.vel_x**2 + s.vel_y**2 + s.vel_z**2)
        obs.append(_minmax(v_mag, 7000.0, 8000.0))
        # velocity direction (normalised components)
        if v_mag > 0:
            obs.append(s.vel_x / v_mag)
            obs.append(s.vel_y / v_mag)
        else:
            obs.extend([0.0, 0.0])

        # ── 2. Power features (4) ─────────────────────────────────
        obs.append(s.battery_soc)  # already [0,1]
        obs.append(_minmax(s.battery_capacity_j, 0.0, 360000.0))
        obs.append(_minmax(s.solar_power_w, 0.0, 100.0))
        obs.append(_minmax(s.power_draw_w, 0.0, 60.0))

        # ── 3. Environment features (5) ───────────────────────────
        obs.append(_robust(_log_safe(s.atm_density),
                           median=-10.0, iqr=1.0))  # log10(ρ) ~ -10
        obs.append(_minmax(_log_safe(max(s.saa_flux_10mev, 0.0) + 1.0),
                           0.0, 5.0))
        obs.append(_minmax(_log_safe(max(s.saa_flux_30mev, 0.0) + 1.0),
                           0.0, 5.0))
        obs.append(float(s.in_eclipse))
        obs.append(float(s.in_saa))

        # ── 4. Communication features (2) ─────────────────────────
        obs.append(1.0 if s.gs_visible > 0 else 0.0)
        obs.append(_minmax(s.time_since_contact_s,
                           0.0, 72.0 * 3600.0))  # normalise to telemetry loss

        # ── 5. FDIR one-hot (4) ───────────────────────────────────
        obs.extend(_one_hot(s.fdir_mode, 4))

        # ── 6. Degradation features (3) ───────────────────────────
        obs.append(s.panel_efficiency)  # [0,1]
        obs.append(_minmax(s.drag_coeff, 1.5, 3.0))
        obs.append(_minmax(float(s.charge_cycles), 0.0, 50000.0))

        # ── 7. SEU (1) ────────────────────────────────────────────
        obs.append(float(s.seu_active))

        # ── 8. Lag features (4) — simplified placeholders ─────────
        # In full implementation, these come from a sliding window
        # of space weather records.  For now, use zeros.
        if weather_lag:
            obs.append(_minmax(weather_lag.get("kp_3h", 0), 0, 9))
            obs.append(_minmax(weather_lag.get("f107_3h", 0), 50, 300))
            obs.append(_minmax(weather_lag.get("kp_6h", 0), 0, 9))
            obs.append(_minmax(weather_lag.get("f107_6h", 0), 50, 300))
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)
