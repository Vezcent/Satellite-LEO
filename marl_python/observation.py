"""
S-MAS Phase A — Observation Space Construction & Normalisation.

Converts the raw StatePacket from C++ into a normalised float
tensor suitable for neural network input.

Normalisation strategy (from pipeline §2.2.4):
  - Bounded variables  → Min-Max scaling to [0, 1]
  - Unbounded variables → Robust Scaling (clip + scale)
  - Categorical (FDIR) → One-hot encoding

Performance: Uses pre-allocated numpy array with direct field access
instead of Python list appends and getattr() fallbacks.
"""
import numpy as np
from typing import Optional
from config import ObsConfig
from env_wrapper import StatePacket

# ═══════════════════════════════════════════════════════════════════
#  Pre-computed constants (avoid recomputation every step)
# ═══════════════════════════════════════════════════════════════════

_LOG10_FLOOR = np.float32(np.log10(1e-20))
_ALT_RANGE_INV = np.float32(1.0 / 600.0)       # 1 / (800 - 200)
_LAT_RANGE_INV = np.float32(1.0 / 180.0)        # 1 / (90 - (-90))
_LON_RANGE_INV = np.float32(1.0 / 360.0)        # 1 / (180 - (-180))
_VMAG_RANGE_INV = np.float32(1.0 / 1000.0)      # 1 / (8000 - 7000)
_BATT_CAP_INV = np.float32(1.0 / 360000.0)
_SOLAR_INV = np.float32(1.0 / 100.0)
_DRAW_INV = np.float32(1.0 / 60.0)
_TELEM_LOSS_INV = np.float32(1.0 / (72.0 * 3600.0))
_CD_RANGE_INV = np.float32(1.0 / 1.5)           # 1 / (3.0 - 1.5)
_CYCLES_INV = np.float32(1.0 / 50000.0)
_TEMP_RANGE_INV = np.float32(1.0 / 100.0)       # 1 / (60 - (-40))
_PI_INV = np.float32(1.0 / np.pi)
_WHEEL_RANGE_INV = np.float32(1.0 / 0.4)        # 1 / (0.2 - (-0.2))


def _clamp01(val: float) -> float:
    """Clamp to [0, 1] — branchless-friendly."""
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _log10_safe(val: float) -> float:
    """log10 of a non-negative value (clamped to floor)."""
    if val < 1e-20:
        return float(_LOG10_FLOOR)
    return np.log10(val)


# ═══════════════════════════════════════════════════════════════════
#  State → Observation vector (vectorized)
# ═══════════════════════════════════════════════════════════════════

class ObservationBuilder:
    """
    Converts a raw StatePacket into a normalised numpy vector.

    The observation order is deterministic and documented:
      [orbit(7) | power(4) | env(5) | comm(2) | fdir(4) | degrad(3) | seu(1)
       | fuel(2) | thermal(4) | target_alt(1) | adcs(5) | lag(4)]
    Total = obs_dim (from ObsConfig, default 42).

    Performance: Pre-allocates a numpy buffer and fills it with direct
    index writes — no Python list appends or np.array() conversion.
    """

    def __init__(self, cfg: Optional[ObsConfig] = None):
        self.cfg = cfg or ObsConfig()
        self._buf = np.zeros(self.cfg.obs_dim, dtype=np.float32)
        # Pre-compute target alt normalisation denominator
        self._target_alt_range_inv = np.float32(
            1.0 / max(self.cfg.target_alt_max - self.cfg.target_alt_min, 1e-6))

    @property
    def obs_dim(self) -> int:
        return self.cfg.obs_dim

    def build(self, s: StatePacket,
              weather_lag: Optional[dict] = None,
              target_alt_km: float = 600.0) -> np.ndarray:
        """
        Build a flat observation vector from a StatePacket.

        Parameters
        ----------
        s : StatePacket   — raw state from C++ engine
        weather_lag : dict (optional) — pre-computed lag features
        target_alt_km : float — goal altitude for this episode (Phase A)

        Returns
        -------
        np.ndarray of shape (obs_dim,), dtype float32
        """
        buf = self._buf
        i = 0

        # ── 1. Orbit features (7) ─────────────────────────────────
        buf[i] = _clamp01((s.altitude_km - 200.0) * _ALT_RANGE_INV); i += 1
        buf[i] = _clamp01((s.latitude_deg + 90.0) * _LAT_RANGE_INV); i += 1
        buf[i] = _clamp01((s.longitude_deg + 180.0) * _LON_RANGE_INV); i += 1

        vx, vy, vz = s.vel_x, s.vel_y, s.vel_z
        v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)
        buf[i] = _clamp01((v_mag - 7000.0) * _VMAG_RANGE_INV); i += 1

        if v_mag > 0.0:
            v_inv = 1.0 / v_mag
            buf[i]     = vx * v_inv
            buf[i + 1] = vy * v_inv
            buf[i + 2] = vz * v_inv
        else:
            buf[i] = 0.0; buf[i + 1] = 0.0; buf[i + 2] = 0.0
        i += 3

        # ── 2. Power features (4) ─────────────────────────────────
        buf[i] = s.battery_soc; i += 1
        buf[i] = _clamp01(s.battery_capacity_j * _BATT_CAP_INV); i += 1
        buf[i] = _clamp01(s.solar_power_w * _SOLAR_INV); i += 1
        buf[i] = _clamp01(s.power_draw_w * _DRAW_INV); i += 1

        # ── 3. Environment features (5) ───────────────────────────
        log_rho = _log10_safe(s.atm_density)
        scaled = (log_rho + 10.0)  # median=-10, iqr=1
        buf[i] = max(-5.0, min(5.0, scaled)); i += 1

        flux10 = max(s.saa_flux_10mev, 0.0) + 1.0
        buf[i] = _clamp01(_log10_safe(flux10) * 0.2); i += 1  # / 5.0
        flux30 = max(s.saa_flux_30mev, 0.0) + 1.0
        buf[i] = _clamp01(_log10_safe(flux30) * 0.2); i += 1

        buf[i] = float(s.in_eclipse); i += 1
        buf[i] = float(s.in_saa); i += 1

        # ── 4. Communication features (2) ─────────────────────────
        buf[i] = 1.0 if s.gs_visible > 0 else 0.0; i += 1
        buf[i] = _clamp01(s.time_since_contact_s * _TELEM_LOSS_INV); i += 1

        # ── 5. FDIR one-hot (4) ───────────────────────────────────
        fdir = s.fdir_mode
        buf[i] = 1.0 if fdir == 0 else 0.0; i += 1
        buf[i] = 1.0 if fdir == 1 else 0.0; i += 1
        buf[i] = 1.0 if fdir == 2 else 0.0; i += 1
        buf[i] = 1.0 if fdir == 3 else 0.0; i += 1

        # ── 6. Degradation features (3) ───────────────────────────
        buf[i] = s.panel_efficiency; i += 1
        buf[i] = _clamp01((s.drag_coeff - 1.5) * _CD_RANGE_INV); i += 1
        buf[i] = _clamp01(float(s.charge_cycles) * _CYCLES_INV); i += 1

        # ── 7. SEU (1) ────────────────────────────────────────────
        buf[i] = float(s.seu_active); i += 1

        # ── 8. Fuel features (2) ──────────────────────────────────
        buf[i] = s.fuel_fraction; i += 1
        buf[i] = float(s.fuel_depleted); i += 1

        # ── 9. Thermal features (4) ───────────────────────────────
        buf[i] = _clamp01((s.temp_bus + 40.0) * _TEMP_RANGE_INV); i += 1
        buf[i] = _clamp01((s.temp_battery + 40.0) * _TEMP_RANGE_INV); i += 1
        buf[i] = _clamp01((s.temp_payload + 40.0) * _TEMP_RANGE_INV); i += 1
        buf[i] = float(s.heater_on); i += 1

        # ── 10. Target altitude (1) ───────────────────────────────
        buf[i] = _clamp01(
            (target_alt_km - self.cfg.target_alt_min) * self._target_alt_range_inv
        ); i += 1

        # ── 11. ADCS features (5) ─────────────────────────────────
        buf[i] = _clamp01(s.sun_angle * _PI_INV); i += 1
        buf[i] = _clamp01(s.nadir_error * _PI_INV); i += 1
        buf[i] = _clamp01((s.wheel_momentum_x + 0.2) * _WHEEL_RANGE_INV); i += 1
        buf[i] = _clamp01((s.wheel_momentum_y + 0.2) * _WHEEL_RANGE_INV); i += 1
        buf[i] = _clamp01((s.wheel_momentum_z + 0.2) * _WHEEL_RANGE_INV); i += 1

        # ── 12. Lag features (4) ──────────────────────────────────
        if weather_lag:
            buf[i]     = _clamp01(weather_lag.get("kp_3h", 0) / 9.0)
            buf[i + 1] = _clamp01((weather_lag.get("f107_3h", 0) - 50.0) / 250.0)
            buf[i + 2] = _clamp01(weather_lag.get("kp_6h", 0) / 9.0)
            buf[i + 3] = _clamp01((weather_lag.get("f107_6h", 0) - 50.0) / 250.0)
        else:
            buf[i] = 0.0; buf[i + 1] = 0.0; buf[i + 2] = 0.0; buf[i + 3] = 0.0

        # Return a COPY so the caller's reference isn't invalidated
        # when we overwrite self._buf on the next call.
        return buf.copy()
