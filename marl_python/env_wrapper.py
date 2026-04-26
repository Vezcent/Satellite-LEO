"""
S-MAS Phase 2/3 — Tasks 2.1 / 2.2 / 3.3
ctypes bridge to libsmas_engine.dll.

Provides a Gym-like interface:
    env = SatelliteEnv(cfg)
    obs = env.reset()
    obs, reward, done, info = env.step(action_dict)

Phase 3: Supports mission agent action with meta-coordination.
"""
import ctypes as ct
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from config import EnvConfig


# ═══════════════════════════════════════════════════════════════════
#  Mirror the packed C structs from contracts.h
# ═══════════════════════════════════════════════════════════════════

class StatePacket(ct.Structure):
    """Matches smas::StatePacket (version 1, 184 bytes, packed)."""
    _pack_ = 1
    _fields_ = [
        ("version",             ct.c_uint8),
        # Time
        ("sim_time_s",          ct.c_double),
        ("year",                ct.c_int32),
        ("doy",                 ct.c_int32),
        ("hour",                ct.c_int32),
        # Orbital (ECI, m & m/s)
        ("pos_x",               ct.c_double),
        ("pos_y",               ct.c_double),
        ("pos_z",               ct.c_double),
        ("vel_x",               ct.c_double),
        ("vel_y",               ct.c_double),
        ("vel_z",               ct.c_double),
        ("altitude_km",         ct.c_double),
        ("latitude_deg",        ct.c_double),
        ("longitude_deg",       ct.c_double),
        # Power
        ("battery_soc",         ct.c_double),
        ("battery_capacity_j",  ct.c_double),
        ("solar_power_w",       ct.c_double),
        ("power_draw_w",        ct.c_double),
        # Environment
        ("atm_density",         ct.c_double),
        ("drag_force_n",        ct.c_double),
        ("saa_flux_10mev",      ct.c_float),
        ("saa_flux_30mev",      ct.c_float),
        ("in_eclipse",          ct.c_uint8),
        ("in_saa",              ct.c_uint8),
        # Communication
        ("gs_visible",          ct.c_uint8),
        ("time_since_contact_s",ct.c_double),
        # FDIR
        ("fdir_mode",           ct.c_uint8),
        # Degradation
        ("panel_efficiency",    ct.c_double),
        ("drag_coeff",          ct.c_double),
        ("charge_cycles",       ct.c_uint32),
        # Terminal
        ("is_done",             ct.c_uint8),
        ("done_reason",         ct.c_uint8),
        # SEU
        ("seu_active",          ct.c_uint8),
    ]


class ActionPacket(ct.Structure):
    """Matches smas::ActionPacket (version 1, 19 bytes, packed)."""
    _pack_ = 1
    _fields_ = [
        ("version",     ct.c_uint8),
        ("thrust_x",    ct.c_float),
        ("thrust_y",    ct.c_float),
        ("thrust_z",    ct.c_float),
        ("throttle",    ct.c_float),
        ("deep_sleep",  ct.c_uint8),
        ("payload_on",  ct.c_uint8),
    ]


# ═══════════════════════════════════════════════════════════════════
#  Engine wrapper
# ═══════════════════════════════════════════════════════════════════

class SatelliteEnv:
    """
    Single-environment wrapper around the C++ physics engine.

    Usage
    -----
    >>> env = SatelliteEnv(EnvConfig())
    >>> obs = env.reset()
    >>> action = {"nav": np.zeros(4), "bus": 0, "mission": 0}
    >>> obs, reward, done, info = env.step(action)
    """

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        self._load_dll()
        self._create_engine()
        self._step_count = 0
        self._prev_fdir = 0
        self._state = StatePacket()
        # Progressive degradation tracking
        self._current_panel_eff = 1.0
        self._current_capacity_j = 360000.0
        self._progressive_degradation = False
        self._action = ActionPacket()

    # ── DLL management ─────────────────────────────────────────────

    def _load_dll(self):
        dll_path = Path(self.cfg.dll_path)
        if not dll_path.exists():
            raise FileNotFoundError(
                f"Engine DLL not found at {dll_path}.\n"
                f"Build it first:  cd backend_cpp && cmake --build build"
            )
        self._lib = ct.CDLL(str(dll_path))

        # smas_create(data_dir, seed, density_multiplier) -> void*
        self._lib.smas_create.argtypes = [ct.c_char_p, ct.c_ulonglong, ct.c_double]
        self._lib.smas_create.restype  = ct.c_void_p
        # smas_init(engine) -> int
        self._lib.smas_init.argtypes = [ct.c_void_p]
        self._lib.smas_init.restype  = ct.c_int
        # smas_reset(engine) -> void
        self._lib.smas_reset.argtypes = [ct.c_void_p]
        self._lib.smas_reset.restype  = None
        # smas_step(engine, action*, state*) -> void
        self._lib.smas_step.argtypes = [
            ct.c_void_p, ct.POINTER(ActionPacket), ct.POINTER(StatePacket)
        ]
        self._lib.smas_step.restype = None
        # smas_is_done(engine) -> int
        self._lib.smas_is_done.argtypes = [ct.c_void_p]
        self._lib.smas_is_done.restype  = ct.c_int
        # smas_set_time(engine, time_s) -> void
        self._lib.smas_set_time.argtypes = [ct.c_void_p, ct.c_double]
        self._lib.smas_set_time.restype  = None
        # smas_set_degradation(engine, capacity_j, panel_eff) -> void
        self._lib.smas_set_degradation.argtypes = [ct.c_void_p, ct.c_double, ct.c_double]
        self._lib.smas_set_degradation.restype  = None
        # smas_destroy(engine) -> void
        self._lib.smas_destroy.argtypes = [ct.c_void_p]
        self._lib.smas_destroy.restype  = None
        # size checks
        self._lib.smas_state_packet_size.restype  = ct.c_int
        self._lib.smas_action_packet_size.restype = ct.c_int

    def _create_engine(self):
        data_dir = self.cfg.data_dir.encode("utf-8")
        self._handle = self._lib.smas_create(data_dir, self.cfg.seed, ct.c_double(self.cfg.density_multiplier))
        if not self._handle:
            raise RuntimeError("smas_create returned NULL")

        # ABI sanity check
        c_state_sz  = self._lib.smas_state_packet_size()
        c_action_sz = self._lib.smas_action_packet_size()
        py_state_sz = ct.sizeof(StatePacket)
        py_action_sz = ct.sizeof(ActionPacket)
        if c_state_sz != py_state_sz or c_action_sz != py_action_sz:
            raise RuntimeError(
                f"ABI mismatch! C++: State={c_state_sz} Action={c_action_sz}, "
                f"Python: State={py_state_sz} Action={py_action_sz}"
            )

        rc = self._lib.smas_init(self._handle)
        if rc != 0:
            raise RuntimeError("smas_init failed — check data paths")

    # ── Gym-like interface ─────────────────────────────────────────

    def reset(self, randomize: bool = False) -> StatePacket:
        """Reset environment. Returns raw StatePacket."""
        self._lib.smas_reset(self._handle)
        
        if randomize:
            # Sample random start time from the 17-year space weather dataset
            max_start = (17 * 365 - 30) * 86400.0
            start_time = np.random.uniform(0, max_start)
            self._lib.smas_set_time(self._handle, ct.c_double(start_time))
            
            # ── Degradation Training ──
            # Start at a random "mission age" (healthier range since we
            # progressively degrade during the episode)
            capacity_j = np.random.uniform(200000.0, 360000.0)
            panel_eff = np.random.uniform(0.70, 1.0)
            self._lib.smas_set_degradation(self._handle, ct.c_double(capacity_j), ct.c_double(panel_eff))
            
            # Track for progressive degradation during the episode
            self._current_panel_eff = panel_eff
            self._current_capacity_j = capacity_j
            self._progressive_degradation = True
        else:
            self._current_panel_eff = 1.0
            self._current_capacity_j = 360000.0
            self._progressive_degradation = False
            
        self._step_count = 0
        self._prev_fdir = 0
        # Do one no-op step to get initial state
        self._action.version    = 1
        self._action.thrust_x   = 0.0
        self._action.thrust_y   = 0.0
        self._action.thrust_z   = 0.0
        self._action.throttle   = 0.0
        self._action.deep_sleep = 0
        self._action.payload_on = 0
        self._lib.smas_step(
            self._handle,
            ct.byref(self._action),
            ct.byref(self._state),
        )
        return self._state

    def step(self, action: Dict[str, np.ndarray]
             ) -> Tuple[StatePacket, float, bool, dict]:
        """
        Step the environment with agent actions.

        Parameters
        ----------
        action : dict
            "nav"     : np.array shape (4,) → [thrust_x, y, z, throttle]
            "bus"     : int or float        → 0 or 1 (deep_sleep)
            "mission" : int or float        → 0 or 1 (payload_on, Phase 3)

        Returns (state, reward, done, info)
        """
        nav = action.get("nav", np.zeros(4, dtype=np.float32))
        bus = int(action.get("bus", 0))
        mission = int(action.get("mission", 0))

        # ── Meta-Coordination (Task 3.3) ───────────────────────────
        # Software-Defined Resiliency: override payload to OFF
        # when the Resource Agent has triggered deep_sleep.
        requested_mission = int(action.get("mission", 0))
        if bus == 1:
            mission = 0
        
        meta_override = (requested_mission == 1 and mission == 0)

        self._action.version    = 1
        self._action.thrust_x   = float(nav[0])
        self._action.thrust_y   = float(nav[1])
        self._action.thrust_z   = float(nav[2])
        self._action.throttle   = float(np.clip(nav[3], 0.0, 1.0))
        self._action.deep_sleep = bus
        self._action.payload_on = mission

        self._lib.smas_step(
            self._handle,
            ct.byref(self._action),
            ct.byref(self._state),
        )
        self._step_count += 1

        # ── Progressive degradation: worsen hardware every orbit ──
        if (self._progressive_degradation and
                self._step_count % self.cfg.orbit_steps == 0):
            self._current_panel_eff = max(
                self.cfg.min_panel_eff,
                self._current_panel_eff - self.cfg.panel_decay_per_orbit
            )
            self._current_capacity_j = max(
                self.cfg.min_capacity_j,
                self._current_capacity_j - self.cfg.capacity_decay_per_orbit
            )
            self._lib.smas_set_degradation(
                self._handle,
                ct.c_double(self._current_capacity_j),
                ct.c_double(self._current_panel_eff),
            )

        done = bool(self._state.is_done) or \
               self._step_count >= self.cfg.max_steps_per_episode

        info = {
            "step":        self._step_count,
            "sim_time_h":  self._state.sim_time_s / 3600.0,
            "altitude_km": self._state.altitude_km,
            "soc":         self._state.battery_soc,
            "fdir_mode":   self._state.fdir_mode,
            "done_reason": self._state.done_reason,
            "prev_fdir":   self._prev_fdir,
            "payload_on":  mission,         # actual value sent (post-override)
            "meta_override": bus == 1 and int(action.get("mission", 0)) == 1,
        }

        self._prev_fdir = self._state.fdir_mode
        return self._state, 0.0, done, info   # reward computed externally

    @property
    def state(self) -> StatePacket:
        return self._state

    def close(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.smas_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
