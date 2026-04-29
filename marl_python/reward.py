"""
S-MAS Phase 2/3 — Tasks 2.3 / 3.2
Explicit Reward Shaping for Survival + Mission.

Phase 2 (Survival):
    R_t = w1(alive) - w2(ΔV) - w3(DoD) - w4(FDIR_intervention) - w5(fatal)

Phase 3 (Mission):
    R_mission = R_survival
                + (payload_on × valid_target) × w_valid_target
                - (payload_on × in_saa) × w_saa_penalty
                - (payload_on × NOT_over_target) × w_idle_power

Design notes:
  • FDIR penalty is applied when the mode transitions FROM NOMINAL
    to DEGRADED or SAFE, teaching the AI that it triggered the safety net.
  • The fatal penalty is only applied on the terminal step.
  • Fuel penalty uses throttle magnitude as a proxy for ΔV.
  • Valid target: not in SAA, not in eclipse, within lat band, sufficient solar.
"""
import numpy as np
from typing import Optional
from config import RewardConfig, MissionRewardConfig
from env_wrapper import StatePacket


class SurvivalReward:
    """
    Computes the per-step survival reward.

    Usage
    -----
    >>> rew = SurvivalReward()
    >>> r  = rew.compute(state, action_dict, done, info)
    """

    def __init__(self, cfg: Optional[RewardConfig] = None):
        self.cfg = cfg or RewardConfig()

    def compute(self,
                state: StatePacket,
                action: dict,
                done: bool,
                info: dict) -> float:
        """
        Compute the reward for a single step.

        Parameters
        ----------
        state  : StatePacket from the C++ engine (post-step)
        action : dict with "nav" (np.array shape 4) and "bus" (int)
        done   : bool — whether the episode terminated
        info   : dict — from env.step(), includes prev_fdir

        Returns
        -------
        float — the shaped reward
        """
        reward = 0.0

        # ── 1. Alive bonus ─────────────────────────────────────────
        reward += self.cfg.w_alive * 1.0

        # ── 2. Fuel penalty (ΔV proxy = throttle magnitude) ────────
        nav = action.get("nav", np.zeros(4, dtype=np.float32))
        throttle = float(np.clip(nav[3], 0.0, 1.0))
        thrust_mag = np.sqrt(nav[0]**2 + nav[1]**2 + nav[2]**2)
        # Normalised ΔV proxy: direction magnitude × throttle
        dv_proxy = thrust_mag * throttle
        reward -= self.cfg.w_fuel * dv_proxy

        # ── 3. Depth of Discharge penalty ──────────────────────────
        #    DoD = 1 - SoC.  Penalise low SoC to teach battery care.
        dod = 1.0 - state.battery_soc
        reward -= self.cfg.w_dod * dod

        # ── 4. FDIR intervention penalty ───────────────────────────
        #    Penalise transitions FROM NOMINAL to DEGRADED or SAFE.
        prev_fdir = info.get("prev_fdir", 0)
        curr_fdir = state.fdir_mode
        if prev_fdir == 0 and curr_fdir in (1, 2):
            # AI caused a downgrade → sharp penalty
            reward -= self.cfg.w_fdir
        elif prev_fdir == 1 and curr_fdir == 2:
            # Further degradation DEGRADED → SAFE
            reward -= self.cfg.w_fdir * 0.5

        # ── 5. Fatal penalty (terminal failure) ────────────────────
        if done and state.done_reason > 0:
            reward -= self.cfg.w_fatal

        # ── 6. Altitude Maintenance Penalty ────────────────────────
        #    Forces the AI to stay near the target orbit (e.g. 600km).
        alt_km = state.altitude_km
        alt_err = abs(alt_km - self.cfg.target_alt_km)
        if alt_err > self.cfg.alt_deadband_km:
            # Penalise linearly for every km outside the deadband
            penalty = self.cfg.w_alt * (alt_err - self.cfg.alt_deadband_km)
            reward -= penalty

        return reward


class MissionReward:
    """
    Phase 3: Mission-layer reward that composes with SurvivalReward.

    Adds three mission-specific terms:
      +w_valid_target  : payload ON over a valid imaging target
      -w_saa_penalty   : payload ON inside the SAA (radiation risk)
      -w_idle_power    : payload ON when NOT over a valid target (wasted power)

    A "valid target" requires ALL of:
      1. Not inside the SAA boundary
      2. Not in eclipse (need sunlight for optical imaging)
      3. Latitude within the imaging band (configurable, default ±60°)
      4. Sufficient solar power (panel illumination)

    Usage
    -----
    >>> rew = MissionReward()
    >>> r, info = rew.compute(state, action_dict, done, info)
    """

    def __init__(self,
                 survival_cfg: Optional[RewardConfig] = None,
                 mission_cfg: Optional[MissionRewardConfig] = None):
        self.survival = SurvivalReward(survival_cfg)
        self.cfg = mission_cfg or MissionRewardConfig()
        self._has_imaged_current_target = False  # Fix #1: prevent reward farming

    def _is_valid_target(self, state: StatePacket) -> bool:
        """Check if current position is a valid imaging target."""
        # Must NOT be in SAA
        if state.in_saa:
            return False
        # Must NOT be in eclipse (need sunlight for CHRIS optical instrument)
        if state.in_eclipse:
            return False
        # Must be within valid latitude band
        lat = state.latitude_deg
        if lat < self.cfg.target_lat_min or lat > self.cfg.target_lat_max:
            return False
        # Must have sufficient solar power (panel illumination)
        if state.solar_power_w < self.cfg.target_min_solar_w:
            return False
        return True

    def compute(self,
                state: StatePacket,
                action: dict,
                done: bool,
                info: dict) -> tuple:
        """
        Compute the combined survival + mission reward.

        Parameters
        ----------
        state  : StatePacket from the C++ engine (post-step)
        action : dict with "nav", "bus", "mission" keys
        done   : bool — whether the episode terminated
        info   : dict — from env.step(), includes prev_fdir

        Returns
        -------
        (float, dict) — total reward and mission metrics
        """
        # ── Survival baseline ──────────────────────────────────────
        r_survival = self.survival.compute(state, action, done, info)

        # ── Mission terms ──────────────────────────────────────────
        # Use actual payload status (post-override) from info dict
        actual_payload = float(info.get("payload_on", action.get("mission", 0)))
        requested_payload = float(action.get("mission", 0))
        
        r_mission = 0.0
        valid_target = self._is_valid_target(state)

        if actual_payload > 0.5:
            if state.in_saa:
                # CRITICAL: Payload ON inside SAA → massive penalty
                r_mission -= self.cfg.w_saa_penalty
            elif valid_target:
                # Fix #2: Dynamic battery safety floor based on degraded capacity
                # A 35-min eclipse requires ~60,000J. We enforce an 80,000J buffer to be safe.
                safe_soc_floor = 80000.0 / max(state.battery_capacity_j, 1000.0)
                
                if state.battery_soc < safe_soc_floor:
                    # Low battery but still trying to image → punish greed
                    # Penalty MUST be larger than w_valid_target to prevent reward hacking
                    r_mission -= self.cfg.w_valid_target * 1.5
                elif not self._has_imaged_current_target:
                    # Fix #1: First step over this target → grant full bonus
                    r_mission += self.cfg.w_valid_target
                    self._has_imaged_current_target = True
                else:
                    # Fix #1: Already imaged this pass → tiny maintenance reward
                    r_mission += 2.0
            else:
                # Payload ON but not over target → wasted power
                r_mission -= self.cfg.w_idle_power
        else:
            # Payload is OFF. Check for "Sloth" (Sleeping when should be imaging)
            # Only apply if the Agent INTENTIONALLY slept, not if it was overridden
            deep_sleep = float(action.get("bus", 0)) > 0.5
            if deep_sleep and valid_target and state.battery_soc > 0.9 and not info.get("meta_override", False):
                # Sat is over target with >90% battery but choosing to sleep → Sloth Penalty
                r_mission -= self.cfg.w_sloth_penalty

        # Fix #1: Reset the flag when flying out of the target area
        if not valid_target:
            self._has_imaged_current_target = False

        # Coordination Penalty: If agents disagreed and the hardcode had to step in
        if info.get("meta_override", False):
            r_mission -= 50.0  # Small penalty for lack of agent coordination


        # ── Compose ────────────────────────────────────────────────
        total = r_survival + r_mission

        mission_info = {
            "r_survival": r_survival,
            "r_mission": r_mission,
            "payload_on": actual_payload > 0.5,
            "valid_target": valid_target,
            "saa_violation": actual_payload > 0.5 and bool(state.in_saa),
        }

        return total, mission_info
