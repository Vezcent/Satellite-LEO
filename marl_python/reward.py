"""
S-MAS Phase 2 — Task 2.3
Explicit Reward Shaping for Survival.

Implements the reward function from pipeline §3.2.2:
    R_t = w1(alive) - w2(ΔV) - w3(DoD) - w4(FDIR_intervention) - w5(fatal)

Design notes:
  • FDIR penalty is applied when the mode transitions FROM NOMINAL
    to DEGRADED or SAFE, teaching the AI that it triggered the safety net.
  • The fatal penalty is only applied on the terminal step.
  • Fuel penalty uses throttle magnitude as a proxy for ΔV.
"""
import numpy as np
from typing import Optional
from config import RewardConfig
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
        dv_proxy = min(thrust_mag, 1.0) * throttle
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

        return reward
