import numpy as np
from env_wrapper import StatePacket

class RuleBasedPolicy:
    """
    Rule-based heuristic policy for S-MAS satellite control.
    
    Rules:
      1. Navigation: Thrust prograde (along velocity vector) at full throttle
         when altitude_km < target_altitude - 5.0 km. Otherwise, coast (throttle = 0.0).
      2. Bus Management: Force deep sleep when SoC < 20% (0.20) to conserve power.
      3. Mission Management: Turn payload ON when ground station is visible and not in SAA.
    """
    def __init__(self, target_altitude_km: float = 600.0):
        self.target_altitude_km = target_altitude_km

    def select_action(self, state: StatePacket) -> dict:
        # 1. Navigation Action: [thrust_x, thrust_y, thrust_z, throttle]
        vx, vy, vz = state.vel_x, state.vel_y, state.vel_z
        v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)
        
        # Default: coasting
        thrust_dir = np.zeros(3, dtype=np.float32)
        throttle = 0.0
        
        # Check altitude gap
        if state.altitude_km < (self.target_altitude_km - 5.0):
            if v_mag > 1e-6:
                thrust_dir = np.array([vx / v_mag, vy / v_mag, vz / v_mag], dtype=np.float32)
                throttle = 1.0
        
        # Action space mapping for python wrapper:
        # In python train.py:
        #   thrust_x = nav_acts[0]
        #   thrust_y = nav_acts[1]
        #   thrust_z = nav_acts[2]
        #   throttle = (nav_acts[3] + 1) / 2 if (nav_acts[3] + 1) / 2 > 0.05 else 0
        # Wait, for the rule-based policy, we can bypass the squashing tanh and raw action mapping
        # when passing directly to env.step(), because env.step() accepts raw values:
        #   self._action.thrust_x = float(nav[0])
        #   self._action.thrust_y = float(nav[1])
        #   self._action.thrust_z = float(nav[2])
        #   self._action.throttle = float(np.clip(nav[3], 0.0, 1.0))
        # So we can output nav action directly as [thrust_dir[0], thrust_dir[1], thrust_dir[2], throttle]
        nav_action = np.array([thrust_dir[0], thrust_dir[1], thrust_dir[2], throttle], dtype=np.float32)
        
        # 2. Bus Management: deep_sleep (0 or 1)
        # Rule: sleep when SoC < 20%
        bus_action = 1 if state.battery_soc < 0.20 else 0
        
        # 3. Mission Management: payload_on (0 or 1)
        # Rule: payload on when gs_visible AND NOT in_saa
        mission_action = 1 if (state.gs_visible > 0 and not state.in_saa) else 0
        
        return {
            "nav": nav_action,
            "bus": bus_action,
            "mission": mission_action
        }
