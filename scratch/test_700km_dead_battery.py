import os
import sys
import ctypes as ct
import numpy as np

# Insert the marl_python directory into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../marl_python")))

from env_wrapper import SatelliteEnv
from config import EnvConfig

def run_test():
    cfg = EnvConfig()
    env = SatelliteEnv(cfg)
    
    # Setup target altitude to 700km
    env._load_dll()
    env._create_engine()
    
    # Reset env to nominal FIRST
    state = env.reset(randomize=False)
    
    # Set target altitude to 700.0 km AFTER reset so it is not overwritten
    env._lib.smas_set_target_altitude(env._handle, ct.c_double(700.0))
    
    # Set panel efficiency to 0.0 (no solar charging) AFTER reset so it locks in!
    # Keep battery capacity normal at 360000.0J so it drains naturally
    env._lib.smas_set_degradation(env._handle, ct.c_double(360000.0), ct.c_double(0.0))
    
    # Run a no-op step to make sure the state updates with these settings
    action_noop = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 0}
    state, _, _, _ = env.step(action_noop)
    
    print("================================================================================")
    print("  SIMULATION: PUSH TO 700KM & DRAIN BATTERY TO 0%")
    print("================================================================================")
    print(f"Initial State: Alt={state.altitude_km:.2f}km | SoC={state.battery_soc*100:.1f}% | Fuel={state.fuel_fraction*100:.1f}%")
    print("  (Note: solar panel efficiency set to 0.0% to force total battery discharge)\n")
    
    step = 0
    done = False
    
    # Loop until satellite is deceased
    while not done:
        # Action: active thrust to push altitude up, draw maximum power, payload ON
        action = {"nav": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32), "bus": 0, "mission": 1}
        state, _, done, info = env.step(action)
        step += 1
        
        # Print telemetry every 10 steps to show the decay/discharge curve clearly
        if step % 20 == 0 or done:
            done_reasons = ["ONGOING", "BATTERY_DEAD", "TELEMETRY_LOSS", "REENTRY", "SEU_FATAL", "FUEL_DEPLETED_LOW"]
            reason_str = done_reasons[state.done_reason] if state.done_reason < len(done_reasons) else f"CODE_{state.done_reason}"
            fdir_modes = ["NOMINAL", "DEGRADED", "SAFE", "RECOVERY"]
            fdir_str = fdir_modes[state.fdir_mode] if state.fdir_mode < len(fdir_modes) else f"MODE_{state.fdir_mode}"
            
            print(f"Step {step:3d} | Alt={state.altitude_km:.2f}km | SoC={state.battery_soc*100:6.2f}% | Fuel={state.fuel_fraction*100:5.2f}% | FDIR={fdir_str:<8} | Done={state.is_done} ({reason_str})")
            
    print("================================================================================")
    print("  SIMULATION ENDED")
    print("================================================================================")
    print(f"Final State: Alt={state.altitude_km:.2f}km | SoC={state.battery_soc*100:.2f}% | Fuel={state.fuel_fraction*100:.2f}%")
    
    done_reasons = ["ONGOING", "BATTERY_DEAD", "TELEMETRY_LOSS", "REENTRY", "SEU_FATAL", "FUEL_DEPLETED_LOW"]
    reason_str = done_reasons[state.done_reason] if state.done_reason < len(done_reasons) else f"CODE_{state.done_reason}"
    print(f"Done Reason: {reason_str}")
    print("================================================================================")
    env.close()

if __name__ == "__main__":
    run_test()
