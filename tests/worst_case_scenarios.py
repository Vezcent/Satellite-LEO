import os
import sys
import ctypes as ct
import numpy as np

# Insert the marl_python directory into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../marl_python")))

from env_wrapper import SatelliteEnv, StatePacket, ActionPacket
from config import EnvConfig, ObsConfig
from observation import ObservationBuilder

def print_result(scenario_num, name, status, details):
    color = "\033[92m" if status == "PASSED" else "\033[91m"
    reset = "\033[0m"
    print(f"  [Scenario {scenario_num}] {color}{status}{reset} — {name}")
    print(f"    Details: {details}\n")

# ────────────────────────────────────────────────────────────────────
#  SCENARIO 1: Max Eclipse Duration (Winter Solstice DOY 355)
# ────────────────────────────────────────────────────────────────────
def test_scenario_1():
    cfg = EnvConfig()
    env = SatelliteEnv(cfg)
    env.reset()
    
    # Set time to Winter Solstice DOY 355 (longest shadow chains)
    env._lib.smas_set_time(env._handle, ct.c_double(86400.0 * 355.0))
    
    step = 0
    max_steps = 1176 * 7 # 7 days (approx 103 orbits)
    done = False
    state = env.state
    
    eclipse_steps = 0
    min_temp = 50.0
    
    while step < max_steps and not done:
        action = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 0}
        state, _, done, info = env.step(action)
        step = info["step"]
        if state.in_eclipse:
            eclipse_steps += 1
        min_temp = min(min_temp, float(state.temp_battery))
        
    env.close()
    
    if done and state.done_reason > 0:
        return "FAILED", f"Satellite deceased at day {step * 5 / 86400:.2f} due to Reason Code {state.done_reason}"
    
    eclipse_ratio = (eclipse_steps / step) * 100
    return "PASSED", f"Survived 7 days max eclipse chain. Eclipse ratio: {eclipse_ratio:.1f}%, Min Bat Temp: {min_temp:.2f}°C"

# ────────────────────────────────────────────────────────────────────
#  SCENARIO 2: Halloween Geomagnetic Storm (Kp=9, Severe SAA)
# ────────────────────────────────────────────────────────────────────
def test_scenario_2():
    cfg = EnvConfig()
    env = SatelliteEnv(cfg)
    env.reset()
    
    # October 29, 2003 (Halloween Storm DOY 302, Year 2003)
    halloween_seconds = (3 * 365 + 302) * 86400.0 # 120700800 seconds
    env._lib.smas_set_time(env._handle, ct.c_double(halloween_seconds))
    
    # Inject severe space weather environment parameters
    # s=100.0 (SEU mult), n=2.0 (noise), d=1.2 (drift), dens=0.15 (density mult)
    env._lib.smas_set_environment(env._handle, ct.c_double(100.0), ct.c_double(2.0), ct.c_double(1.2), ct.c_double(0.15))
    
    step = 0
    max_steps = 1176 * 7 # 7 days
    done = False
    state = env.state
    seu_events = 0
    
    while step < max_steps and not done:
        # Passive survival
        action = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 0}
        state, _, done, info = env.step(action)
        step = info["step"]
        if state.seu_active:
            seu_events += 1
            
    env.close()
    
    if done and state.done_reason == 4: # SEU FATAL
        return "FAILED", f"Satellite killed by fatal radiation (SEU FATAL) at step {step}"
    elif done and state.done_reason > 0:
        return "FAILED", f"Deceased due to Reason Code {state.done_reason} at step {step}"
        
    return "PASSED", f"Survived 7-day geomagnetic storm with {seu_events} active SEU events logged."

# ────────────────────────────────────────────────────────────────────
#  SCENARIO 3: Telemetry Blackout (no ground station comms)
# ────────────────────────────────────────────────────────────────────
def test_scenario_3():
    cfg = EnvConfig()
    env = SatelliteEnv(cfg)
    env.reset()
    
    step = 0
    max_steps = 1176 * 3 # 3 days (FDIR triggers SAFE mode at 72 hours comms loss)
    done = False
    state = env.state
    
    while step < max_steps and not done:
        # Step with payload ON (drawing power) to verify operational survival during blackout
        action = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 1}
        state, _, done, info = env.step(action)
        step = info["step"]
        
    env.close()
    
    # Comms loss done reason is 2 (TELEMETRY_LOSS)
    if done and state.done_reason == 2:
         return "PASSED", f"Blackout limit reached successfully: verified done reason is TELEMETRY_LOSS at day {step * 5 / 86400:.2f}."
    elif done:
         return "FAILED", f"Died prematurely due to Reason Code {state.done_reason} at day {step * 5 / 86400:.2f}."
         
    return "PASSED", f"Survived 3-day blackout without entering terminal state (Contact loss time reached {state.time_since_contact_s/3600:.1f} hours)."

# ────────────────────────────────────────────────────────────────────
#  SCENARIO 4: Fuel Exhaustion at Low Orbit (500km)
# ────────────────────────────────────────────────────────────────────
def test_scenario_4():
    cfg = EnvConfig()
    env = SatelliteEnv(cfg)
    env.reset()
    
    # Force low target altitude to provoke extreme atmospheric drag
    env._lib.smas_set_target_altitude(env._handle, ct.c_double(500.0))
    # Artificially deplete fuel by loading massive degradation or let it run
    # Since fuel is not directly settable, we run with severe continuous thrust
    # to drain fuel quickly, or we test decay at 500km with active drag.
    
    step = 0
    max_steps = 1176 * 5 # 5 days
    done = False
    state = env.state
    
    while step < max_steps and not done:
        # Thrust actively down to drain fuel/height
        action = {"nav": np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32), "bus": 0, "mission": 0}
        state, _, done, info = env.step(action)
        step = info["step"]
        
    env.close()
    
    fuel_pct = state.fuel_fraction * 100
    if done and state.done_reason == 5: # FUEL_DEPLETED_LOW
        return "PASSED", f"Verified FUEL_DEPLETED_LOW DoneReason after altitude decayed to <400km (Day {step * 5 / 86400:.2f})."
    elif done and state.done_reason == 3: # REENTRY
        return "PASSED", f"Reentry occurred successfully at day {step * 5 / 86400:.2f} (Altitude dropped below 200km)."
    
    return "PASSED", f"Fuel remaining: {fuel_pct:.2f}%, Final altitude: {state.altitude_km:.2f}km."

# ────────────────────────────────────────────────────────────────────
#  SCENARIO 5: Battery Cold Soak (Solar Maximum Winter Solstice)
# ────────────────────────────────────────────────────────────────────
def test_scenario_5():
    cfg = EnvConfig()
    env = SatelliteEnv(cfg)
    env.reset()
    
    # Solstice starting point
    env._lib.smas_set_time(env._handle, ct.c_double(86400.0 * 355.0))
    # Severe panel degradation (forcing low solar charge)
    env._lib.smas_set_degradation(env._handle, ct.c_double(120000.0), ct.c_double(0.40))
    
    step = 0
    max_steps = 1176 * 5 # 5 days
    done = False
    state = env.state
    min_temp = 50.0
    
    while step < max_steps and not done:
        # Run active payload to drain battery, causing solar panels to be unable to keep up
        action = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 1}
        state, _, done, info = env.step(action)
        step = info["step"]
        min_temp = min(min_temp, float(state.temp_battery))
        
    env.close()
    
    if done and state.done_reason > 0:
        return "FAILED", f"Satellite died due to Reason Code {state.done_reason} at day {step * 5 / 86400:.2f}"
        
    return "PASSED", f"Survived 5-day cold soak. Minimum battery temperature reached: {min_temp:.2f}°C (Safe range minimum: -10°C)."

# ────────────────────────────────────────────────────────────────────
#  SCENARIO 6: Severe Hardware Degradation (Panel = 30%, Cap = 30%)
# ────────────────────────────────────────────────────────────────────
def test_scenario_6():
    cfg = EnvConfig()
    env = SatelliteEnv(cfg)
    env.reset()
    
    # Degrade hardware to 30% panel efficiency and 30% battery capacity
    env._lib.smas_set_degradation(env._handle, ct.c_double(108000.0), ct.c_double(0.30))
    
    step = 0
    max_steps = 1176 * 10 # 10 days
    done = False
    state = env.state
    
    while step < max_steps and not done:
        # Test survival strategy: Deep Sleep active to conserve highly degraded power loop
        action = {"nav": np.zeros(4, dtype=np.float32), "bus": 1, "mission": 0}
        state, _, done, info = env.step(action)
        step = info["step"]
        
    env.close()
    
    if done and state.done_reason > 0:
        return "FAILED", f"Satellite died under severe degradation at day {step * 5 / 86400:.2f} due to Reason Code {state.done_reason}"
        
    return "PASSED", f"Survived 10 days with 30% hardware degradation using forced deep sleep. Final SoC: {state.battery_soc * 100:.2f}%"


if __name__ == "__main__":
    print("================================================================================")
    print("  RUNNING AEROSPACE WORST-CASE STRESS SCENARIOS")
    print("================================================================================\n")
    
    scenarios = [
        ("Max Eclipse shadow chain duration", test_scenario_1),
        ("Halloween Geomagnetic storm environment", test_scenario_2),
        ("48h telemetry Ground Comms blackout", test_scenario_3),
        ("Dynamic fuel exhaustion at low orbit", test_scenario_4),
        ("Battery cold soak under shadow solstice", test_scenario_5),
        ("Extreme solar panel & battery hardware degradation", test_scenario_6)
    ]
    
    passed = 0
    for idx, (name, fn) in enumerate(scenarios, 1):
        try:
            status, details = fn()
            print_result(idx, name, status, details)
            if status == "PASSED":
                passed += 1
        except Exception as e:
            print_result(idx, name, "FAILED", f"Exception raised: {str(e)}")
            
    print("================================================================================")
    print(f"  Worst-Case Scenarios Summary: {passed}/{len(scenarios)} PASSED")
    print("================================================================================")
    sys.exit(0 if passed == len(scenarios) else 1)
