import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sgp4.api import Satrec, WGS84

# Add the current directory to sys.path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EnvConfig
from env_wrapper import SatelliteEnv

def get_altitude_sgp4(satrec, jd, fr):
    """Calculate altitude from SGP4 at a given Julian Date."""
    e, r, v = satrec.sgp4(jd, fr)
    if e != 0:
        return None
    # Calculate magnitude of position vector (in km)
    r_mag = np.linalg.norm(r)
    # Earth equatorial radius in WGS84 is ~6378.137 km
    return r_mag - 6378.137

def parse_proba1_tles(filepath):
    """Generator that yields (Satrec, jd, fr, alt) from a raw TLE file."""
    with open(filepath, 'r') as f:
        while True:
            line1 = f.readline()
            if not line1:
                break
            line1 = line1.strip()
            line2 = f.readline()
            if line2:
                line2 = line2.strip()
                try:
                    satrec = Satrec.twoline2rv(line1, line2)
                    yield line1, line2, satrec
                except Exception as e:
                    pass

def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_tle_path = project_root / "dataset" / "PROBA-1_Orbit_Raw.txt"
    initial_state_path = project_root / "preprocessed-data" / "initial_state.txt"
    
    print("Parsing PROBA-1 Historical TLEs...")
    tles = list(parse_proba1_tles(raw_tle_path))
    if not tles:
        print("Failed to parse TLEs.")
        return

    # Extract ground truth over the first 30 days (or all if less)
    first_satrec = tles[0][2]
    start_jd = first_satrec.jdsatepoch
    start_fr = first_satrec.jdsatepochF
    
    # Write the first TLE to the initial_state.txt
    print(f"Writing initial TLE to {initial_state_path}")
    with open(initial_state_path, "w") as f:
        f.write(tles[0][0] + "\n")
        f.write(tles[0][1] + "\n")

    ground_truth_times = []
    ground_truth_alts = []

    print("Extracting SGP4 Ground Truth Altitudes...")
    for l1, l2, satrec in tles:
        dt_days = (satrec.jdsatepoch - start_jd) + (satrec.jdsatepochF - start_fr)
        if dt_days < 0 or dt_days > 30:
            if dt_days > 30:
                break
            continue
        
        alt = get_altitude_sgp4(satrec, satrec.jdsatepoch, satrec.jdsatepochF)
        if alt is not None:
            ground_truth_times.append(dt_days)
            ground_truth_alts.append(alt)

    print(f"Collected {len(ground_truth_alts)} ground truth points over {ground_truth_times[-1]:.2f} days.")

    # Density multipliers to test
    density_multipliers = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth_times, ground_truth_alts, 'k-', label="PROBA-1 SGP4 (Ground Truth)", linewidth=2)

    # 1 step = 5 seconds
    steps_per_day = int(86400 / 5.0)
    total_days = int(np.ceil(ground_truth_times[-1]))
    total_steps = steps_per_day * total_days
    
    for multiplier in density_multipliers:
        print(f"Running S-MAS Engine with Density Multiplier: {multiplier}x")
        cfg = EnvConfig()
        cfg.density_multiplier = multiplier
        cfg.max_steps_per_episode = total_steps
        cfg.enable_noise = False
        cfg.enable_drift = False # Keep drag coeff constant for baseline test
        
        env = SatelliteEnv(cfg)
        obs = env.reset()
        
        sim_times = []
        sim_alts = []
        
        # Action with 0 thrust and payload off
        zero_action = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 0}
        
        for step in range(total_steps):
            if step % (steps_per_day // 4) == 0:  # Log 4 times a day
                sim_times.append(step * 5.0 / 86400.0)
                sim_alts.append(env._state.altitude_km)
                
            obs, reward, done, info = env.step(zero_action)
            if done:
                break
        
        plt.plot(sim_times, sim_alts, label=f"S-MAS (Multiplier={multiplier})", linestyle='--')
        
        # Calculate approximate MSE (interpolating sim onto ground truth times)
        sim_interp = np.interp(ground_truth_times, sim_times, sim_alts)
        mse = np.mean((sim_interp - ground_truth_alts)**2)
        print(f"  -> MSE: {mse:.4f}")

    plt.xlabel("Days since Epoch")
    plt.ylabel("Altitude (km)")
    plt.title("Orbital Decay Validation: SGP4 vs S-MAS (RK4 + NRLMSISE-00)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    out_plot = project_root / "marl_python" / "decay_comparison.png"
    plt.savefig(out_plot, dpi=300)
    print(f"Saved validation plot to {out_plot}")
    
if __name__ == "__main__":
    main()
