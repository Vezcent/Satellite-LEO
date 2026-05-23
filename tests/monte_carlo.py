import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Insert the marl_python directory into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../marl_python")))

from env_wrapper import SatelliteEnv
from config import EnvConfig, ObsConfig, MAPPOConfig
from observation import ObservationBuilder

def run_monte_carlo(num_seeds=100, max_days=30, policy_type="passive", checkpoint_path=None):
    max_steps = int(max_days * 86400 / 5) # 5s per step
    print("================================================================================")
    print(f"  RUNNING MONTE CARLO V&V TEST SUITE")
    print(f"  Seeds: {num_seeds} | Max Duration: {max_days} days ({max_steps} steps)")
    print(f"  Policy: {policy_type.upper()}")
    if policy_type == "active":
        print(f"  Checkpoint: {checkpoint_path}")
    print("================================================================================")

    # Setup PyTorch policy if active
    policy = None
    device = "cpu"
    if policy_type == "active":
        import torch
        from mappo import SharedActorCritic
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            # Try to auto-locate best checkpoint in checkpoints/ or marl_python/checkpoints/
            checkpoints_dirs = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../marl_python/checkpoints"))
            ]
            chk_files = []
            import glob
            for d in checkpoints_dirs:
                chk_files.extend(glob.glob(os.path.join(d, "mappo_phase3_ep*.pt")))
            if chk_files:
                chk_files.sort(key=os.path.getmtime)
                checkpoint_path = chk_files[-1]
                print(f"Auto-located latest checkpoint: {checkpoint_path}")
            else:
                print(f"Error: Active policy specified but checkpoint file not found at: {checkpoint_path}")
                sys.exit(1)
        
        # Load checkpoint
        obs_dim = ObsConfig().obs_dim
        policy = SharedActorCritic(obs_dim=obs_dim, cfg=MAPPOConfig())
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
        policy.load_state_dict(state_dict)
        policy.to(device)
        policy.eval()
        print("Trained MAPPO policy loaded successfully ✓")

    obs_builder = ObservationBuilder()
    results = []

    # Ensure output directory exists
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"monte_carlo_{policy_type}.csv")
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "seed", "survival_steps", "survival_days", "final_alt_km", 
        "final_soc", "final_fuel", "min_temp_bat", "max_temp_bat", 
        "done_reason", "target_alt_km", "completed_orbits"
    ])

    survival_days_list = []

    for seed in range(num_seeds):
        np.random.seed(seed)
        if policy_type == "active":
            torch.manual_seed(seed)

        # Create env config with specific seed
        cfg = EnvConfig()
        cfg.seed = seed
        env = SatelliteEnv(cfg)

        # Reset env with randomize=True (random starting time, degradation, and target altitude)
        state = env.reset(randomize=True)
        target_alt = env._target_alt_km

        min_temp_bat = float(state.temp_battery)
        max_temp_bat = float(state.temp_battery)
        done = False
        step = 0

        # Run step loop
        while step < max_steps and not done:
            if policy_type == "active":
                # Build normalized observation
                obs = obs_builder.build(state)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    act_out = policy.act(obs_t)
                
                nav_act = act_out["nav_action"][0].cpu().numpy()
                bus_act = int(act_out["bus_action"][0].item())
                mission_act = int(act_out["mission_action"][0].item())
                
                action = {"nav": nav_act, "bus": bus_act, "mission": mission_act}
            else:
                # Passive: No-op
                action = {"nav": np.zeros(4, dtype=np.float32), "bus": 0, "mission": 0}

            # Step env
            state, _, done, info = env.step(action)
            step = info["step"]

            # Track battery temperature extremes
            min_temp_bat = min(min_temp_bat, float(state.temp_battery))
            max_temp_bat = max(max_temp_bat, float(state.temp_battery))

        survival_days = step * 5.0 / 86400.0
        survival_days_list.append(survival_days)
        orbits = step / 1176.0 # ~1176 steps per orbit (98 mins)

        # Map done reason string
        done_reasons = ["ONGOING", "BATTERY_DEAD", "TELEMETRY_LOSS", "REENTRY", "SEU_FATAL", "FUEL_DEPLETED_LOW"]
        reason_str = done_reasons[state.done_reason] if state.done_reason < len(done_reasons) else f"CODE_{state.done_reason}"

        # Write to CSV
        csv_writer.writerow([
            seed, step, f"{survival_days:.4f}", f"{state.altitude_km:.2f}",
            f"{state.battery_soc:.4f}", f"{state.fuel_fraction:.4f}",
            f"{min_temp_bat:.2f}", f"{max_temp_bat:.2f}",
            reason_str, f"{target_alt:.1f}", f"{orbits:.2f}"
        ])

        if (seed + 1) % 10 == 0:
            print(f"  Processed {seed + 1}/{num_seeds} seeds... (Latest survival: {survival_days:.2f} days)")

        env.close()

    csv_file.close()
    print(f"\nCSV results exported to: {csv_path}")

    # Calculate statistics
    mean_surv = np.mean(survival_days_list)
    std_surv = np.std(survival_days_list)
    min_surv = np.min(survival_days_list)
    max_surv = np.max(survival_days_list)

    print("\n================================================================================")
    print("  MONTE CARLO STATISTICAL SUMMARY")
    print("================================================================================")
    print(f"  Survival Mean: {mean_surv:.2f} days")
    print(f"  Survival Std:  {std_surv:.2f} days")
    print(f"  Survival Min:  {min_surv:.2f} days")
    print(f"  Survival Max:  {max_surv:.2f} days")
    print("================================================================================")

    # Plot histogram of survival days
    plt.figure(figsize=(10, 6))
    plt.hist(survival_days_list, bins=20, color='royalblue', edgecolor='black', alpha=0.8)
    plt.axvline(mean_surv, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_surv:.2f}d')
    plt.title(f"S-MAS Monte Carlo Survival Distribution ({policy_type.upper()} Policy)")
    plt.xlabel("Survival Time (Days)")
    plt.ylabel("Seed Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(output_dir, f"monte_carlo_{policy_type}_survival.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved survival distribution histogram to: {plot_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo V&V Test Suite")
    parser.add_argument("--seeds", type=int, default=100, help="Number of seeds (default: 100)")
    parser.add_argument("--days", type=float, default=30.0, help="Max duration in days per seed (default: 30)")
    parser.add_argument("--policy", type=str, choices=["passive", "active"], default="passive", 
                        help="Policy mode (passive = no-op, active = trained checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy checkpoint .pt file")
    
    args = parser.parse_args()
    run_monte_carlo(num_seeds=args.seeds, max_days=args.days, policy_type=args.policy, checkpoint_path=args.checkpoint)
