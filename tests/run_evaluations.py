import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import pandas as pd

# Add the marl_python directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../marl_python")))

from env_wrapper import VectorSatelliteEnv, StatePacket
from config import EnvConfig, ObsConfig, MAPPOConfig, TrainConfig
from observation import ObservationBuilder
from rule_based_policy import RuleBasedPolicy
from mappo import SharedActorCritic

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest .pt checkpoint in the given directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    import glob
    files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def run_evaluation(policy_name, checkpoint_path=None, num_seeds=20, max_days=21):
    max_steps = int(max_days * 86400 / 5) # 5s per step
    print(f"\nEvaluating: {policy_name.upper()} policy for {num_seeds} seeds, max {max_days} days...")
    
    # 1. Setup Environment
    cfg = EnvConfig()
    cfg.num_envs = num_seeds
    cfg.seed = 0 # seeds: 0, 1, 2, ...
    vec_env = VectorSatelliteEnv(cfg)
    
    # 2. Setup Policy
    policy = None
    device = "cpu"
    
    if policy_name in ["mappo", "ippo"]:
        obs_dim = ObsConfig().obs_dim
        shared_policy = (policy_name == "mappo")
        mappo_cfg = MAPPOConfig(shared_policy=shared_policy)
        policy = SharedActorCritic(obs_dim=obs_dim, cfg=mappo_cfg)
        
        if not checkpoint_path:
            # Auto-locate checkpoint
            if policy_name == "mappo":
                # Check standard directories
                dirs = ["checkpoints/mappo_seed42", "marl_python/checkpoints", "checkpoints"]
                for d in dirs:
                    checkpoint_path = get_latest_checkpoint(d)
                    if checkpoint_path:
                        break
            else:
                dirs = ["checkpoints/ippo_seed42", "marl_python/checkpoints"]
                for d in dirs:
                    checkpoint_path = get_latest_checkpoint(d)
                    if checkpoint_path:
                        break
                
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"  Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
            
            # Use compatibility loader shim
            from train import load_compatible_state_dict
            load_compatible_state_dict(policy, state_dict)
            policy.to(device)
            policy.eval()
            print("  Trained model loaded successfully.")
        else:
            print(f"  Warning: No trained model checkpoint found for {policy_name.upper()}! Running with random weights.")
            policy.to(device)
            policy.eval()

    obs_builder = ObservationBuilder()
    
    # Reset environments
    # We want reproducible start states per seed, so we do randomize=True which sets time/degrad based on seed
    raw_states = vec_env.reset(randomize=True)
    
    rule_policies = [RuleBasedPolicy(target_altitude_km=vec_env.envs[i]._target_alt_km) for i in range(num_seeds)]
    
    # Prepare metrics tracking
    survival_steps = np.zeros(num_seeds, dtype=np.int32)
    done_flags = np.zeros(num_seeds, dtype=bool)
    final_soc = np.zeros(num_seeds, dtype=np.float32)
    final_fuel = np.zeros(num_seeds, dtype=np.float32)
    final_alt = np.zeros(num_seeds, dtype=np.float32)
    done_reasons = np.zeros(num_seeds, dtype=np.int32)
    payload_on_steps = np.zeros(num_seeds, dtype=np.int32)
    valid_targets = np.zeros(num_seeds, dtype=np.int32)
    saa_violations = np.zeros(num_seeds, dtype=np.int32)
    min_bat_temp = np.ones(num_seeds, dtype=np.float32) * 50.0
    max_bat_temp = np.ones(num_seeds, dtype=np.float32) * -50.0

    # Initialize min/max bat temp with start state
    for i in range(num_seeds):
        min_bat_temp[i] = raw_states[i].temp_battery
        max_bat_temp[i] = raw_states[i].temp_battery
        
    obs_list = [obs_builder.build(raw_states[i], target_alt_km=vec_env.envs[i]._target_alt_km) for i in range(num_seeds)]
    batch_obs = np.empty((num_seeds, ObsConfig().obs_dim), dtype=np.float32)

    step = 0
    start_time = time.time()
    
    # Run simulation step loop
    while step < max_steps and not np.all(done_flags):
        action_dicts = []
        
        # Make actions for active policy in batch
        if policy_name in ["mappo", "ippo"]:
            for i in range(num_seeds):
                batch_obs[i] = obs_list[i]
            obs_tensor = torch.from_numpy(batch_obs).to(device)
            with torch.no_grad():
                out = policy.act(obs_tensor)
            nav_acts = out["nav_action"].cpu().numpy()
            bus_acts = out["bus_action"].cpu().numpy()
            mis_acts = out["mission_action"].cpu().numpy()

        for i in range(num_seeds):
            if done_flags[i]:
                # Send no-ops for finished envs
                action_dicts.append({
                    "nav": np.zeros(4, dtype=np.float32),
                    "bus": 0,
                    "mission": 0
                })
                continue
                
            if policy_name == "passive":
                action_dicts.append({
                    "nav": np.zeros(4, dtype=np.float32),
                    "bus": 0,
                    "mission": 0
                })
            elif policy_name == "rule_based":
                action_dicts.append(rule_policies[i].select_action(raw_states[i]))
            elif policy_name == "random":
                action_dicts.append({
                    "nav": np.random.uniform(-1, 1, 4).astype(np.float32),
                    "bus": int(np.random.choice([0, 1])),
                    "mission": int(np.random.choice([0, 1]))
                })
            elif policy_name in ["mappo", "ippo"]:
                action_dicts.append({
                    "nav": np.array([
                        nav_acts[i,0], nav_acts[i,1], nav_acts[i,2],
                        ((nav_acts[i,3] + 1.0) / 2.0) if ((nav_acts[i,3] + 1.0) / 2.0) > 0.05 else 0.0
                    ], dtype=np.float32),
                    "bus": int(bus_acts[i]),
                    "mission": int(mis_acts[i])
                })

        # Step environments
        results = vec_env.step(action_dicts)
        
        # Update states & stats
        for i in range(num_seeds):
            if done_flags[i]:
                continue
                
            state, _, done, info = results[i]
            raw_states[i] = state
            
            # Update observation
            obs_list[i] = obs_builder.build(state, target_alt_km=vec_env.envs[i]._target_alt_km)
            
            # Update metrics
            survival_steps[i] += 1
            min_bat_temp[i] = min(min_bat_temp[i], float(state.temp_battery))
            max_bat_temp[i] = max(max_bat_temp[i], float(state.temp_battery))
            
            # Payload tracking
            if info["payload_on"]:
                payload_on_steps[i] += 1
                # Check valid target (between -60 and 60 deg lat, not in saa, not in eclipse, solar arrays > 10W)
                is_valid_target = (abs(state.latitude_deg) <= 60.0 and 
                                   not state.in_saa and 
                                   not state.in_eclipse and 
                                   state.solar_power_w >= 10.0)
                if is_valid_target:
                    valid_targets[i] += 1
                if state.in_saa:
                    saa_violations[i] += 1

            if done or step == max_steps - 1:
                done_flags[i] = True
                final_soc[i] = float(state.battery_soc * 100)
                final_fuel[i] = float(state.fuel_fraction * 100)
                final_alt[i] = float(state.altitude_km)
                done_reasons[i] = int(state.done_reason)

        step += 1

    vec_env.close()
    duration = time.time() - start_time
    
    # Calculate stats
    survival_days = (survival_steps * 5.0) / 86400.0
    mean_days = np.mean(survival_days)
    std_days = np.std(survival_days)
    
    # Orbits survived
    mean_orbits = np.mean(survival_steps / 1176.0)
    
    # Calculate survival threshold rates
    survived_full = np.sum(survival_days >= (max_days - 0.01)) / num_seeds * 100
    
    mean_soc = np.mean(final_soc)
    mean_fuel = np.mean(final_fuel)
    mean_targets = np.mean(valid_targets)
    mean_violations = np.mean(saa_violations)
    
    # Projected lifetime: extrapolate from fuel burn rate
    # If fuel > 0 at end, project how long it would last at the observed burn rate
    projected_lifetimes = []
    for i in range(num_seeds):
        fuel_remaining = final_fuel[i] / 100.0  # fraction
        days_survived = survival_days[i]
        if fuel_remaining > 0.01 and days_survived > 0:
            fuel_burned = 1.0 - fuel_remaining
            if fuel_burned > 0.001:
                projected_days = days_survived / fuel_burned
            else:
                projected_days = days_survived * 100  # essentially infinite fuel
        else:
            projected_days = days_survived  # fuel ran out, this IS the lifetime
        projected_lifetimes.append(projected_days / 365.0)  # convert to years
    
    mean_projected_years = float(np.mean(projected_lifetimes))
    std_projected_years = float(np.std(projected_lifetimes))
    
    # Fuel efficiency: targets per % fuel spent
    fuel_spent_pcts = 100.0 - final_fuel
    fuel_efficiencies = []
    for i in range(num_seeds):
        if fuel_spent_pcts[i] > 0.1:
            fuel_efficiencies.append(valid_targets[i] / fuel_spent_pcts[i])
        else:
            fuel_efficiencies.append(float(valid_targets[i]))
    mean_fuel_efficiency = float(np.mean(fuel_efficiencies))
    
    print(f"Done in {duration:.1f}s. Survival Mean: {mean_days:.2f} +/- {std_days:.2f} days. "
          f"Surv {max_days}d: {survived_full:.1f}%. Avg Fuel: {mean_fuel:.1f}%. "
          f"Targets: {mean_targets:.1f}. Projected: {mean_projected_years:.1f} yr")

    return {
        "policy": policy_name,
        "mean_days": float(mean_days),
        "std_days": float(std_days),
        "mean_orbits": float(mean_orbits),
        "survival_full_pct": float(survived_full),
        "mean_soc": float(mean_soc),
        "mean_fuel": float(mean_fuel),
        "mean_targets": float(mean_targets),
        "mean_violations": float(mean_violations),
        "mean_projected_years": mean_projected_years,
        "std_projected_years": std_projected_years,
        "mean_fuel_efficiency": mean_fuel_efficiency,
        "raw_survival_days": survival_days.tolist(),
        "done_reasons": done_reasons.tolist()
    }

def main():
    parser = argparse.ArgumentParser(description="S-MAS Comparative Evaluations")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default: 20)")
    parser.add_argument("--days", type=float, default=21.0, help="Max evaluation days (default: 21.0)")
    parser.add_argument("--mappo_ckpt", type=str, default=None, help="Custom MAPPO checkpoint path")
    parser.add_argument("--ippo_ckpt", type=str, default=None, help="Custom IPPO checkpoint path")
    args = parser.parse_args()
    
    print("================================================================================")
    print(f"  RUNNING COMPARATIVE EVALUATION SUITE")
    print(f"  Seeds: {args.seeds} | Max Duration: {args.days} days")
    print("================================================================================")

    # 1. Evaluate policies
    results = []
    
    # Passive
    res_passive = run_evaluation("passive", num_seeds=args.seeds, max_days=args.days)
    results.append(res_passive)
    
    # Random
    res_random = run_evaluation("random", num_seeds=args.seeds, max_days=args.days)
    results.append(res_random)
    
    # Heuristic Rule-Based
    res_rule = run_evaluation("rule_based", num_seeds=args.seeds, max_days=args.days)
    results.append(res_rule)
    
    # IPPO
    res_ippo = run_evaluation("ippo", checkpoint_path=args.ippo_ckpt, num_seeds=args.seeds, max_days=args.days)
    results.append(res_ippo)
    
    # MAPPO
    res_mappo = run_evaluation("mappo", checkpoint_path=args.mappo_ckpt, num_seeds=args.seeds, max_days=args.days)
    results.append(res_mappo)

    # 2. Save results
    output_dir = "tests/results"
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nJSON results saved to: {json_path}")
    
    # 3. Create Markdown Table
    df = pd.DataFrame(results)
    markdown_path = os.path.join(output_dir, "evaluation_summary.md")
    
    md_content = "# S-MAS Policy Evaluation Summary\n\n"
    md_content += f"Evaluated across **{args.seeds} seeds** for a maximum duration of **{args.days} days** per seed (with progressive degradation).\n\n"
    md_content += "| Policy | Mean Survival (Days) | Survived Full (%) | Final Fuel (%) | Target Images | Fuel Efficiency | Projected Lifetime (Yr) |\n"
    md_content += "|---|---|---|---|---|---|---|\n"
    
    for r in results:
        policy_label = {
            "passive": "No-op (Passive)",
            "random": "Random Policy",
            "rule_based": "Rule-Based Heuristic",
            "ippo": "IPPO (Independent PPO)",
            "mappo": "MAPPO (Ours)"
        }.get(r["policy"], r["policy"].upper())
        
        md_content += (f"| {policy_label} | {r['mean_days']:.2f} +/- {r['std_days']:.2f} | "
                       f"{r['survival_full_pct']:.1f}% | "
                       f"{r['mean_fuel']:.1f}% | "
                       f"{r['mean_targets']:.1f} | {r['mean_fuel_efficiency']:.1f} | "
                       f"{r['mean_projected_years']:.1f} +/- {r['std_projected_years']:.1f} |\n")
                       
    with open(markdown_path, "w") as f:
        f.write(md_content)
        
    print(f"Markdown summary report saved to: {markdown_path}")
    print("\n" + "=" * 80)
    print("  EVALUATION COMPLETE")
    print("=" * 80)
    print(md_content)

if __name__ == "__main__":
    main()
