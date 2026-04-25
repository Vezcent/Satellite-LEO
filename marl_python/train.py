import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainConfig, ObsConfig, RewardConfig, MAPPOConfig, EnvConfig
from mappo import SharedActorCritic, RolloutBuffer, ppo_update
from env_wrapper import SMASWrapper
from observation import ObservationBuilder
from reward import SurvivalReward, MissionReward

def train(train_cfg: TrainConfig,
          env_cfg: EnvConfig,
          obs_cfg: ObsConfig,
          reward_cfg: RewardConfig,
          mappo_cfg: MAPPOConfig,
          device: str = "cpu",
          phase: int = 1):
    
    # ── Setup ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  S-MAS Multi-Agent Training — Phase {phase}")
    print("=" * 70)
    print(f"  Agents: 3 (Independent Trunks)")
    print(f"  Envs:   {train_cfg.num_envs}")
    print(f"  Device: {device}")
    print(f"  Rewards: alive={reward_cfg.w_alive:.1f} mission={reward_cfg.w_valid_target:.1f} sloth={-reward_cfg.w_sloth_penalty:.1f}")
    print("=" * 70 + "\n")

    # ── Initialize environments ──────────────────────────────────
    envs = [SMASWrapper(env_cfg) for _ in range(train_cfg.num_envs)]
    obs_builder = ObservationBuilder()
    reward_fn = SurvivalReward(reward_cfg) if phase < 3 else MissionReward(reward_cfg)

    obs_list = [obs_builder.build(e.reset(randomize=True)) for e in envs]
    done_list = [False] * train_cfg.num_envs
    
    total_steps = 0
    episode_count = 0
    
    # Brain
    model = SharedActorCritic(obs_cfg.obs_dim, mappo_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    
    # Buffer: Stores data for all environments combined
    # Capacity = rollout_steps * num_envs
    buffer = RolloutBuffer(mappo_cfg.rollout_steps * train_cfg.num_envs, obs_cfg.obs_dim, 4, device)

    # ── Main training loop ─────────────────────────────────────────
    while total_steps < train_cfg.total_timesteps:
        episode_start = time.time()
        episode_reward = 0.0
        episode_steps = 0
        ep_payload_on_count = 0
        ep_valid_targets = 0
        ep_saa_violations = 0
        
        ep_policy_loss = 0.0
        ep_value_loss = 0.0
        ep_entropy = 0.0
        ep_update_count = 0

        # We keep running until Env[0] is done or we hit a step limit
        # This is a simple way to track "episodes" in a multi-env setup
        while not done_list[0]:
            buffer.reset()
            
            # ── Collect rollout ──
            for _ in range(mappo_cfg.rollout_steps):
                # 1. Batch Inference
                batch_obs = np.array(obs_list)
                obs_tensor = torch.tensor(batch_obs, device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    out = model.act(obs_tensor)
                
                nav_acts = out["nav_action"].cpu().numpy()
                raw_navs = out["raw_nav"].cpu().numpy()
                bus_acts = out["bus_action"].cpu().numpy()
                mis_acts = out["mission_action"].cpu().numpy()
                values   = out["value"].cpu().numpy()
                nav_lps  = out["nav_log_prob"].cpu().numpy()
                bus_lps  = out["bus_log_prob"].cpu().numpy()
                mis_lps  = out["mission_log_prob"].cpu().numpy()

                # 2. Step all envs
                for i in range(train_cfg.num_envs):
                    if done_list[i]: continue

                    action_dict = {
                        "nav": np.array([
                            nav_acts[i,0], nav_acts[i,1], nav_acts[i,2],
                            (nav_acts[i,3] + 1.0) / 2.0
                        ], dtype=np.float32),
                        "bus": int(bus_acts[i]),
                        "mission": int(mis_acts[i]) if phase >= 3 else 0,
                    }

                    raw_state, _, done, info = envs[i].step(action_dict)
                    next_obs = obs_builder.build(raw_state)

                    # Reward
                    if phase >= 3:
                        r, m_info = reward_fn.compute(raw_state, action_dict, done, info)
                        if i == 0:
                            if m_info["payload_on"]: ep_payload_on_count += 1
                            if m_info["valid_target"] and m_info["payload_on"]: ep_valid_targets += 1
                            if m_info["saa_violation"]: ep_saa_violations += 1
                    else:
                        r = reward_fn.compute(raw_state, action_dict, done, info)

                    r_scaled = r * 0.001
                    if i == 0: episode_reward += r

                    buffer.push(obs_list[i], raw_navs[i], float(bus_acts[i]), r_scaled, values[i],
                                nav_lps[i], bus_lps[i], done,
                                mission_act=float(mis_acts[i]), mission_lp=mis_lps[i])

                    obs_list[i] = next_obs
                    done_list[i] = done
                    total_steps += 1
                    episode_steps += 1

                if all(done_list): break
            
            # ── PPO Update ──
            # Estimate last value for GAE
            with torch.no_grad():
                # We use the value of the first environment that isn't done
                valid_idx = 0
                for idx, d in enumerate(done_list):
                    if not d:
                        valid_idx = idx
                        break
                last_obs_tensor = torch.tensor(obs_list[valid_idx], device=device).unsqueeze(0)
                last_val = model.get_value(last_obs_tensor).item() if not done_list[valid_idx] else 0.0

            buffer.compute_gae(last_val, mappo_cfg.gamma, mappo_cfg.gae_lambda)
            losses = ppo_update(model, optimizer, buffer, mappo_cfg, device)
            
            ep_policy_loss += losses.get("policy_loss", 0)
            ep_value_loss += losses.get("value_loss", 0)
            ep_entropy += losses.get("entropy", 0)
            ep_update_count += 1

            if done_list[0]: break

        # ── Episode Summary ──
        episode_count += 1
        ep_time = time.time() - episode_start
        sps = episode_steps / max(ep_time, 1e-6)
        
        avg_pi = ep_policy_loss / max(ep_update_count, 1)
        avg_v  = ep_value_loss / max(ep_update_count, 1)
        avg_ent = ep_entropy / max(ep_update_count, 1)

        # Print
        base = (f"  Ep {episode_count:5d} | "
                f"Steps {total_steps:8,d} | "
                f"R {episode_reward:8.1f} | "
                f"SoC {envs[0].last_state.battery_soc*100:5.1f}% | "
                f"Alt {envs[0].last_state.altitude_km:6.1f}km | "
                f"FDIR {envs[0].last_state.fdir_mode} | "
                f"pi {avg_pi:.4f} | "
                f"v {avg_v:.4f} | "
                f"H {avg_ent:.3f}")
        if phase >= 3:
            base += (f" | PL {ep_payload_on_count:4d} "
                     f"VT {ep_valid_targets:3d} "
                     f"SAA! {ep_saa_violations:2d}")
        print(base)

        # Save
        if episode_count % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/mappo_phase{phase}_ep{episode_count}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "best_reward": episode_reward,
                "episode": episode_count,
                "phase": phase
            }, path)
            print(f"    â†’ Checkpoint saved: {path}")

        # Reset for next episode
        for i in range(train_cfg.num_envs):
            obs_list[i] = obs_builder.build(envs[i].reset(randomize=True))
            done_list[i] = False

    print("\nTraining Complete.")

def main():
    parser = argparse.ArgumentParser(description="S-MAS MAPPO Training")
    parser.add_argument("--total_steps", type=int, default=10_000_000)
    parser.add_argument("--rollout_steps", type=int, default=1176)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--phase", type=int, default=1)
    args = parser.parse_args()

    # Load configs
    train_cfg = TrainConfig(total_timesteps=args.total_steps, lr=args.lr, device=args.device)
    env_cfg = EnvConfig()
    obs_cfg = ObsConfig()
    reward_cfg = RewardConfig()
    mappo_cfg = MAPPOConfig(rollout_steps=args.rollout_steps)

    train(train_cfg, env_cfg, obs_cfg, reward_cfg, mappo_cfg, args.device, args.phase)

if __name__ == "__main__":
    main()
