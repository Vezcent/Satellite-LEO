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

from config import TrainConfig, ObsConfig, RewardConfig, MissionRewardConfig, MAPPOConfig, EnvConfig
from mappo import SharedActorCritic, RolloutBuffer, ppo_update
from env_wrapper import SatelliteEnv, VectorSatelliteEnv
from observation import ObservationBuilder
from reward import SurvivalReward, MissionReward

def train(train_cfg: TrainConfig,
          env_cfg: EnvConfig,
          obs_cfg: ObsConfig,
          reward_cfg: RewardConfig,
          mission_rew_cfg: MissionRewardConfig,
          mappo_cfg: MAPPOConfig,
          device: str = "cpu",
          phase: int = 1,
          resume_ckpt: str = None):
    
    # ── Setup ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  S-MAS Multi-Agent Training — Phase {phase}")
    print("=" * 70)
    print(f"  Agents: 3 (Independent Trunks)")
    print(f"  Envs:   {env_cfg.num_envs}")
    print(f"  Device: {device}")
    print(f"  Rewards: alive={reward_cfg.w_alive:.1f} mission={mission_rew_cfg.w_valid_target:.1f} sloth={-mission_rew_cfg.w_sloth_penalty:.1f}")
    print("=" * 70 + "\n")

    # ── Initialize environments ──────────────────────────────────
    vec_env = VectorSatelliteEnv(env_cfg)
    obs_builder = ObservationBuilder()
    _build_obs = obs_builder.build  # cache method ref (avoids repeated attr lookup)
    reward_fn = SurvivalReward(reward_cfg) if phase < 3 else MissionReward(reward_cfg, mission_rew_cfg)

    obs_list = [_build_obs(state,
                           target_alt_km=vec_env.envs[i]._target_alt_km)
                for i, state in enumerate(vec_env.reset(randomize=True))]
    done_list = [False] * env_cfg.num_envs
    # Pre-allocate batch observation array (reused every rollout step)
    batch_obs = np.empty((env_cfg.num_envs, obs_cfg.obs_dim), dtype=np.float32)
    
    # Brain
    model = SharedActorCritic(obs_cfg.obs_dim, mappo_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mappo_cfg.lr)
    
    # Buffers: One for each environment to prevent GAE timeline mixing
    buffers = [RolloutBuffer(mappo_cfg.rollout_steps, obs_cfg.obs_dim, 4) 
               for _ in range(env_cfg.num_envs)]

    total_steps = 0
    episode_count = 0

    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"  Resuming from: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        episode_count = ckpt.get("episode", 0)
        # Old checkpoints didn't have total_steps and were 17,280 steps/ep
        total_steps = ckpt.get("total_steps", episode_count * 17280)
        print(f"  Loaded episode {episode_count}, total_steps {total_steps}")

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
        while episode_steps < env_cfg.max_steps_per_episode:
            for b in buffers: b.reset()
            
            # ── Collect rollout ──
            for _ in range(mappo_cfg.rollout_steps):
                # 1. Batch Inference (reuse pre-allocated array)
                for _bi in range(env_cfg.num_envs):
                    batch_obs[_bi] = obs_list[_bi]
                obs_tensor = torch.from_numpy(batch_obs).to(device)
                
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

                # 2. Step all envs in parallel
                action_dicts = []
                for i in range(env_cfg.num_envs):
                    action_dicts.append({
                        "nav": np.array([
                            nav_acts[i,0], nav_acts[i,1], nav_acts[i,2],
                            ((nav_acts[i,3] + 1.0) / 2.0) if ((nav_acts[i,3] + 1.0) / 2.0) > 0.05 else 0.0
                        ], dtype=np.float32),
                        "bus": int(bus_acts[i]),
                        "mission": int(mis_acts[i]) if phase >= 3 else 0,
                    })

                results = vec_env.step(action_dicts)

                for i in range(env_cfg.num_envs):
                    raw_state, _, done, info = results[i]
                    action_dict = action_dicts[i]

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

                    buffers[i].push(obs_list[i], raw_navs[i], float(bus_acts[i]), r_scaled, values[i],
                                    nav_lps[i], bus_lps[i], done,
                                    mission_act=float(mis_acts[i]), mission_lp=mis_lps[i])

                    # Auto-reset environment on death so it cannot skip time
                    if done:
                        raw_state = vec_env.reset_at(i, randomize=True)

                    obs_list[i] = _build_obs(raw_state,
                                             target_alt_km=vec_env.envs[i]._target_alt_km)
                    done_list[i] = done
                    total_steps += 1
                    if total_steps >= train_cfg.total_timesteps:
                        break
                
                # episode_steps tracks simulation time (1 step = 5s across all envs)
                episode_steps += 1
                if total_steps >= train_cfg.total_timesteps:
                    break

            # ── PPO Update ──
            # Estimate last value and compute GAE for each environment separately
            for i in range(env_cfg.num_envs):
                if done_list[i]:
                    last_val = 0.0
                else:
                    with torch.no_grad():
                        last_obs_tensor = torch.tensor(obs_list[i], device=device).unsqueeze(0)
                        last_val = model.get_value(last_obs_tensor).item()
                buffers[i].compute_gae(last_val, mappo_cfg.gamma, mappo_cfg.gae_lambda)

            losses = ppo_update(model, optimizer, buffers, mappo_cfg, device)
            
            ep_policy_loss += losses.get("policy_loss", 0)
            ep_value_loss += losses.get("value_loss", 0)
            ep_entropy += losses.get("entropy", 0)
            ep_update_count += 1

            # Print intermediate progress every 5 updates so the user has immediate feedback
            if ep_update_count % 5 == 0:
                print(f"      Update {ep_update_count:3d}/{int(env_cfg.max_steps_per_episode/mappo_cfg.rollout_steps)} | "
                      f"Steps {total_steps:8,d} | "
                      f"R0_curr {episode_reward:8.1f} | "
                      f"pi {losses.get('policy_loss', 0):.4f} | "
                      f"v {losses.get('value_loss', 0):.4f} | "
                      f"H {losses.get('entropy', 0):.3f}")

            if total_steps >= train_cfg.total_timesteps:
                break

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
                f"SoC {vec_env.envs[0].state.battery_soc*100:5.1f}% | "
                f"Alt {vec_env.envs[0].state.altitude_km:6.1f}km | "
                f"Fuel {vec_env.envs[0].state.fuel_fraction*100:4.1f}% | "
                f"T {vec_env.envs[0].state.temp_battery:5.1f}C | "
                f"FDIR {vec_env.envs[0].state.fdir_mode} | "
                f"pi {avg_pi:.4f} | "
                f"v {avg_v:.4f} | "
                f"H {avg_ent:.3f}")
        if phase >= 3:
            base += (f" | PL {ep_payload_on_count:4d} "
                     f"VT {ep_valid_targets:3d} "
                     f"SAA! {ep_saa_violations:2d}")
        print(base)

        # Save
        if episode_count % 50 == 0 or total_steps >= train_cfg.total_timesteps:
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/mappo_phase{phase}_ep{episode_count}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_reward": episode_reward,
                "episode": episode_count,
                "total_steps": total_steps,
                "phase": phase
            }, path)
            print(f"    -> Checkpoint saved: {path}")

        # Reset for next episode
        for i, state in enumerate(vec_env.reset(randomize=True)):
            obs_list[i] = _build_obs(state, target_alt_km=vec_env.envs[i]._target_alt_km)
            done_list[i] = False

    print("\nTraining Complete.")

def main():
    parser = argparse.ArgumentParser(description="S-MAS MAPPO Training")
    parser.add_argument("--total_steps", type=int, default=40_000_000)
    parser.add_argument("--rollout_steps", type=int, default=1176)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Normalize device name ("gpu" → "cuda")
    if args.device.lower() == "gpu":
        args.device = "cuda"

    # Load configs
    train_cfg = TrainConfig(total_timesteps=args.total_steps, device=args.device)
    env_cfg = EnvConfig()
    obs_cfg = ObsConfig()
    reward_cfg = RewardConfig()
    mission_rew_cfg = MissionRewardConfig()
    mappo_cfg = MAPPOConfig(rollout_steps=args.rollout_steps, lr=args.lr)

    train(train_cfg, env_cfg, obs_cfg, reward_cfg, mission_rew_cfg, mappo_cfg, args.device, args.phase, args.resume)

if __name__ == "__main__":
    main()
