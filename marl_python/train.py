"""
S-MAS Phase 2 — Task 2.5
Training Loop: Reset → Rollout → GAE → Update Policy → Log Metrics.

Usage
-----
  cd marl_python
  python train.py                     # default settings
  python train.py --total_steps 1000000 --device cuda
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch

# Ensure we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (EnvConfig, ObsConfig, ActionConfig,
                     RewardConfig, MAPPOConfig, TrainConfig)
from env_wrapper import SatelliteEnv
from observation import ObservationBuilder
from reward import SurvivalReward
from mappo import SharedActorCritic, RolloutBuffer, ppo_update


# ═══════════════════════════════════════════════════════════════════
#  Seed everything
# ═══════════════════════════════════════════════════════════════════

def set_global_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════

def train(train_cfg: TrainConfig,
          env_cfg: EnvConfig,
          obs_cfg: ObsConfig,
          rew_cfg: RewardConfig,
          mappo_cfg: MAPPOConfig):

    set_global_seeds(train_cfg.seed)
    device = train_cfg.device

    # ── Create environment, observation builder, reward ────────────
    env = SatelliteEnv(env_cfg)
    obs_builder = ObservationBuilder(obs_cfg)
    reward_fn = SurvivalReward(rew_cfg)

    obs_dim = obs_builder.obs_dim
    print(f"Observation dim: {obs_dim}")
    print(f"Device: {device}")

    # ── Create model & optimizer ───────────────────────────────────
    model = SharedActorCritic(obs_dim, mappo_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mappo_cfg.lr)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ── Rollout buffer ─────────────────────────────────────────────
    buffer = RolloutBuffer(mappo_cfg.rollout_steps, obs_dim, nav_dim=4)

    # ── Logging setup ──────────────────────────────────────────────
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    log_path = os.path.join(train_cfg.log_dir, "train_log.jsonl")
    log_file = open(log_path, "w")

    # ── Metrics tracking ───────────────────────────────────────────
    total_steps = 0
    episode_count = 0
    best_ep_reward = -float("inf")

    print("\n" + "=" * 70)
    print("  S-MAS MAPPO Training — Phase 2 (Survival)")
    print("=" * 70)
    print(f"  Rollout steps:   {mappo_cfg.rollout_steps}")
    print(f"  Batch size:      {mappo_cfg.batch_size}")
    print(f"  PPO epochs:      {mappo_cfg.num_epochs}")
    print(f"  Total timesteps: {train_cfg.total_timesteps:,}")
    print(f"  Episode length:  {env_cfg.max_steps_per_episode} steps "
          f"(~{env_cfg.max_steps_per_episode * env_cfg.dt / 3600:.1f}h)")
    print("=" * 70 + "\n")

    # ── Main training loop ─────────────────────────────────────────
    while total_steps < train_cfg.total_timesteps:
        # ── Episode start ──
        raw_state = env.reset()
        obs = obs_builder.build(raw_state)
        episode_reward = 0.0
        episode_steps = 0
        episode_start = time.time()

        done = False
        while not done:
            # ── Collect rollout ────────────────────────────────────
            buffer.reset()
            for _ in range(mappo_cfg.rollout_steps):
                obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)

                with torch.no_grad():
                    out = model.act(obs_tensor)

                nav_action = out["nav_action"].squeeze(0).cpu().numpy()
                bus_action = int(out["bus_action"].item())
                value      = out["value"].item()
                nav_lp     = out["nav_log_prob"].item()
                bus_lp     = out["bus_log_prob"].item()

                # Convert nav action: tanh already applied, scale throttle
                action_dict = {
                    "nav": np.array([
                        nav_action[0],              # thrust_x [-1,1]
                        nav_action[1],              # thrust_y [-1,1]
                        nav_action[2],              # thrust_z [-1,1]
                        (nav_action[3] + 1.0) / 2.0 # throttle: tanh→[0,1]
                    ], dtype=np.float32),
                    "bus": bus_action,
                }

                raw_state, _, done, info = env.step(action_dict)
                next_obs = obs_builder.build(raw_state)

                r = reward_fn.compute(raw_state, action_dict, done, info)
                episode_reward += r

                buffer.push(obs, nav_action, float(bus_action),
                            r, value, nav_lp, bus_lp, done)

                obs = next_obs
                episode_steps += 1
                total_steps += 1

                if done:
                    break

            # ── GAE computation ────────────────────────────────────
            if done:
                last_val = 0.0
            else:
                with torch.no_grad():
                    last_val = model.get_value(
                        torch.tensor(obs, device=device).unsqueeze(0)
                    ).item()

            buffer.compute_gae(last_val, mappo_cfg.gamma, mappo_cfg.gae_lambda)

            # ── PPO update ─────────────────────────────────────────
            losses = ppo_update(model, optimizer, buffer, mappo_cfg, device)

            if done:
                break

        # ── Episode complete ───────────────────────────────────────
        episode_count += 1
        ep_time = time.time() - episode_start
        sps = episode_steps / max(ep_time, 1e-6)

        # Log entry
        log_entry = {
            "episode":    episode_count,
            "steps":      episode_steps,
            "total_steps": total_steps,
            "reward":     round(float(episode_reward), 2),
            "soc_final":  round(float(raw_state.battery_soc) * 100, 1),
            "alt_final":  round(float(raw_state.altitude_km), 1),
            "fdir":       int(raw_state.fdir_mode),
            "done_reason": int(raw_state.done_reason),
            "policy_loss": round(float(losses.get("policy_loss", 0)), 5),
            "value_loss":  round(float(losses.get("value_loss", 0)), 5),
            "entropy":     round(float(losses.get("entropy", 0)), 4),
            "sps":         round(float(sps), 0),
            "time_s":      round(float(ep_time), 1),
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        # Print progress
        if episode_count % train_cfg.log_interval == 0 or episode_count == 1:
            print(f"  Ep {episode_count:5d} | "
                  f"Steps {total_steps:8,d} | "
                  f"R {episode_reward:8.1f} | "
                  f"SoC {raw_state.battery_soc*100:5.1f}% | "
                  f"Alt {raw_state.altitude_km:6.1f}km | "
                  f"FDIR {raw_state.fdir_mode} | "
                  f"Done {raw_state.done_reason} | "
                  f"pi_loss {losses.get('policy_loss',0):.4f} | "
                  f"v_loss {losses.get('value_loss',0):.4f} | "
                  f"H {losses.get('entropy',0):.3f} | "
                  f"{sps:.0f} sps")

        # Save checkpoint
        if episode_count % train_cfg.save_interval == 0:
            ckpt_path = os.path.join(
                train_cfg.checkpoint_dir,
                f"mappo_ep{episode_count}.pt"
            )
            torch.save({
                "episode": episode_count,
                "total_steps": total_steps,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_reward": best_ep_reward,
            }, ckpt_path)
            print(f"    → Checkpoint saved: {ckpt_path}")

        if episode_reward > best_ep_reward:
            best_ep_reward = episode_reward
            best_path = os.path.join(train_cfg.checkpoint_dir, "mappo_best.pt")
            torch.save({
                "episode": episode_count,
                "total_steps": total_steps,
                "model_state": model.state_dict(),
                "reward": best_ep_reward,
            }, best_path)

    # ── Cleanup ────────────────────────────────────────────────────
    log_file.close()
    env.close()

    print("\n" + "=" * 70)
    print(f"  Training complete. {episode_count} episodes, "
          f"{total_steps:,} total steps.")
    print(f"  Best episode reward: {best_ep_reward:.1f}")
    print(f"  Logs: {log_path}")
    print(f"  Checkpoints: {train_cfg.checkpoint_dir}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="S-MAS MAPPO Training")
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--rollout_steps", type=int, default=1176)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_steps", type=int, default=17_280)
    args = parser.parse_args()

    train_cfg = TrainConfig(
        total_timesteps=args.total_steps,
        seed=args.seed,
        device=args.device,
    )
    env_cfg = EnvConfig(
        seed=args.seed,
        max_steps_per_episode=args.max_episode_steps,
    )
    mappo_cfg = MAPPOConfig(
        rollout_steps=args.rollout_steps,
        lr=args.lr,
    )

    train(train_cfg, env_cfg, ObsConfig(), RewardConfig(), mappo_cfg)


if __name__ == "__main__":
    main()
