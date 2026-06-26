"""
S-MAS IPPO Training Script
Trains with shared_policy=False (independent trunks per agent).
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "marl_python"))

from config import TrainConfig, ObsConfig, RewardConfig, MissionRewardConfig, MAPPOConfig, EnvConfig
from train import train

def main():
    parser = argparse.ArgumentParser(description="S-MAS IPPO Training")
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # IPPO = Independent trunks (shared_policy=False)
    mappo_cfg = MAPPOConfig(shared_policy=False, rollout_steps=1176, lr=3e-4)
    
    env_cfg = EnvConfig(seed=args.seed)
    obs_cfg = ObsConfig()
    reward_cfg = RewardConfig()
    mission_rew_cfg = MissionRewardConfig()
    
    # Save to ippo-specific directory
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", f"ippo_seed{args.seed}")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "marl_python", "logs")
    
    train_cfg = TrainConfig(
        total_timesteps=args.total_steps,
        device="cpu",
        seed=args.seed,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
    )
    # Tag log file
    train_cfg.log_suffix = f"_ippo_seed{args.seed}"
    
    print(f"Training IPPO (Independent PPO) with shared_policy=False")
    print(f"  Steps: {args.total_steps}, Seed: {args.seed}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    
    train(train_cfg, env_cfg, obs_cfg, reward_cfg, mission_rew_cfg, mappo_cfg, 
          device="cpu", phase=3, resume_ckpt=None)

if __name__ == "__main__":
    main()
