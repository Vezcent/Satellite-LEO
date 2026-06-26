import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainConfig, ObsConfig, RewardConfig, MissionRewardConfig, MAPPOConfig, EnvConfig
from train import train

def run_experiment(name: str, 
                   seed: int, 
                   shared_policy: bool, 
                   log_suffix: str,
                   checkpoint_dir: str,
                   total_steps: int,
                   w_fdir: float = 200.0,
                   w_fatal: float = 50000.0,
                   w_dod: float = 50.0):
    
    print("\n" + "=" * 80)
    print(f"  STARTING EXPERIMENT: {name}")
    print(f"  Seed: {seed} | Shared Policy: {shared_policy} | Steps: {total_steps:,}")
    print(f"  Reward Weights: w_fdir={w_fdir:.1f}, w_fatal={w_fatal:.1f}, w_dod={w_dod:.1f}")
    print(f"  Checkpoint Dir: {checkpoint_dir}")
    print("" + "=" * 80 + "\n")
    
    # Initialize standard configurations
    train_cfg = TrainConfig(total_timesteps=total_steps, seed=seed)
    env_cfg = EnvConfig(seed=seed)
    obs_cfg = ObsConfig()
    reward_cfg = RewardConfig(w_fdir=w_fdir, w_fatal=w_fatal, w_dod=w_dod)
    mission_rew_cfg = MissionRewardConfig()
    mappo_cfg = MAPPOConfig(shared_policy=shared_policy)
    
    # Override paths and identifiers
    train_cfg.checkpoint_dir = checkpoint_dir
    train_cfg.log_suffix = log_suffix
    
    # Run training (Phase 3: full mission control)
    train(
        train_cfg=train_cfg,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        mission_rew_cfg=mission_rew_cfg,
        mappo_cfg=mappo_cfg,
        device="cpu",
        phase=3,
        resume_ckpt=None
    )

def main():
    parser = argparse.ArgumentParser(description="S-MAS Multi-Seed & Ablation Trainer")
    parser.add_argument("--steps", type=int, default=1600000, 
                        help="Number of steps per training run (default: 1,600,000 for quick convergence)")
    parser.add_argument("--run_type", type=str, choices=["all", "mappo", "ippo", "ablation"], default="all",
                        help="Which experiments to run (default: all)")
    args = parser.parse_args()
    
    # 1. MAPPO multi-seed runs (seeds 42, 43, 44)
    if args.run_type in ["all", "mappo"]:
        for seed in [42, 43, 44]:
            run_experiment(
                name=f"MAPPO Seed {seed}",
                seed=seed,
                shared_policy=True,
                log_suffix=f"_seed{seed}",
                checkpoint_dir=f"checkpoints/mappo_seed{seed}",
                total_steps=args.steps
            )
            
    # 2. IPPO multi-seed runs (seeds 42, 43, 44)
    if args.run_type in ["all", "ippo"]:
        for seed in [42, 43, 44]:
            run_experiment(
                name=f"IPPO Seed {seed}",
                seed=seed,
                shared_policy=False,
                log_suffix=f"_ippo_seed{seed}",
                checkpoint_dir=f"checkpoints/ippo_seed{seed}",
                total_steps=args.steps
            )
            
    # 3. Ablation studies (based on seed 42)
    if args.run_type in ["all", "ablation"]:
        # Ablation 1: No FDIR penalty
        run_experiment(
            name="Ablation: No FDIR Penalty (w_fdir=0)",
            seed=42,
            shared_policy=True,
            log_suffix="_ablation_fdir",
            checkpoint_dir="checkpoints/ablation_fdir",
            total_steps=args.steps,
            w_fdir=0.0
        )
        # Ablation 2: No Fatal penalty
        run_experiment(
            name="Ablation: No Fatal Penalty (w_fatal=0)",
            seed=42,
            shared_policy=True,
            log_suffix="_ablation_fatal",
            checkpoint_dir="checkpoints/ablation_fatal",
            total_steps=args.steps,
            w_fatal=0.0
        )
        # Ablation 3: No DoD penalty
        run_experiment(
            name="Ablation: No DoD Penalty (w_dod=0)",
            seed=42,
            shared_policy=True,
            log_suffix="_ablation_dod",
            checkpoint_dir="checkpoints/ablation_dod",
            total_steps=args.steps,
            w_dod=0.0
        )

if __name__ == "__main__":
    main()
