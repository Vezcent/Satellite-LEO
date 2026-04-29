"""
S-MAS Phase 3 â€” Task 3.4
ONNX Export Pipeline with Dynamic Axes and FP16.

Exports the trained SharedActorCritic model to ONNX format for
deployment via the ONNX Runtime C++ API in Phase 4.

Produces three separate ONNX files:
  smas_nav.onnx     â€” Navigation head (continuous 4D output)
  smas_bus.onnx     â€” Resource head (binary deep_sleep logit)
  smas_mission.onnx â€” Mission head (binary payload_on logit)

Usage
-----
  cd marl_python
  python export_onnx.py --checkpoint checkpoints/mappo_phase3_best.pt
  python export_onnx.py --checkpoint checkpoints/mappo_phase3_best.pt --fp16
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ObsConfig, MAPPOConfig
from mappo import SharedActorCritic


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Wrapper modules for clean ONNX export
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class NavExportModule(nn.Module):
    """Wraps trunk + NavigationHead for ONNX export.
    Output: action[4] (tanh-squashed mu) — deterministic policy."""
    def __init__(self, model: SharedActorCritic):
        super().__init__()
        self.trunk = model.nav_trunk
        self.nav_head = model.nav_head

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        mu, std = self.nav_head(features)
        # Apply tanh to match training-time squashing
        action = torch.tanh(mu)
        return action


class BusExportModule(nn.Module):
    """Wraps trunk + ResourceHead for ONNX export.
    Output: logit (1-dim per batch)."""
    def __init__(self, model: SharedActorCritic):
        super().__init__()
        self.trunk = model.bus_trunk
        self.bus_head = model.bus_head

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        logit = self.bus_head(features)
        return logit


class MissionExportModule(nn.Module):
    """Wraps trunk + MissionHead for ONNX export.
    Output: logit (1-dim per batch)."""
    def __init__(self, model: SharedActorCritic):
        super().__init__()
        self.trunk = model.mission_trunk
        self.mission_head = model.mission_head

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        logit = self.mission_head(features)
        return logit


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Export logic
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def export_model(checkpoint_path: str,
                 output_dir: str = "onnx_export",
                 fp16: bool = False,
                 obs_dim: int = None):
    """
    Load a checkpoint and export all three heads to ONNX.

    Parameters
    ----------
    checkpoint_path : str â€” path to .pt checkpoint
    output_dir : str â€” directory for ONNX files
    fp16 : bool â€” convert to half precision
    obs_dim : int â€” observation dimension (default from ObsConfig)
    """
    if obs_dim is None:
        obs_dim = ObsConfig().obs_dim

    print("=" * 60)
    print("  S-MAS ONNX Export Pipeline â€” Phase 3")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  FP16:       {fp16}")
    print(f"  Obs dim:    {obs_dim}")
    print()

    # â”€â”€ Load checkpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SharedActorCritic(obs_dim, MAPPOConfig())
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Loaded episode {ckpt.get('episode', '?')}, "
          f"phase {ckpt.get('phase', '?')}, "
          f"reward {ckpt.get('reward', ckpt.get('best_reward', '?'))}")

    os.makedirs(output_dir, exist_ok=True)

    # â”€â”€ Prepare export modules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    modules = {
        "smas_nav":     NavExportModule(model),
        "smas_bus":     BusExportModule(model),
        "smas_mission": MissionExportModule(model),
    }

    output_names_map = {
        "smas_nav":     ["action"],
        "smas_bus":     ["logit"],
        "smas_mission": ["logit"],
    }

    # â”€â”€ Export each head â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for name, module in modules.items():
        module.eval()

        # Prepare dummy input
        dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            module = module.half()
        dummy = torch.randn(1, obs_dim, dtype=dtype)

        out_path = os.path.join(output_dir, f"{name}.onnx")
        out_names = output_names_map[name]

        # Dynamic axes: batch dimension is variable
        dynamic_axes = {"obs_input": {0: "batch_size"}}
        for oname in out_names:
            dynamic_axes[oname] = {0: "batch_size"}

        torch.onnx.export(
            module,
            dummy,
            out_path,
            opset_version=14,
            do_constant_folding=True,
            input_names=["obs_input"],
            output_names=out_names,
            dynamic_axes=dynamic_axes,
        )

        size_kb = os.path.getsize(out_path) / 1024
        print(f"  âœ“ Exported: {out_path}  ({size_kb:.1f} KB)")

    # â”€â”€ Verification with ONNX Runtime â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n  Verifying ONNX outputs (Numerical Parity)...")
    try:
        import onnxruntime as ort

        # Create a single test batch for consistent comparison
        test_input_np = np.random.randn(1, obs_dim).astype(np.float32)
        test_input_torch = torch.from_numpy(test_input_np)

        for name, module in modules.items():
            onnx_path = os.path.join(output_dir, f"{name}.onnx")
            session = ort.InferenceSession(onnx_path)
            
            # 1. Run ONNX Inference
            # Handle FP16 input cast if needed
            ort_in = test_input_np.astype(np.float16) if fp16 else test_input_np
            ort_out = session.run(None, {"obs_input": ort_in})

            # 2. Run PyTorch Inference
            module.eval()
            with torch.no_grad():
                pt_in = test_input_torch.half() if fp16 else test_input_torch
                pt_out = module(pt_in)
                
                # Handle tuple vs single tensor output
                if isinstance(pt_out, tuple):
                    pt_out_np = [p.cpu().numpy() for p in pt_out]
                else:
                    pt_out_np = [pt_out.cpu().numpy()]

            # 3. Compare
            diffs = []
            for i in range(len(ort_out)):
                diff = np.max(np.abs(ort_out[i].astype(np.float32) - pt_out_np[i].astype(np.float32)))
                diffs.append(diff)
            
            max_diff = max(diffs)
            status = "âœ“" if max_diff < (1e-3 if fp16 else 1e-5) else "Γ£ù"
            print(f"    {name:13}: Max Diff = {max_diff:.8f} {status}")

        print("\n  ONNX verification COMPLETE")

    except ImportError:
        print("    âš  onnxruntime not installed â€” skipping verification.")
        print("    Install with: pip install onnxruntime")

    print("\n" + "=" * 60)
    print("  Export complete.")
    print("=" * 60)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CLI entry point
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _find_best_checkpoint(ckpt_dir: str = "checkpoints") -> str:
    """Find the best checkpoint from the FIXED training pipeline based on reward.
    """
    import glob
    candidates = glob.glob(os.path.join(ckpt_dir, "mappo_phase3_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    
    new_pipeline = candidates
    
    best_file = None
    best_reward = float("-inf")
    
    # Read each checkpoint and find the one with the highest reward
    for f in new_pipeline:
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            rew = ckpt.get("best_reward", float("-inf"))
            if rew > best_reward:
                best_reward = rew
                best_file = f
        except Exception:
            continue
            
    if best_file is None:
        # Fallback to newest if we couldn't read rewards
        new_pipeline.sort(key=os.path.getmtime, reverse=True)
        return new_pipeline[0]
        
    return best_file


def main():
    parser = argparse.ArgumentParser(description="S-MAS ONNX Export")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint file (auto-detects BEST if omitted)")
    parser.add_argument("--output_dir", type=str, default="onnx_export",
                        help="Directory for ONNX files")
    parser.add_argument("--fp16", action="store_true",
                        help="Export in half precision (FP16)")
    parser.add_argument("--deploy", action="store_true",
                        help="Copy exported models to controller_csharp/models/")
    args = parser.parse_args()

    # Auto-detect best checkpoint if not specified
    if args.checkpoint is None:
        args.checkpoint = _find_best_checkpoint()
        print(f"  Auto-detected best checkpoint: {args.checkpoint}")

    export_model(args.checkpoint, args.output_dir, args.fp16)

    # Deploy to C# controller if requested
    if args.deploy:
        import shutil
        deploy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "controller_csharp", "models")
        os.makedirs(deploy_dir, exist_ok=True)
        for f in ["smas_nav.onnx", "smas_bus.onnx", "smas_mission.onnx"]:
            src = os.path.join(args.output_dir, f)
            dst = os.path.join(deploy_dir, f)
            shutil.copy2(src, dst)
        print(f"\n  Deployed to: {deploy_dir}")


if __name__ == "__main__":
    main()

