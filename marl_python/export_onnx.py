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
    Output: (mu, std) â€” 4-dim each."""
    def __init__(self, model: SharedActorCritic):
        super().__init__()
        self.trunk = model.trunk
        self.nav_head = model.nav_head

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        mu, std = self.nav_head(features)
        return mu, std


class BusExportModule(nn.Module):
    """Wraps trunk + ResourceHead for ONNX export.
    Output: logit (1-dim per batch)."""
    def __init__(self, model: SharedActorCritic):
        super().__init__()
        self.trunk = model.trunk
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
        self.trunk = model.trunk
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
        "smas_nav":     ["mu", "std"],
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
    print("\n  Verifying ONNX outputs...")
    try:
        import onnxruntime as ort

        test_input = np.random.randn(4, obs_dim).astype(
            np.float16 if fp16 else np.float32)

        for name in modules:
            onnx_path = os.path.join(output_dir, f"{name}.onnx")
            session = ort.InferenceSession(onnx_path)
            ort_out = session.run(None, {"obs_input": test_input})

            shapes = [o.shape for o in ort_out]
            print(f"    {name}: batch=4 â†’ output shapes {shapes} âœ“")

        print("\n  ONNX verification PASSED âœ“")

    except ImportError:
        print("    âš  onnxruntime not installed â€” skipping verification.")
        print("    Install with: pip install onnxruntime")

    print("\n" + "=" * 60)
    print("  Export complete.")
    print("=" * 60)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  CLI entry point
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main():
    parser = argparse.ArgumentParser(description="S-MAS ONNX Export")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--output_dir", type=str, default="onnx_export",
                        help="Directory for ONNX files")
    parser.add_argument("--fp16", action="store_true",
                        help="Export in half precision (FP16)")
    args = parser.parse_args()

    export_model(args.checkpoint, args.output_dir, args.fp16)


if __name__ == "__main__":
    main()

