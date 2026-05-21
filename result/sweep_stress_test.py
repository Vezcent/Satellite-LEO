"""
S-MAS Automated Stress Test Sweep
Runs multiple simulations across SEU multiplier levels and seeds,
then generates a survival curve for paper results.

Usage:
    python sweep_stress_test.py                # Full sweep
    python sweep_stress_test.py --quick        # Quick test (3 levels x 2 seeds)
    python sweep_stress_test.py --dry-run      # Show plan without running
"""
import subprocess
import os
import sys
import time
import argparse

BASE_DIR = r"E:\Satellite LEO"
CONTROLLER_DIR = os.path.join(BASE_DIR, "controller_csharp")
VISUALIZE_SCRIPT = os.path.join(BASE_DIR, "result", "visualize", "visualize.py")

# ── Full sweep config ──
SEU_MULTS = [1, 2, 5, 10, 20, 50, 100]
SEEDS_PER_MULT = 10
SIM_STEPS = 63_072_000  # ~10 years at 5s timestep

# ── Quick test config ──
QUICK_MULTS = [1, 10, 100]
QUICK_SEEDS = 2
QUICK_STEPS = 172_800  # ~10 days


def run_single(seu_mult: float, seed: int, steps: int) -> int:
    """Run a single simulation and return exit code."""
    cmd = [
        "dotnet", "run", "-c", "Release", "--",
        "--steps", str(steps),
        "--seed", str(seed),
        "--no-ws",  # No WebSocket needed for batch runs
    ]
    print(f"  Running SEU={seu_mult}x  seed={seed}  steps={steps}...")
    start = time.time()

    result = subprocess.run(cmd, cwd=CONTROLLER_DIR,
                            capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"    ✗ FAILED ({elapsed:.1f}s): {result.stderr[:200]}")
    else:
        print(f"    ✓ Done ({elapsed:.1f}s)")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="S-MAS Stress Test Sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (3 levels x 2 seeds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without running")
    args = parser.parse_args()

    mults = QUICK_MULTS if args.quick else SEU_MULTS
    seeds = QUICK_SEEDS if args.quick else SEEDS_PER_MULT
    steps = QUICK_STEPS if args.quick else SIM_STEPS
    total_runs = len(mults) * seeds

    print("=" * 60)
    print("  S-MAS Stress Test Sweep")
    print("=" * 60)
    print(f"  Mode:           {'QUICK' if args.quick else 'FULL'}")
    print(f"  SEU Multipliers: {mults}")
    print(f"  Seeds per mult:  {seeds}")
    print(f"  Steps per run:   {steps:,} (~{steps * 5 / 86400:.0f} days)")
    print(f"  Total runs:      {total_runs}")
    print()

    if args.dry_run:
        print("  DRY RUN — no simulations will be executed.")
        for mult in mults:
            for seed_idx in range(seeds):
                seed = seed_idx + 42
                print(f"    Would run: SEU={mult}x  seed={seed}  steps={steps}")
        return

    # ── Run sweep ──
    results = []
    failed = 0
    sweep_start = time.time()

    for mult in mults:
        print(f"\n── SEU Multiplier: {mult}x ──")
        for seed_idx in range(seeds):
            seed = seed_idx + 42
            rc = run_single(mult, seed, steps)
            results.append({"seu_mult": mult, "seed": seed, "ok": rc == 0})
            if rc != 0:
                failed += 1

    sweep_elapsed = time.time() - sweep_start

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SWEEP COMPLETE")
    print("=" * 60)
    print(f"  Total runs:  {total_runs}")
    print(f"  Succeeded:   {total_runs - failed}")
    print(f"  Failed:      {failed}")
    print(f"  Total time:  {sweep_elapsed:.0f}s ({sweep_elapsed / 60:.1f}m)")

    # ── Generate survival curve ──
    print("\n  Generating survival curve...")
    subprocess.run([sys.executable, VISUALIZE_SCRIPT, "--survival-curve"],
                   cwd=BASE_DIR)
    print("  Done! Check result/save/survival_curve.png")


if __name__ == "__main__":
    main()
