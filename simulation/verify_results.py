"""Verify simulation results — checks stored API results and optionally runs a fresh simulation.

Usage:
    python verify_results.py                   # inspect all stored API results
    python verify_results.py --run             # also run a fresh 500-step simulation
    python verify_results.py --file path.npz   # inspect a specific npz file
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
API_RESULTS = REPO_ROOT / "api" / "results"


def inspect_npz(path: Path) -> None:
    """Print detailed statistics for a result .npz file."""
    print(f"\n{'='*70}")
    print(f"File: {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{'='*70}")

    with np.load(str(path), allow_pickle=True) as data:
        keys = list(data.keys())
        print(f"Keys: {keys}")

        for key in keys:
            arr = data[key]
            print(f"\n  [{key}]  shape={arr.shape}  dtype={arr.dtype}")
            if arr.size == 0:
                print("    EMPTY ARRAY")
                continue
            if arr.dtype.kind == 'f':  # float
                print(f"    min={arr.min():.6g}  max={arr.max():.6g}  "
                      f"mean={arr.mean():.6g}  std={arr.std():.6g}")
                print(f"    zeros: {(arr == 0).sum()}/{arr.size} ({100*(arr==0).sum()/arr.size:.1f}%)")
                print(f"    NaN: {np.isnan(arr).sum()}  Inf: {np.isinf(arr).sum()}")
                nz = arr[arr != 0]
                if nz.size > 0:
                    print(f"    non-zero min={np.abs(nz).min():.6g}  max={np.abs(nz).max():.6g}")
            elif arr.dtype.kind in ('u', 'i'):  # unsigned/signed int
                print(f"    min={arr.min()}  max={arr.max()}  unique={np.unique(arr).tolist()[:20]}")
            else:
                print(f"    (non-numeric dtype)")

        # Velocity-specific analysis
        if "vel" in keys:
            vel = data["vel"]
            print(f"\n  --- Velocity field analysis ---")
            speed = np.linalg.norm(vel, axis=-1)
            print(f"  Speed: min={speed.min():.6g}  max={speed.max():.6g}  mean={speed.mean():.6g}")
            all_zero = np.all(vel == 0, axis=-1)
            print(f"  All-zero voxels: {all_zero.sum()}/{all_zero.size} ({100*all_zero.sum()/all_zero.size:.1f}%)")
            # Check per-layer (z)
            Nz = vel.shape[0]
            print(f"  Per z-layer max speed:")
            for z in range(Nz):
                layer_speed = np.linalg.norm(vel[z], axis=-1)
                ms = layer_speed.max()
                marker = " <-- ground" if z == 0 else ""
                if ms > 0 or z < 3 or z == Nz - 1:
                    print(f"    z={z:3d}: max_speed={ms:.6g}{marker}")

        elif "ux" in keys and "uy" in keys and "uz" in keys:
            ux, uy, uz = data["ux"], data["uy"], data["uz"]
            print(f"\n  --- Velocity component analysis ---")
            speed = np.sqrt(ux**2 + uy**2 + uz**2)
            print(f"  Speed: min={speed.min():.6g}  max={speed.max():.6g}  mean={speed.mean():.6g}")
            all_zero = (speed == 0)
            print(f"  Zero-speed voxels: {all_zero.sum()}/{all_zero.size} ({100*all_zero.sum()/all_zero.size:.1f}%)")
            if "solid" in keys:
                solid = data["solid"]
                fluid = solid == 0
                fluid_speed = speed[fluid]
                print(f"  Fluid cells: {fluid.sum()}  Solid cells: {(~fluid).sum()}")
                if fluid_speed.size > 0:
                    print(f"  Fluid speed: min={fluid_speed.min():.6g}  max={fluid_speed.max():.6g}  mean={fluid_speed.mean():.6g}")
                    fluid_zero = (fluid_speed == 0).sum()
                    print(f"  Fluid zero-speed: {fluid_zero}/{fluid_speed.size} ({100*fluid_zero/fluid_speed.size:.1f}%)")
            Nz = ux.shape[0]
            print(f"  Per z-layer max speed:")
            for z in range(min(Nz, 5)):
                ms = speed[z].max()
                print(f"    z={z:3d}: max_speed={ms:.6g}")
            if Nz > 5:
                print(f"    ...")
                ms = speed[-1].max()
                print(f"    z={Nz-1:3d}: max_speed={ms:.6g}")

        if "comfort" in keys:
            comfort = data["comfort"]
            print(f"\n  --- Comfort map ---")
            print(f"  min={comfort.min():.6g}  max={comfort.max():.6g}  mean={comfort.mean():.6g}")
            print(f"  zeros: {(comfort == 0).sum()}/{comfort.size} ({100*(comfort==0).sum()/comfort.size:.1f}%)")


def run_fresh_simulation():
    """Run a fresh simulation and inspect the output."""
    print("\n" + "#" * 70)
    print("# Running fresh CUDA simulation (500 steps, 8m pitch)")
    print("#" * 70)

    sys.path.insert(0, str(REPO_ROOT / "simulation" / "lbm_py"))
    from lbm.run_cuda import run_cuda_lbm

    voxel_path = str(REPO_ROOT / "assets" / "presets" / "chicago.h5")

    def on_progress(val):
        print(f"\r  Progress: {val*100:.0f}%", end="", flush=True)

    snapshots = []
    def on_snapshot(step, ux, uy, uz):
        speed = np.sqrt(ux**2 + uy**2 + uz**2)
        print(f"\n  Snapshot step={step}: max_speed={speed.max():.6g}, "
              f"mean_speed={speed[speed > 0].mean():.6g if (speed > 0).any() else 0}")
        snapshots.append((step, speed.max()))

    result = run_cuda_lbm(
        voxel_path,
        angle_deg=0,
        wind_speed=5.0,
        roughness=0.3,
        pitch=8.0,
        num_steps=500,
        progress_callback=on_progress,
        snapshot_interval=100,
        snapshot_callback=on_snapshot,
    )
    print()

    # Save and inspect
    out_path = Path("fresh_simulation_result.npz")
    np.savez(str(out_path), **result)
    print(f"Saved to {out_path}")
    inspect_npz(out_path)

    if snapshots:
        print("\n  Snapshot progression:")
        for step, max_speed in snapshots:
            bar = "#" * int(max_speed * 200)
            print(f"    step {step:5d}: max_speed={max_speed:.6g}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Verify simulation results")
    parser.add_argument("--run", action="store_true", help="Run a fresh simulation")
    parser.add_argument("--file", type=str, help="Inspect a specific .npz file")
    args = parser.parse_args()

    if args.file:
        inspect_npz(Path(args.file))
        return

    # Inspect all stored API results
    if API_RESULTS.exists():
        npz_files = sorted(API_RESULTS.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if npz_files:
            print(f"Found {len(npz_files)} result files in {API_RESULTS}")
            # Show latest 3
            for f in npz_files[-3:]:
                inspect_npz(f)
        else:
            print(f"No .npz files in {API_RESULTS}")
    else:
        print(f"Results directory not found: {API_RESULTS}")

    if args.run:
        run_fresh_simulation()


if __name__ == "__main__":
    main()
