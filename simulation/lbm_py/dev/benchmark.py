"""Benchmark: CUDA vs Python D3Q19 LBM solver.

Runs both solvers on the same preprocessed input and compares
wall-clock time, throughput (MLUPS), and numerical agreement.
"""

import time
from pathlib import Path

import h5py
import numpy as np

from d3q19 import (
    CS2,
    TAU,
    EX,
    EY,
    EZ,
    W,
    OPP,
    apply_inlet,
    apply_outlet,
    equilibrium,
    log_wind_profile,
    stream_and_bounce,
)
from lbm.preprocess import downsample_block_max
from lbm.run_cuda import run_cuda_lbm


def run_python_lbm(solid, domain_z, wind_speed, roughness, num_steps):
    """Run the Python D3Q19 solver (no visualisation) and return ux, uy, uz."""
    NZ, NY, NX = solid.shape
    dx = domain_z / NZ

    u_ref_lattice = 0.06
    z0_lattice = roughness / dx
    z_ref_lattice = 10.0 / dx

    u_profile = np.array([
        log_wind_profile(k + 0.5, z_ref_lattice, u_ref_lattice, z0_lattice)
        for k in range(NZ)
    ])

    rho = np.ones((NZ, NY, NX))
    ux = np.zeros((NZ, NY, NX))
    uy = np.tile(u_profile[:, None, None], (1, NY, NX))
    uz = np.zeros((NZ, NY, NX))
    uy[solid] = 0.0

    f = equilibrium(rho, ux, uy, uz)

    for step in range(num_steps):
        feq = equilibrium(rho, ux, uy, uz)
        f += (feq - f) / TAU
        f[:, solid] = f[OPP][:, solid]

        f = stream_and_bounce(f, solid)
        apply_inlet(f, u_profile, NZ, NX)
        apply_outlet(f)

        rho = f.sum(axis=0)
        ux = (f * EX[:, None, None, None]).sum(axis=0) / rho
        uy = (f * EY[:, None, None, None]).sum(axis=0) / rho
        uz = (f * EZ[:, None, None, None]).sum(axis=0) / rho
        ux[solid] = 0.0
        uy[solid] = 0.0
        uz[solid] = 0.0

        if (step + 1) % 100 == 0:
            speed = np.sqrt(ux**2 + uy**2 + uz**2)
            print(f"  Python step {step+1}/{num_steps}, max speed: {speed.max():.4f}")

    return ux, uy, uz


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark CUDA vs Python LBM")
    parser.add_argument("--preset", default="chicago", help="Preset name")
    parser.add_argument("--pitch", type=float, default=8.0, help="Grid pitch (m)")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps")
    parser.add_argument("--speed", type=float, default=2.0, help="Wind speed (m/s)")
    parser.add_argument("--roughness", type=float, default=0.5, help="Roughness z0 (m)")
    args = parser.parse_args()

    # ── Load and preprocess ────────────────────────────────────────────────
    assets_dir = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "presets"
    h5_path = str(assets_dir / f"{args.preset}.h5")

    with h5py.File(h5_path, "r") as hf:
        occ = hf["occupancy"][:]
        domain_size = hf["occupancy"].attrs["domain_size"]

    native_pitch = domain_size[0] / occ.shape[2]
    factor = max(1, round(args.pitch / native_pitch))
    occ = downsample_block_max(occ, factor)
    occ[0, :, :] = 1  # ground plane

    solid = occ > 0
    NZ, NY, NX = solid.shape
    N = NX * NY * NZ
    domain_z = float(domain_size[2])

    print(f"Grid: {NX}x{NY}x{NZ} = {N:,} cells")
    print(f"Pitch: {domain_z/NZ:.1f}m, Steps: {args.steps}")
    print()

    # ── Run Python solver ──────────────────────────────────────────────────
    print("Running Python solver...")
    t0 = time.perf_counter()
    py_ux, py_uy, py_uz = run_python_lbm(
        solid, domain_z, args.speed, args.roughness, args.steps
    )
    t_py = time.perf_counter() - t0
    mlups_py = N * args.steps / t_py / 1e6

    # ── Run CUDA solver ────────────────────────────────────────────────────
    print("\nRunning CUDA solver...")
    t0 = time.perf_counter()
    result = run_cuda_lbm(
        h5_path,
        angle_deg=0.0,
        wind_speed=args.speed,
        roughness=args.roughness,
        pitch=args.pitch,
        num_steps=args.steps,
    )
    t_cuda = time.perf_counter() - t0
    mlups_cuda = N * args.steps / t_cuda / 1e6

    cu_ux, cu_uy, cu_uz = result["ux"], result["uy"], result["uz"]

    # ── Compare ────────────────────────────────────────────────────────────
    diff_ux = np.max(np.abs(cu_ux - py_ux))
    diff_uy = np.max(np.abs(cu_uy - py_uy))
    diff_uz = np.max(np.abs(cu_uz - py_uz))

    speedup = t_py / t_cuda if t_cuda > 0 else float("inf")

    print("\n" + "=" * 60)
    print(f"{'Benchmark Results':^60}")
    print("=" * 60)
    print(f"{'':20} {'Python':>15} {'CUDA':>15}")
    print("-" * 60)
    print(f"{'Wall clock (s)':20} {t_py:15.2f} {t_cuda:15.2f}")
    print(f"{'MLUPS':20} {mlups_py:15.2f} {mlups_cuda:15.2f}")
    print(f"{'Max |ux|':20} {np.max(np.abs(py_ux)):15.6f} {np.max(np.abs(cu_ux)):15.6f}")
    print(f"{'Max |uy|':20} {np.max(np.abs(py_uy)):15.6f} {np.max(np.abs(cu_uy)):15.6f}")
    print(f"{'Max |uz|':20} {np.max(np.abs(py_uz)):15.6f} {np.max(np.abs(cu_uz)):15.6f}")
    print("-" * 60)
    print(f"{'Speedup':20} {speedup:15.1f}x")
    print(f"{'Max |diff ux|':20} {diff_ux:15.2e}")
    print(f"{'Max |diff uy|':20} {diff_uy:15.2e}")
    print(f"{'Max |diff uz|':20} {diff_uz:15.2e}")
    print("=" * 60)

    if max(diff_ux, diff_uy, diff_uz) < 1e-4:
        print("PASS: Numerical agreement within 1e-4")
    else:
        print("WARNING: Numerical difference exceeds 1e-4")
        print("  (Some difference is expected due to float32 vs float64 and operation order)")


if __name__ == "__main__":
    main()
