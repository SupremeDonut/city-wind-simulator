"""Batch data generation pipeline: run LBM solver over parameter sweeps and store results.

Usage:
    cd surrogate
    uv run python -m surrogate.data.generate [--preset chicago] [--workers 4] [--dry-run]

How parallelism works
---------------------
Each worker thread owns one ``PersistentLBMWorker`` — a long-lived lbm_cuda.exe
subprocess spawned with ``--persistent``.  The CUDA context and device allocations
inside that process are reused for every simulation call, so the ~300-500 ms
per-run overhead (CUDA init + cudaMalloc) is paid only once per worker thread.

With --workers 4 you get 4 concurrent lbm_cuda.exe processes each running at
~full GPU utilisation, saturating the RTX 4070.  Increase until GPU memory runs
out (each worker at 8 m pitch uses ~30 MB device memory).

Ctrl+C / kill
-------------
A signal handler kills all worker subprocesses immediately and exits.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import signal
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from surrogate.config import ASSETS_DIR, CanonicalGrid, DataGenConfig


# ── Global worker registry for clean shutdown ──────────────────────────────

_all_workers: list = []  # PersistentLBMWorker instances
_registry_lock = threading.Lock()
_shutting_down = False
_executor: ThreadPoolExecutor | None = None  # set before submitting futures


def _shutdown_all(signum=None, frame=None):
    """Kill all live worker subprocesses and exit immediately.

    Uses os._exit() rather than sys.exit() to bypass Python's cleanup phase,
    which would otherwise block in ThreadPoolExecutor.shutdown(wait=True)
    waiting for all worker threads to finish their current task.
    """
    global _shutting_down
    _shutting_down = True

    # Cancel any futures that haven't started yet
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)

    with _registry_lock:
        workers = list(_all_workers)

    if workers:
        print(
            f"\nInterrupted — killing {len(workers)} worker process(es)...", flush=True
        )
        for w in workers:
            try:
                w.kill()
            except Exception:
                pass

    # os._exit skips all Python cleanup (atexit, __del__, thread joins, etc.)
    # so the process terminates immediately after the subprocesses are killed.
    os._exit(1)


signal.signal(signal.SIGINT, _shutdown_all)
signal.signal(signal.SIGTERM, _shutdown_all)


# ── Thread-local persistent worker ────────────────────────────────────────

_thread_local = threading.local()


def _get_worker() -> "PersistentLBMWorker":  # noqa: F821
    """Return the persistent worker for this thread, creating it if needed."""
    from lbm.run_cuda import PersistentLBMWorker

    if not getattr(_thread_local, "worker", None) or not _thread_local.worker.is_alive:
        w = PersistentLBMWorker()
        _thread_local.worker = w
        with _registry_lock:
            _all_workers.append(w)
    return _thread_local.worker


# ── Padding helper ─────────────────────────────────────────────────────────


def _fit_to_canonical(
    arr: np.ndarray, grid: CanonicalGrid, fill: float = 0.0
) -> np.ndarray:
    """Clip or pad a 3D array [D, H, W, ...] to exactly the canonical grid dimensions.

    Dimensions larger than canonical are truncated (upper Z layers are atmospheric
    free-stream and contribute nothing to pedestrian-level predictions).
    Dimensions smaller than canonical are zero-padded at the end.
    """
    target = (grid.depth, grid.height, grid.width)

    # Clip excess on each spatial axis first
    slices = tuple(slice(0, min(arr.shape[i], target[i])) for i in range(3))
    slices += (slice(None),) * (arr.ndim - 3)  # preserve trailing dims (e.g. channels)
    arr = arr[slices]

    # Pad any remaining shortfall
    pad_widths = [(0, target[i] - arr.shape[i]) for i in range(3)]
    pad_widths += [(0, 0)] * (arr.ndim - 3)
    return np.pad(arr, pad_widths, mode="constant", constant_values=fill)


# Keep old name as alias so any other callers aren't broken
_pad_to_canonical = _fit_to_canonical


def _load_and_prepare_occupancy(
    h5_path: Path, domain_size: list, pitch: float
) -> tuple[np.ndarray, float, int]:
    """Load the HDF5 occupancy once and downsample to working resolution.

    Returns:
        base_occ_ds: Downsampled occupancy [NZ, NY, NX] uint8 (no ground plane yet,
                     so it can be rotated first then the ground plane added).
        domain_z:    Physical domain height in metres.
        factor:      Downsample factor (native_pitch -> pitch).
    """
    from lbm.preprocess import downsample_block_max

    with h5py.File(str(h5_path), "r") as hf:
        base_occ = hf["occupancy"][:]

    native_pitch = domain_size[0] / base_occ.shape[2]
    domain_z = float(domain_size[2])
    factor = max(1, round(pitch / native_pitch))
    base_occ = downsample_block_max(base_occ, factor)
    return base_occ, domain_z, factor


def _build_rotated_occ_cache(
    base_occ_ds: np.ndarray, directions: list[float]
) -> dict[float, tuple[np.ndarray, tuple, float]]:
    """Pre-rotate and cache the downsampled occupancy for every wind direction.

    This is the expensive step (scipy ndimage.rotate on each angle) but it only
    runs once per preset rather than once per sample.

    Returns:
        dict mapping direction -> (rotated_occ [NZ,NY,NX] uint8, target_shape, voxel_rotation)
    """
    from lbm.preprocess import rotate_voxels

    orig_NZ, orig_NY, orig_NX = base_occ_ds.shape
    target_shape = (orig_NZ, orig_NY, orig_NX)  # output shape after rotating back
    cache = {}
    for direction in directions:
        voxel_rotation = (direction + 180) % 360
        occ_rot = rotate_voxels(base_occ_ds.copy(), voxel_rotation)
        occ_rot[0, :, :] = 1  # ground plane
        cache[direction] = (occ_rot.astype(np.uint8), target_shape, voxel_rotation)
    return cache


# ── Per-sample worker function ─────────────────────────────────────────────


def _run_one(
    direction: float,
    speed: float,
    roughness: float,
    occ: np.ndarray,
    domain_z: float,
    target_shape: tuple,
    voxel_rotation: float,
    num_steps: int,
    grid: CanonicalGrid,
) -> np.ndarray:
    """Run one simulation using this thread's persistent worker.

    Accepts a pre-rotated occupancy array — no H5 I/O or scipy rotation per call.
    Returns the padded velocity array [D, H, W, 3].
    """
    worker = _get_worker()

    result = worker.run_lbm_preloaded(
        occ=occ,
        domain_z=domain_z,
        voxel_rotation=voxel_rotation,
        target_shape=target_shape,
        wind_speed=speed,
        roughness=roughness,
        num_steps=num_steps,
    )

    vel = np.stack([result["ux"], result["uy"], result["uz"]], axis=-1).astype(
        np.float16
    )
    return _pad_to_canonical(vel, grid, fill=0.0)


# ── Main dataset generation function ──────────────────────────────────────


def generate_dataset(
    preset_id: str,
    cfg: DataGenConfig | None = None,
    dry_run: bool = False,
    workers: int = 4,
) -> Path:
    """Generate training data for a single preset geometry.

    Uses ``workers`` persistent lbm_cuda.exe processes running concurrently.
    Each process holds its CUDA context open across all its assigned samples,
    paying init overhead only once per worker (not once per sample).
    """
    if cfg is None:
        cfg = DataGenConfig()
    grid = cfg.grid

    out_dir = (
        cfg.data_dir
        if hasattr(cfg, "data_dir")
        else Path(__file__).resolve().parent.parent.parent / "data"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{preset_id}.h5"

    h5_path = ASSETS_DIR / f"{preset_id}.h5"
    if not h5_path.exists():
        print(f"WARNING: {h5_path} not found, skipping {preset_id}")
        return out_path

    with h5py.File(str(h5_path), "r") as hf:
        domain_size = list(hf["occupancy"].attrs["domain_size"])

    directions = [i * (360.0 / cfg.n_directions) for i in range(cfg.n_directions)]
    combos = list(itertools.product(directions, cfg.speeds, cfg.roughnesses))
    n_samples = len(combos)

    print(f"\n{'='*60}")
    print(f"Generating {n_samples} samples for preset '{preset_id}'")
    print(f"  Directions:  {len(directions)} ({cfg.n_directions} steps)")
    print(f"  Speeds:      {cfg.speeds}")
    print(f"  Roughnesses: {cfg.roughnesses}")
    print(f"  Grid:        {grid.depth}x{grid.height}x{grid.width}")
    print(f"  Workers:     {workers} persistent lbm_cuda.exe processes")
    print(f"  Output:      {out_path}")
    print(f"{'='*60}")

    if dry_run:
        print("DRY RUN — would generate the above. Exiting.")
        return out_path

    # ── Resume: check existing file for shape compatibility ────────────────
    completed: set[int] = set()
    if out_path.exists():
        with h5py.File(str(out_path), "r") as hf:
            if "velocity" in hf:
                saved_shape = list(hf.attrs.get("canonical_shape", [0, 0, 0]))
                current_shape = [grid.depth, grid.height, grid.width]
                if saved_shape != current_shape:
                    raise ValueError(
                        f"Existing data at '{out_path}' has canonical shape {saved_shape} "
                        f"but the current config specifies {current_shape}.\n"
                        f"Delete the file and re-run to regenerate with the new grid size:\n"
                        f"  del {out_path}"
                    )
                if "completed_indices" in hf:
                    completed = set(hf["completed_indices"][:].tolist())
                else:
                    n_written = int(hf["velocity"].attrs.get("n_written", 0))
                    completed = set(range(n_written))
                if completed:
                    print(f"Resuming: {len(completed)}/{n_samples} already done")

    pending = [(i, combos[i]) for i in range(n_samples) if i not in completed]
    if not pending:
        print("All samples already generated.")
        return out_path

    # ── Pre-compute everything that doesn't depend on speed/roughness ─────
    # Load H5 once, downsample once, then rotate for each unique direction.
    # This avoids re-reading the file and re-running scipy rotate per sample.
    print("Loading and downsampling occupancy grid...", end=" ", flush=True)
    base_occ_ds, domain_z, _ = _load_and_prepare_occupancy(
        h5_path, domain_size, cfg.pitch
    )
    print("done")

    # Only rotate for directions that still have pending samples
    pending_directions = {combos[i][0] for i, _ in pending}
    print(
        f"Pre-rotating occupancy for {len(pending_directions)} unique directions...",
        end=" ",
        flush=True,
    )
    occ_cache = _build_rotated_occ_cache(base_occ_ds, list(pending_directions))
    print("done")

    # Also keep the un-rotated padded occupancy for storing as model input
    base_occ_ds[0, :, :] = 1  # ground plane (for storage only)
    base_occ_padded = _pad_to_canonical(base_occ_ds[..., np.newaxis], grid, fill=0.0)[
        ..., 0
    ].astype(np.uint8)

    # ── Create HDF5 datasets ───────────────────────────────────────────────
    write_lock = threading.Lock()

    with h5py.File(str(out_path), "a") as hf:
        if "occupancy" not in hf:
            hf.create_dataset(
                "occupancy",
                shape=(n_samples, grid.depth, grid.height, grid.width),
                dtype=np.uint8,
                chunks=(1, grid.depth, grid.height, grid.width),
                compression="gzip",
                compression_opts=4,
            )
        if "velocity" not in hf:
            hf.create_dataset(
                "velocity",
                shape=(n_samples, grid.depth, grid.height, grid.width, 3),
                dtype=np.float16,
                chunks=(1, grid.depth, grid.height, grid.width, 3),
                compression="gzip",
                compression_opts=4,
            )
        if "params" not in hf:
            hf.create_dataset("params", shape=(n_samples, 3), dtype=np.float32)
        if "completed_indices" not in hf:
            hf.create_dataset(
                "completed_indices",
                shape=(len(completed),),
                maxshape=(None,),
                dtype=np.int32,
                data=(
                    np.array(sorted(completed), dtype=np.int32)
                    if completed
                    else np.empty(0, dtype=np.int32)
                ),
            )

        hf.attrs["preset_id"] = preset_id
        hf.attrs["domain_size"] = domain_size
        hf.attrs["pitch"] = cfg.pitch
        hf.attrs["canonical_shape"] = [grid.depth, grid.height, grid.width]
        hf.attrs["num_steps"] = cfg.num_steps

        ds_occ = hf["occupancy"]
        ds_vel = hf["velocity"]
        ds_params = hf["params"]
        ds_done = hf["completed_indices"]

        # ── Submit all pending jobs ────────────────────────────────────────
        pbar = tqdm(total=n_samples, initial=len(completed), desc=preset_id, unit="sim")
        error_count = 0

        global _executor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            _executor = pool
            future_to_idx: dict[Future, int] = {}
            for sample_idx, (direction, speed, roughness) in pending:
                occ, target_shape, voxel_rotation = occ_cache[direction]
                fut = pool.submit(
                    _run_one,
                    direction,
                    speed,
                    roughness,
                    occ,
                    domain_z,
                    target_shape,
                    voxel_rotation,
                    cfg.num_steps,
                    grid,
                )
                future_to_idx[fut] = sample_idx

            for fut in as_completed(future_to_idx):
                if _shutting_down:
                    break

                sample_idx = future_to_idx[fut]
                direction, speed, roughness = combos[sample_idx]

                try:
                    vel_padded = fut.result()
                except Exception as e:
                    error_count += 1
                    pbar.write(
                        f"ERROR sample {sample_idx} "
                        f"(dir={direction:.0f}, spd={speed:.1f}, z0={roughness:.3f}): {e}"
                    )
                    pbar.update(1)
                    continue

                with write_lock:
                    ds_occ[sample_idx] = base_occ_padded
                    ds_vel[sample_idx] = vel_padded
                    ds_params[sample_idx] = [math.radians(direction), speed, roughness]
                    n_done = ds_done.shape[0]
                    ds_done.resize((n_done + 1,))
                    ds_done[n_done] = sample_idx
                    hf.flush()

                pbar.set_postfix(
                    dir=f"{direction:.0f}",
                    spd=f"{speed:.1f}",
                    z0=f"{roughness:.3f}",
                    err=error_count or None,
                )
                pbar.update(1)

        pbar.close()
        _executor = None

    # Close persistent workers for this preset run
    with _registry_lock:
        for w in _all_workers:
            try:
                w.close()
            except Exception:
                pass
        _all_workers.clear()

    total_done = len(completed) + len(pending) - error_count
    print(
        f"\nDone: {out_path} ({total_done}/{n_samples} samples, {error_count} errors)"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate LBM training data for FNO surrogate"
    )
    parser.add_argument(
        "--preset", type=str, default=None, help="Single preset (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without running LBM"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="LBM iterations per sample"
    )
    parser.add_argument("--pitch", type=float, default=8.0, help="Grid pitch in metres")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Persistent worker processes (default 4). "
        "Raise until GPU stays saturated; lower if CUDA OOM.",
    )
    args = parser.parse_args()

    cfg = DataGenConfig(num_steps=args.steps, pitch=args.pitch)

    presets = [args.preset] if args.preset else cfg.preset_ids
    for pid in presets:
        generate_dataset(pid, cfg=cfg, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
