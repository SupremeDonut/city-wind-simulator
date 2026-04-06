"""Python wrapper that invokes the CUDA LBM solver via subprocess.

Handles HDF5 loading, preprocessing (rotation + downsampling),
binary stdin/stdout serialisation, and subprocess management.

Single-shot mode (default)
--------------------------
Each call to ``run_cuda_lbm`` spawns a fresh ``lbm_cuda.exe`` process,
pays the CUDA context init cost once, runs one simulation, and exits.

Persistent mode
---------------
``PersistentLBMWorker`` spawns ``lbm_cuda.exe --persistent`` once and reuses
it for many sequential simulations, amortising the CUDA context init cost
(~200-500 ms) and ``cudaMalloc`` overhead (~50 ms) across all calls.
Use this for batch data generation.  Thread-safe: one worker per thread.
"""

import os
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import h5py
import numpy as np

from lbm.preprocess import downsample_block_max, rotate_field_back, rotate_voxels


def _rotate_velocity_to_world(ux, uy, angle_deg):
    """Inverse-rotate velocity vectors from the rotated grid frame back to world frame.

    The solver always produces flow in the +Y grid direction.  After rotating
    voxels by `angle_deg`, we must rotate the (ux, uy) velocity components
    back by -angle_deg so the vectors align with the original world axes.
    """
    if angle_deg == 0.0:
        return ux, uy
    rad = -np.deg2rad(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    ux_w = ux * cos_a - uy * sin_a
    uy_w = ux * sin_a + uy * cos_a
    return ux_w, uy_w


# Repo root: lbm_py/lbm/ -> lbm_py/ -> simulation/ -> wind_sim/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SIM_SRC = _REPO_ROOT / "simulation" / "src"
_DEFAULT_EXE = _SIM_SRC / "build" / "Release" / "lbm_cuda.exe"
_EXE_PATH = Path(os.environ.get("LBM_EXE", str(_DEFAULT_EXE)))


def find_exe() -> Path:
    """Locate lbm_cuda.exe, checking LBM_EXE env var then common build paths."""
    if "LBM_EXE" in os.environ:
        p = Path(os.environ["LBM_EXE"])
        if p.exists():
            return p
    candidates = [
        _EXE_PATH,
        _SIM_SRC / "build" / "lbm_cuda.exe",
        _SIM_SRC / "build" / "Debug" / "lbm_cuda.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "lbm_cuda.exe not found. Searched:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\nBuild it first: cd simulation/src && cmake -B build && cmake --build build --config Release"
    )


def run_cuda_lbm(
    voxel_path: str,
    angle_deg: float = 0.0,
    wind_speed: float = 2.0,
    roughness: float = 0.5,
    pitch: float = 8.0,
    num_steps: int = 500,
    progress_callback=None,
    snapshot_interval: int = 200,
    snapshot_callback=None,
) -> dict:
    """Run the CUDA D3Q19 LBM solver.

    Parameters
    ----------
    voxel_path : str
        Path to HDF5 file with 'occupancy' dataset.
    angle_deg : float
        Wind direction in degrees (0=north, clockwise).
    wind_speed : float
        Reference wind speed at 10 m height (m/s).
    roughness : float
        Aerodynamic roughness length z0 (m).
    pitch : float
        Target grid spacing in metres (default 8 m).
    num_steps : int
        Number of LBM iterations.
    progress_callback : callable, optional
        Called with float 0..1 as simulation progresses.
    snapshot_interval : int
        Write a velocity snapshot every N steps (0 to disable).
    snapshot_callback : callable, optional
        Called with (step, ux, uy, uz) numpy arrays when a snapshot is written.

    Returns
    -------
    dict with keys: ux, uy, uz, solid, dx (numpy arrays)
    """
    exe = find_exe()

    # ── Load HDF5 ──────────────────────────────────────────────────────────
    with h5py.File(voxel_path, "r") as hf:
        occ = hf["occupancy"][:]
        domain_size = hf["occupancy"].attrs["domain_size"]

    native_pitch = domain_size[0] / occ.shape[2]  # metres per voxel
    domain_z = float(domain_size[2])
    factor = max(1, round(pitch / native_pitch))

    # Compute the target output shape (unrotated, downsampled) — this is what
    # the frontend expects (aligned with the city model).
    orig_NZ, orig_NY, orig_NX = occ.shape
    target_shape = (orig_NZ // factor, orig_NY // factor, orig_NX // factor)

    # ── Rotate for wind direction ──────────────────────────────────────────
    # The solver always pushes wind in +Y.  With no rotation that means
    # "wind from south" (compass 180°).  To get compass direction `angle_deg`
    # (0=north, meteorological), rotate voxels by (angle_deg + 180) % 360.
    voxel_rotation = (angle_deg + 180) % 360
    occ = rotate_voxels(occ, voxel_rotation)

    # ── Downsample ─────────────────────────────────────────────────────────
    occ = downsample_block_max(occ, factor)

    # Add ground plane
    occ[0, :, :] = 1

    # Scale domain_z to match downsampled grid
    actual_dz = domain_z / occ.shape[0]

    print(
        f"Preprocessed grid: {occ.shape[2]}x{occ.shape[1]}x{occ.shape[0]}, "
        f"dx={actual_dz:.1f}m, factor={factor}"
    )

    # ── Pack binary input ─────────────────────────────────────────────────
    # Header: NX, NY, NZ (i32), wind_speed, roughness, domain_z (f32),
    #         num_steps, snapshot_interval (i32) — 32 bytes little-endian
    NZ, NY, NX = occ.shape
    N = NX * NY * NZ
    header = struct.pack(
        "<iiifffii",
        NX,
        NY,
        NZ,
        float(wind_speed),
        float(roughness),
        float(domain_z),
        num_steps,
        snapshot_interval,
    )
    input_data = header + occ.astype(np.uint8).tobytes()

    # ── Run CUDA solver ────────────────────────────────────────────────────
    # snapshot_dir is only passed when snapshots are requested; use a
    # temporary directory so the files are cleaned up automatically.
    snapshot_dir_path = None
    tmpdir_ctx = tempfile.TemporaryDirectory() if snapshot_interval > 0 else None
    try:
        if tmpdir_ctx is not None:
            snapshot_dir_path = os.path.join(tmpdir_ctx.name, "snapshots")
            os.makedirs(snapshot_dir_path, exist_ok=True)

        cmd = [str(exe)]
        if snapshot_dir_path:
            cmd.append(snapshot_dir_path)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # ── Stream stderr for live progress/snapshot callbacks ─────────────
        lattice_to_phys = wind_speed / 0.06
        debug_lines: list[str] = []

        def _read_stderr():
            for raw in proc.stderr:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("PROGRESS"):
                    try:
                        val = float(line.split()[1])
                        if progress_callback:
                            progress_callback(val)
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("SNAPSHOT"):
                    if snapshot_callback and snapshot_dir_path:
                        try:
                            snap_filename = line.split()[1]
                            snap_path = os.path.join(snapshot_dir_path, snap_filename)
                            if os.path.exists(snap_path):
                                with np.load(snap_path) as snap_data:
                                    snap_step = int(snap_data["step"].flat[0])
                                    snap_ux = (
                                        np.array(snap_data["ux"]) * lattice_to_phys
                                    )
                                    snap_uy = (
                                        np.array(snap_data["uy"]) * lattice_to_phys
                                    )
                                    snap_uz = (
                                        np.array(snap_data["uz"]) * lattice_to_phys
                                    )
                                snap_ux = rotate_field_back(
                                    snap_ux, voxel_rotation, target_shape
                                )
                                snap_uy = rotate_field_back(
                                    snap_uy, voxel_rotation, target_shape
                                )
                                snap_uz = rotate_field_back(
                                    snap_uz, voxel_rotation, target_shape
                                )
                                snap_ux, snap_uy = _rotate_velocity_to_world(
                                    snap_ux, snap_uy, voxel_rotation
                                )
                                snapshot_callback(snap_step, snap_ux, snap_uy, snap_uz)
                        except Exception:
                            pass
                else:
                    debug_lines.append(line)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        # Write all input then close stdin so the solver can start.
        # stdout is read only after the solver exits — no deadlock risk because
        # stdin is fully consumed before the solver writes any output.
        try:
            proc.stdin.write(input_data)
            proc.stdin.close()
        except BrokenPipeError:
            proc.wait()
            stderr_thread.join()
            stderr_out = "\n".join(debug_lines)
            raise RuntimeError(
                f"lbm_cuda.exe died before reading input (code {proc.returncode}).\n"
                f"stderr: {stderr_out}"
            )
        stdout_bytes = proc.stdout.read()
        proc.wait()
        stderr_thread.join()

        if debug_lines:
            print("\n".join(debug_lines), file=sys.stderr)

        if proc.returncode != 0:
            stderr_out = "\n".join(debug_lines)
            raise RuntimeError(
                f"lbm_cuda.exe failed (code {proc.returncode}).\nstderr: {stderr_out}"
            )

        # ── Parse binary output ────────────────────────────────────────────
        # Layout: [dx: f32] [ux: N×f32] [uy: N×f32] [uz: N×f32] [solid: N×u8]
        expected = 4 + N * (3 * 4 + 1)
        if len(stdout_bytes) != expected:
            raise RuntimeError(
                f"lbm_cuda.exe output size mismatch: got {len(stdout_bytes)}, expected {expected}"
            )

        dx_val = struct.unpack_from("<f", stdout_bytes, 0)[0]
        off = 4
        ux = (
            np.frombuffer(stdout_bytes, dtype=np.float32, count=N, offset=off)
            .reshape(NZ, NY, NX)
            .copy()
        )
        off += N * 4
        uy = (
            np.frombuffer(stdout_bytes, dtype=np.float32, count=N, offset=off)
            .reshape(NZ, NY, NX)
            .copy()
        )
        off += N * 4
        uz = (
            np.frombuffer(stdout_bytes, dtype=np.float32, count=N, offset=off)
            .reshape(NZ, NY, NX)
            .copy()
        )
        off += N * 4
        solid = (
            np.frombuffer(stdout_bytes, dtype=np.uint8, count=N, offset=off)
            .reshape(NZ, NY, NX)
            .copy()
        )

        # Convert lattice → physical units
        ux *= lattice_to_phys
        uy *= lattice_to_phys
        uz *= lattice_to_phys

    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()

    # Spatially rotate fields back to the original (unrotated) frame
    # and crop to the target shape so they align with the city model.
    ux = rotate_field_back(ux, voxel_rotation, target_shape)
    uy = rotate_field_back(uy, voxel_rotation, target_shape)
    uz = rotate_field_back(uz, voxel_rotation, target_shape)

    # Rotate velocity vectors back to world frame
    ux, uy = _rotate_velocity_to_world(ux, uy, voxel_rotation)

    return {"ux": ux, "uy": uy, "uz": uz, "solid": solid, "dx": dx_val}


def run_cuda_simulation(
    preset_id: str,
    direction: float,
    speed: float,
    roughness: float,
    resolution: float = 8.0,
    num_steps: int = 500,
    progress_callback=None,
    snapshot_interval: int = 200,
    snapshot_callback=None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """API-compatible wrapper: returns (vel, comfort, domain).

    Matches the signature expected by api/routes/simulate.py.
    """
    assets_dir = _REPO_ROOT / "assets" / "presets"
    voxel_path = str(assets_dir / f"{preset_id}.h5")

    # Wrap snapshot_callback to convert raw arrays into (vel, comfort) format
    wrapped_snap_cb = None
    if snapshot_callback:

        def wrapped_snap_cb(step, ux, uy, uz):
            vel = np.stack([ux, uy, uz], axis=-1).astype(np.float32)
            comfort = np.sqrt(ux[1] ** 2 + uy[1] ** 2).astype(np.float32)
            snapshot_callback(step, vel, comfort)

    result = run_cuda_lbm(
        voxel_path=voxel_path,
        angle_deg=direction,
        wind_speed=speed,
        roughness=roughness,
        pitch=resolution,
        num_steps=num_steps,
        progress_callback=progress_callback,
        snapshot_interval=snapshot_interval,
        snapshot_callback=wrapped_snap_cb,
    )

    ux = result["ux"]
    uy = result["uy"]
    uz = result["uz"]

    # Stack into [NZ, NY, NX, 3] velocity field
    vel = np.stack([ux, uy, uz], axis=-1).astype(np.float32)

    # Comfort map: horizontal wind speed at z=1 (first level above ground)
    comfort = np.sqrt(ux[1] ** 2 + uy[1] ** 2).astype(np.float32)

    # Domain size: XY from HDF5, Z from the (possibly extended) output grid
    with h5py.File(voxel_path, "r") as hf:
        domain = list(hf["occupancy"].attrs["domain_size"])
    # Z domain = NZ_output * dx, where dx = result["dx"]
    domain[2] = float(vel.shape[0] * result["dx"])

    return vel, comfort, domain


# ── Persistent worker ─────────────────────────────────────────────────────


def _read_exact(pipe, n: int) -> bytes:
    """Read exactly n bytes from a pipe, raising RuntimeError on short read."""
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            raise RuntimeError(
                f"lbm_cuda.exe stdout closed unexpectedly after {len(buf)}/{n} bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


class PersistentLBMWorker:
    """Long-lived lbm_cuda.exe subprocess that handles multiple simulations.

    Spawns the solver once with ``--persistent``.  The CUDA context and device
    memory allocations are reused across calls as long as grid dimensions stay
    the same (always true within one preset during data generation).

    Not thread-safe — create one instance per thread.

    Usage::

        worker = PersistentLBMWorker()
        result = worker.run_lbm(voxel_path, angle_deg=45, wind_speed=5, ...)
        result2 = worker.run_lbm(voxel_path, angle_deg=90, wind_speed=3, ...)
        worker.close()

    Or use as a context manager::

        with PersistentLBMWorker() as worker:
            result = worker.run_lbm(voxel_path, ...)
    """

    def __init__(self):
        exe = find_exe()
        self._proc = subprocess.Popen(
            [str(exe), "--persistent"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stderr_lines: list[str] = []
        self._progress_callback = None
        self._lock = threading.Lock()  # serialise calls on this worker

        # Drain stderr in a background daemon thread
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self):
        for raw in self._proc.stderr:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("PROGRESS"):
                try:
                    val = float(line.split()[1])
                    cb = self._progress_callback
                    if cb:
                        cb(val)
                except (IndexError, ValueError):
                    pass
            else:
                self._stderr_lines.append(line)

    def run_lbm(
        self,
        voxel_path: str,
        angle_deg: float = 0.0,
        wind_speed: float = 2.0,
        roughness: float = 0.5,
        pitch: float = 8.0,
        num_steps: int = 500,
        progress_callback=None,
        snapshot_interval: int = 0,
        snapshot_callback=None,
    ) -> dict:
        """Run one LBM simulation on the persistent worker.

        Has the same signature and return value as ``run_cuda_lbm``.
        Snapshots are not supported in persistent mode (snapshot_interval is ignored).
        """
        # Preprocessing (rotation + downsampling) — same as run_cuda_lbm
        with h5py.File(voxel_path, "r") as hf:
            occ = hf["occupancy"][:]
            domain_size = hf["occupancy"].attrs["domain_size"]

        native_pitch = domain_size[0] / occ.shape[2]
        domain_z = float(domain_size[2])
        factor = max(1, round(pitch / native_pitch))

        orig_NZ, orig_NY, orig_NX = occ.shape
        target_shape = (orig_NZ // factor, orig_NY // factor, orig_NX // factor)

        voxel_rotation = (angle_deg + 180) % 360
        occ = rotate_voxels(occ, voxel_rotation)
        occ = downsample_block_max(occ, factor)
        occ[0, :, :] = 1  # ground plane

        NZ, NY, NX = occ.shape
        N = NX * NY * NZ

        header = struct.pack(
            "<iiifffii",
            NX, NY, NZ,
            float(wind_speed), float(roughness), float(domain_z),
            num_steps,
            0,  # snapshot_interval disabled in persistent mode
        )
        payload = header + occ.astype(np.uint8).tobytes()

        with self._lock:
            self._progress_callback = progress_callback
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
            except BrokenPipeError:
                stderr_out = "\n".join(self._stderr_lines)
                raise RuntimeError(
                    f"lbm_cuda.exe worker died (code {self._proc.returncode}).\n"
                    f"stderr: {stderr_out}"
                )

            # Read exactly the expected output bytes
            expected = 4 + N * (3 * 4 + 1)
            stdout_bytes = _read_exact(self._proc.stdout, expected)
            self._progress_callback = None

        # Parse output — identical to run_cuda_lbm
        lattice_to_phys = wind_speed / 0.06
        dx_val = struct.unpack_from("<f", stdout_bytes, 0)[0]
        off = 4
        ux = np.frombuffer(stdout_bytes, dtype=np.float32, count=N, offset=off).reshape(NZ, NY, NX).copy()
        off += N * 4
        uy = np.frombuffer(stdout_bytes, dtype=np.float32, count=N, offset=off).reshape(NZ, NY, NX).copy()
        off += N * 4
        uz = np.frombuffer(stdout_bytes, dtype=np.float32, count=N, offset=off).reshape(NZ, NY, NX).copy()
        off += N * 4
        solid = np.frombuffer(stdout_bytes, dtype=np.uint8, count=N, offset=off).reshape(NZ, NY, NX).copy()

        ux *= lattice_to_phys
        uy *= lattice_to_phys
        uz *= lattice_to_phys

        ux = rotate_field_back(ux, voxel_rotation, target_shape)
        uy = rotate_field_back(uy, voxel_rotation, target_shape)
        uz = rotate_field_back(uz, voxel_rotation, target_shape)
        ux, uy = _rotate_velocity_to_world(ux, uy, voxel_rotation)

        return {"ux": ux, "uy": uy, "uz": uz, "solid": solid, "dx": dx_val}

    def run_lbm_preloaded(
        self,
        occ: np.ndarray,
        domain_z: float,
        voxel_rotation: float,
        target_shape: tuple[int, int, int],
        wind_speed: float = 2.0,
        roughness: float = 0.5,
        num_steps: int = 500,
        progress_callback=None,
    ) -> dict:
        """Run one LBM simulation with a pre-processed occupancy array.

        Skips H5 file I/O, voxel rotation, and downsampling — use this when
        the caller has already prepared the occupancy grid (e.g. data generation
        where the same base geometry is reused across many parameter combinations).

        Args:
            occ:            Pre-rotated, downsampled occupancy [NZ, NY, NX] uint8.
            domain_z:       Physical domain height in metres.
            voxel_rotation: The rotation angle applied to occ (degrees), needed to
                            rotate the velocity field back to world frame.
            target_shape:   (NZ, NY, NX) of the *un-rotated* output grid for cropping.
            wind_speed:     Reference wind speed at 10 m height (m/s).
            roughness:      Aerodynamic roughness length z0 (m).
            num_steps:      Number of LBM iterations.
            progress_callback: Optional callable(float 0..1).
        """
        NZ, NY, NX = occ.shape
        N = NX * NY * NZ

        header = struct.pack(
            "<iiifffii",
            NX, NY, NZ,
            float(wind_speed), float(roughness), float(domain_z),
            num_steps, 0,
        )
        payload = header + occ.astype(np.uint8).tobytes()

        with self._lock:
            self._progress_callback = progress_callback
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
            except BrokenPipeError:
                stderr_out = "\n".join(self._stderr_lines)
                raise RuntimeError(
                    f"lbm_cuda.exe worker died (code {self._proc.returncode}).\n"
                    f"stderr: {stderr_out}"
                )
            expected = 4 + N * (3 * 4 + 1)
            stdout_bytes = _read_exact(self._proc.stdout, expected)
            self._progress_callback = None

        lattice_to_phys = wind_speed / 0.06
        dx_val = struct.unpack_from("<f", stdout_bytes, 0)[0]
        off = 4
        ux = np.frombuffer(stdout_bytes, dtype=np.float32, count=N, offset=off).reshape(NZ, NY, NX).copy(); off += N * 4
        uy = np.frombuffer(stdout_bytes, dtype=np.float32, count=N, offset=off).reshape(NZ, NY, NX).copy(); off += N * 4
        uz = np.frombuffer(stdout_bytes, dtype=np.float32, count=N, offset=off).reshape(NZ, NY, NX).copy(); off += N * 4
        solid = np.frombuffer(stdout_bytes, dtype=np.uint8, count=N, offset=off).reshape(NZ, NY, NX).copy()

        ux *= lattice_to_phys
        uy *= lattice_to_phys
        uz *= lattice_to_phys

        ux = rotate_field_back(ux, voxel_rotation, target_shape)
        uy = rotate_field_back(uy, voxel_rotation, target_shape)
        uz = rotate_field_back(uz, voxel_rotation, target_shape)
        ux, uy = _rotate_velocity_to_world(ux, uy, voxel_rotation)

        return {"ux": ux, "uy": uy, "uz": uz, "solid": solid, "dx": dx_val}

    def close(self, timeout: float = 10.0):
        """Send the shutdown sentinel (NX=0 header) and wait for clean exit."""
        try:
            sentinel = struct.pack("<iiifffii", 0, 0, 0, 0.0, 0.0, 0.0, 0, 0)
            self._proc.stdin.write(sentinel)
            self._proc.stdin.flush()
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def kill(self):
        """Immediately terminate the worker process."""
        self._proc.kill()

    @property
    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CUDA D3Q19 LBM solver")
    parser.add_argument("voxel_path", help="Path to HDF5 occupancy file")
    parser.add_argument(
        "--angle", type=float, default=0.0, help="Wind direction (degrees)"
    )
    parser.add_argument(
        "--speed", type=float, default=2.0, help="Wind speed at 10m (m/s)"
    )
    parser.add_argument(
        "--roughness", type=float, default=0.5, help="Roughness length z0 (m)"
    )
    parser.add_argument(
        "--pitch", type=float, default=8.0, help="Target grid pitch (m)"
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of LBM steps")
    args = parser.parse_args()

    def print_progress(val):
        print(f"\r  Simulation: {val * 100:.0f}%", end="", flush=True)

    result = run_cuda_lbm(
        args.voxel_path,
        angle_deg=args.angle,
        wind_speed=args.speed,
        roughness=args.roughness,
        pitch=args.pitch,
        num_steps=args.steps,
        progress_callback=print_progress,
    )
    print()

    ux, uy, uz = result["ux"], result["uy"], result["uz"]
    speed = np.sqrt(ux**2 + uy**2 + uz**2)
    print(f"Max speed: {speed.max():.4f} lattice units")
    print(f"NaN count: {np.isnan(speed).sum()}")

    # Save a copy locally
    np.savez("cuda_wind_field.npz", **result)
    print("Saved cuda_wind_field.npz")
