import math
from pathlib import Path

import h5py
import numpy as np

NATIVE_RESOLUTION = 2  # metres per voxel — matches HDF5 occupancy grid

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "presets"

def _read_domain_size(preset_id: str) -> list[int]:
    """Read domain_size from the HDF5 occupancy file for a preset."""
    h5_path = _ASSETS_DIR / f"{preset_id}.h5"
    if h5_path.exists():
        with h5py.File(str(h5_path), "r") as hf:
            return list(int(v) for v in hf["occupancy"].attrs["domain_size"])
    return [1000, 1000, 200]  # fallback

PRESETS: dict[str, dict] = {
    "chicago": {
        "id": "chicago",
        "name": "Chicago Loop",
        "description": "Dense urban canyon grid",
        "domainSize": _read_domain_size("chicago"),
        "resolution": NATIVE_RESOLUTION,
    },
    "manhattan": {
        "id": "manhattan",
        "name": "Manhattan Midtown",
        "description": "High-rise corridor with street canyons",
        "domainSize": _read_domain_size("manhattan"),
        "resolution": NATIVE_RESOLUTION,
    },
    "tokyo": {
        "id": "tokyo",
        "name": "Tokyo Shinjuku",
        "description": "Irregular dense urban fabric",
        "domainSize": _read_domain_size("tokyo"),
        "resolution": NATIVE_RESOLUTION,
    },
}


def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average a [Nz, Ny, Nx, 3] float32 array by `factor`."""
    Nz, Ny, Nx, C = arr.shape
    Nz_t = (Nz // factor) * factor
    Ny_t = (Ny // factor) * factor
    Nx_t = (Nx // factor) * factor
    trimmed = arr[:Nz_t, :Ny_t, :Nx_t, :]
    return (
        trimmed.reshape(
            Nz_t // factor, factor, Ny_t // factor, factor, Nx_t // factor, factor, C
        )
        .mean(axis=(1, 3, 5))
        .astype(np.float32)
    )


def _downsample_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average a [Ny, Nx] float32 array by `factor`."""
    Ny, Nx = arr.shape
    Ny_t = (Ny // factor) * factor
    Nx_t = (Nx // factor) * factor
    trimmed = arr[:Ny_t, :Nx_t]
    return (
        trimmed.reshape(Ny_t // factor, factor, Nx_t // factor, factor)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def generate(
    preset_id: str,
    direction: float,
    speed: float,
    roughness: float,
    out_resolution: float = 8.0,  # metres per cell in returned arrays
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Python port of makeFakeVelocityField + deriveComfortMap.

    Returns:
        vel:     float32 ndarray, shape [Nz, Ny, Nx, 3]
        comfort: float32 ndarray, shape [Ny, Nx]
        domain:  [domX, domY, domZ] metres
    """
    preset = PRESETS[preset_id]
    domain: list[int] = preset["domainSize"]
    # Generate directly at the requested resolution; the native 2 m grid is only
    # relevant for the real LBM solver (which reads the HDF5 occupancy grid).
    resolution: float = out_resolution

    Nx = round(domain[0] / resolution)
    Ny = round(domain[1] / resolution)
    Nz = round(domain[2] / resolution)

    # Index grids — shape broadcastable to [Nz, Ny, Nx]
    zs = np.arange(Nz, dtype=np.float64)
    ys = np.arange(Ny, dtype=np.float64)
    xs = np.arange(Nx, dtype=np.float64)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")  # [Nz, Ny, Nx]

    nx = X / (Nx - 1)
    ny_ = Y / (Ny - 1)
    nz_ = (Z + 1) / Nz

    rad = direction * math.pi / 180.0
    speed_factor = speed / 5.0

    # Log-law atmospheric boundary-layer height factor
    z_height = nz_ * 100.0
    height_factor = np.log(z_height / roughness) / np.log(10.0 / roughness)

    # ── Base flow (log-law profile) ───────────────────────────────────────
    vx_base = 5.0 * height_factor * speed_factor
    vy_base = np.zeros_like(vx_base)

    # ── Large-scale sinusoidal meandering ─────────────────────────────────
    meander = np.sin(ny_ * math.pi * 3 + nx * math.pi) * 0.8 * speed_factor
    deflect = np.cos(nx * math.pi * 2) * 0.4 * speed_factor

    # ── Street-canyon channelling ─────────────────────────────────────────
    BLOCK = max(2, round(80 / resolution))
    STREET = max(1, round(20 / resolution))
    Xi = X.astype(np.int64)
    Yi = Y.astype(np.int64)
    in_street_x = (Xi % BLOCK) < STREET
    in_street_y = (Yi % BLOCK) < STREET

    street_boost = np.where((in_street_x | in_street_y) & (nz_ < 0.25), 1.6, 1.0)
    canyon_vy = np.where(
        in_street_y & ~in_street_x & (nz_ < 0.3),
        np.sin(nx * math.pi * 6) * 1.2 * speed_factor,
        0.0,
    )

    # ── Corner vortex ─────────────────────────────────────────────────────
    cx, cy, cr = 0.6, 0.4, 0.12
    dx = nx - cx
    dy = ny_ - cy
    dist = np.sqrt(dx**2 + dy**2)
    vortex_str = np.maximum(0.0, 1.0 - dist / cr) * 2.5 * speed_factor * (1.0 - nz_)
    vortex_vx = dy * vortex_str
    vortex_vy = -dx * vortex_str

    # ── Thermal updraft ───────────────────────────────────────────────────
    tx = nx - 0.5
    ty = ny_ - 0.5
    thermal_r2 = tx**2 + ty**2
    updraft = np.exp(-thermal_r2 / 0.02) * 1.5 * speed_factor * (1.0 - nz_ * 1.5)

    # ── Combine ───────────────────────────────────────────────────────────
    vx = (vx_base + meander + vortex_vx) * street_boost
    vy = (vy_base + deflect + canyon_vy + vortex_vy) * street_boost
    vz = np.maximum(0.0, updraft)

    # Rotate entire field by wind direction
    rvx = vx * math.cos(rad) + vy * math.sin(rad)
    rvy = -vx * math.sin(rad) + vy * math.cos(rad)

    vel = np.stack([rvx, rvy, vz], axis=-1).astype(np.float32)
    # shape: [Nz, Ny, Nx, 3] — same byte layout as JS (z*Ny*Nx + y*Nx + x)*3

    # ── Comfort map: pedestrian level (z=0), speed magnitude ─────────────
    comfort_slice = vel[0, :, :, :2]  # [Ny, Nx, 2]
    comfort = np.sqrt(comfort_slice[..., 0] ** 2 + comfort_slice[..., 1] ** 2).astype(
        np.float32
    )
    # shape: [Ny, Nx]

    return vel, comfort, domain
