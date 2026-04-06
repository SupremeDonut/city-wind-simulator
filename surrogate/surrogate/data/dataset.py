"""PyTorch Dataset for loading LBM training pairs from HDF5 files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from surrogate.config import DATA_DIR, CanonicalGrid


class WindFieldDataset(Dataset):
    """Dataset of (input_tensor, velocity_target) pairs from HDF5 files.

    Each sample constructs an 8-channel input tensor:
        Ch 0:   Occupancy (binary)
        Ch 1:   Signed distance field (positive outside, negative inside buildings)
        Ch 2-3: sin/cos of wind direction
        Ch 4:   Normalised wind speed (speed / 15.0)
        Ch 5-7: Spatial coordinates (z/D, y/H, x/W)

    Target: velocity field [3, D, H, W] in m/s.
    """

    def __init__(
        self,
        preset_ids: list[str],
        split: str = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        data_dir: Path | None = None,
        cache_in_ram: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir or DATA_DIR
        self.split = split
        self._entries: list[tuple[str, int]] = []  # (preset_id, sample_index)

        self.cache_in_ram = cache_in_ram

        # Pre-compute SDF for each preset's base occupancy
        self._sdf_cache: dict[str, np.ndarray] = {}
        self._occ_cache: dict[str, np.ndarray] = {}

        # Optional full RAM cache: {preset_id: np.ndarray [N, D, H, W, 3] float32}
        self._vel_cache: dict[str, np.ndarray] = {}
        self._params_cache: dict[str, np.ndarray] = {}

        for pid in preset_ids:
            h5_path = self.data_dir / f"{pid}.h5"
            if not h5_path.exists():
                continue

            with h5py.File(str(h5_path), "r") as hf:
                n_total = hf["velocity"].attrs.get("n_written", hf["velocity"].shape[0])
                n_total = int(n_total)
                occ_sample = hf["occupancy"][0]  # same for all samples of this preset

            # Compute SDF once per preset
            self._occ_cache[pid] = occ_sample.astype(np.float32)
            self._sdf_cache[pid] = self._compute_sdf(occ_sample)

            # Split indices
            n_train = int(n_total * train_frac)
            n_val = int(n_total * val_frac)
            if split == "train":
                indices = range(0, n_train)
            elif split == "val":
                indices = range(n_train, n_train + n_val)
            else:  # test
                indices = range(n_train + n_val, n_total)

            for idx in indices:
                self._entries.append((pid, idx))

            # RAM cache: load all samples for this preset into memory up front.
            # Eliminates all HDF5 I/O during training at the cost of CPU RAM.
            # Memory: ~50 MB per 1000 samples at float16 storage (float32 after cast).
            # A full preset (288 samples) at 64×128×128×3 float32 ≈ 7 GB — check
            # your RAM before enabling.  Use --cache-ram only if you have headroom.
            if cache_in_ram:
                with h5py.File(str(h5_path), "r") as hf:
                    self._vel_cache[pid] = hf["velocity"][:n_total].astype(np.float32)
                    self._params_cache[pid] = hf["params"][:n_total].astype(np.float32)
                print(f"  Cached {pid} ({n_total} samples, {self._vel_cache[pid].nbytes / 1e6:.0f} MB)")

        # Pre-compute coordinate grids (shared across all samples)
        grid = CanonicalGrid()
        zc = np.linspace(0, 1, grid.depth, dtype=np.float32)
        yc = np.linspace(0, 1, grid.height, dtype=np.float32)
        xc = np.linspace(0, 1, grid.width, dtype=np.float32)
        Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
        self._coords = np.stack([Z, Y, X], axis=0)  # [3, D, H, W]

        # For lazy HDF5 reading: store paths
        self._h5_handles: dict[str, h5py.File] = {}

    @staticmethod
    def _compute_sdf(occupancy: np.ndarray) -> np.ndarray:
        """Compute signed distance field from binary occupancy grid.

        Positive values = distance from nearest building surface (air).
        Negative values = distance inside buildings (negated).
        Normalised to [0, 1] range by dividing by max distance.
        """
        air = occupancy == 0
        solid = occupancy == 1

        # Distance from surface for air cells
        dist_air = distance_transform_edt(air).astype(np.float32)
        # Distance from surface for solid cells (inverted sign)
        dist_solid = distance_transform_edt(solid).astype(np.float32)

        sdf = dist_air - dist_solid
        # Normalise
        max_dist = max(dist_air.max(), dist_solid.max(), 1.0)
        sdf = sdf / max_dist
        return sdf

    def _get_h5(self, preset_id: str) -> h5py.File:
        """Lazily open HDF5 files (one per preset)."""
        if preset_id not in self._h5_handles:
            path = self.data_dir / f"{preset_id}.h5"
            self._h5_handles[preset_id] = h5py.File(str(path), "r")
        return self._h5_handles[preset_id]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (input_tensor, target_velocity, occupancy_mask).

        Shapes:
            input_tensor:    (8, D, H, W)  float32
            target_velocity: (3, D, H, W)  float32
            occupancy_mask:  (1, D, H, W)  float32
        """
        preset_id, sample_idx = self._entries[idx]

        if self.cache_in_ram:
            # Zero-copy slices from pre-loaded numpy arrays
            params = self._params_cache[preset_id][sample_idx]
            vel = self._vel_cache[preset_id][sample_idx]  # already float32
        else:
            hf = self._get_h5(preset_id)
            params = hf["params"][sample_idx]
            # Cast to float32 regardless of on-disk dtype (supports float16 and float32 files)
            vel = hf["velocity"][sample_idx].astype(np.float32)  # [D, H, W, 3]

        direction_rad = params[0]
        speed = params[1]

        # [D, H, W, 3] -> [3, D, H, W]
        vel = vel.transpose(3, 0, 1, 2)

        # Build input channels
        occ = self._occ_cache[preset_id]   # [D, H, W]
        sdf = self._sdf_cache[preset_id]   # [D, H, W]
        grid = CanonicalGrid()

        # Scalar parameter channels (broadcast to spatial dims)
        sin_dir = np.full((1, grid.depth, grid.height, grid.width), np.sin(direction_rad), dtype=np.float32)
        cos_dir = np.full((1, grid.depth, grid.height, grid.width), np.cos(direction_rad), dtype=np.float32)
        norm_speed = np.full((1, grid.depth, grid.height, grid.width), speed / 15.0, dtype=np.float32)

        # Stack all 8 channels
        input_tensor = np.concatenate(
            [
                occ[np.newaxis],   # Ch 0: occupancy
                sdf[np.newaxis],   # Ch 1: SDF
                sin_dir,           # Ch 2: sin(dir)
                cos_dir,           # Ch 3: cos(dir)
                norm_speed,        # Ch 4: speed
                self._coords,      # Ch 5-7: z, y, x coordinates
            ],
            axis=0,
        )

        occ_mask = occ[np.newaxis].astype(np.float32)

        return (
            torch.from_numpy(input_tensor),
            torch.from_numpy(vel.copy()),
            torch.from_numpy(occ_mask),
        )

    def close(self):
        """Close any open HDF5 file handles."""
        for hf in self._h5_handles.values():
            hf.close()
        self._h5_handles.clear()

    def __del__(self):
        self.close()
