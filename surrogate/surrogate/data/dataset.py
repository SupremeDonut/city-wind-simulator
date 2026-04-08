"""PyTorch Dataset for loading LBM training pairs from HDF5 files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from surrogate.config import DATA_DIR, CanonicalGrid, DataGenConfig


class WindFieldDataset(Dataset):
    """Dataset of (input_tensor, velocity_target) pairs from HDF5 files.

    Each sample constructs a 7-channel input tensor:
        Ch 0:   Occupancy (binary) — per-sample rotated occupancy
        Ch 1:   Signed distance field (computed per-sample from rotated occ)
        Ch 2-3: sin/cos of wind direction
        Ch 4-6: Spatial coordinates (z/D, y/H, x/W)

    Target: velocity field [3, D, H, W] normalised by training_speed (unit-speed pattern).
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

        # Optional full RAM cache: {preset_id: np.ndarray}
        self._vel_cache: dict[str, np.ndarray] = {}
        self._params_cache: dict[str, np.ndarray] = {}
        self._occ_cache: dict[str, np.ndarray] = {}
        self._sdf_cache: dict[str, np.ndarray] = {}

        for pid in preset_ids:
            h5_path = self.data_dir / f"{pid}.h5"
            if not h5_path.exists():
                continue

            with h5py.File(str(h5_path), "r") as hf:
                n_total = hf["velocity"].attrs.get("n_written", hf["velocity"].shape[0])
                n_total = int(n_total)

            # Split indices — shuffle with a fixed seed so train/val/test are
            # consistent across runs but cover all directions/speeds in each split.
            rng = np.random.default_rng(seed=42)
            all_indices = np.arange(n_total)
            rng.shuffle(all_indices)

            n_train = int(n_total * train_frac)
            n_val = int(n_total * val_frac)
            if split == "train":
                indices = all_indices[:n_train]
            elif split == "val":
                indices = all_indices[n_train : n_train + n_val]
            else:  # test
                indices = all_indices[n_train + n_val :]

            for idx in indices:
                self._entries.append((pid, idx))

            # RAM cache: load all data for this preset into memory up front.
            # Each sample has its own rotated occupancy, so we cache occ + SDF
            # per sample.
            if cache_in_ram:
                with h5py.File(str(h5_path), "r") as hf:
                    self._vel_cache[pid] = hf["velocity"][:n_total].astype(np.float32)
                    self._params_cache[pid] = hf["params"][:n_total].astype(np.float32)
                    occ_all = hf["occupancy"][:n_total]
                self._occ_cache[pid] = occ_all.astype(np.float32)
                # Pre-compute SDF for every sample
                sdf_all = np.empty_like(self._occ_cache[pid])
                for i in range(n_total):
                    sdf_all[i] = self._compute_sdf(occ_all[i])
                self._sdf_cache[pid] = sdf_all
                mem_mb = (self._vel_cache[pid].nbytes + self._occ_cache[pid].nbytes + sdf_all.nbytes) / 1e6
                print(f"  Cached {pid} ({n_total} samples, {mem_mb:.0f} MB)")

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
            input_tensor:    (7, D, H, W)  float32
            target_velocity: (3, D, H, W)  float32
            occupancy_mask:  (1, D, H, W)  float32
        """
        preset_id, sample_idx = self._entries[idx]

        if self.cache_in_ram:
            params = self._params_cache[preset_id][sample_idx]
            vel = self._vel_cache[preset_id][sample_idx]  # already float32
            occ = self._occ_cache[preset_id][sample_idx]  # [D, H, W] float32
            sdf = self._sdf_cache[preset_id][sample_idx]  # [D, H, W] float32
        else:
            hf = self._get_h5(preset_id)
            params = hf["params"][sample_idx]
            vel = hf["velocity"][sample_idx].astype(np.float32)
            occ = hf["occupancy"][sample_idx].astype(np.float32)
            sdf = self._compute_sdf(occ)

        direction_rad = params[0]

        # [D, H, W, 3] -> [3, D, H, W]
        vel = vel.transpose(3, 0, 1, 2)

        # Normalise by training speed so the model learns a unit-speed flow
        # pattern (free-stream ≈ 1.0).  At inference, multiply the model
        # output by the requested speed to recover physical m/s.
        training_speed = DataGenConfig().training_speed
        vel = vel / training_speed

        grid = CanonicalGrid()

        # Direction channels (broadcast to spatial dims)
        sin_dir = np.full((1, grid.depth, grid.height, grid.width), np.sin(direction_rad), dtype=np.float32)
        cos_dir = np.full((1, grid.depth, grid.height, grid.width), np.cos(direction_rad), dtype=np.float32)

        # Stack 7 channels
        input_tensor = np.concatenate(
            [
                occ[np.newaxis],   # Ch 0: occupancy (per-sample rotated)
                sdf[np.newaxis],   # Ch 1: SDF (computed from rotated occ)
                sin_dir,           # Ch 2: sin(dir)
                cos_dir,           # Ch 3: cos(dir)
                self._coords,      # Ch 4-6: z, y, x coordinates
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
