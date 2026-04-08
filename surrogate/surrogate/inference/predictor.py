"""Inference wrapper for the trained FNO surrogate model.

Provides ``WindSurrogate`` — a class that loads a trained checkpoint and
predicts velocity fields with the exact same signature as the procedural
``generate()`` function in ``api/simulation.py``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from surrogate.config import ASSETS_DIR, CHECKPOINT_DIR, ModelConfig
from surrogate.model.fno3d import FNO3d

logger = logging.getLogger(__name__)


def _compute_sdf(occ: np.ndarray) -> np.ndarray:
    """Compute normalised signed distance field from a binary occupancy grid."""
    air = occ == 0
    solid = occ == 1
    dist_air = distance_transform_edt(air).astype(np.float32)
    dist_solid = distance_transform_edt(solid).astype(np.float32)
    sdf = dist_air - dist_solid
    max_dist = max(dist_air.max(), dist_solid.max(), 1.0)
    return sdf / max_dist


class WindSurrogate:
    """Loads a trained FNO checkpoint and provides fast wind field predictions.

    Caches preset base occupancy grids and per-direction rotated occupancy +
    SDF on the target device to minimise per-prediction overhead.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ):
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "best.pt"
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}\n"
                "Train the surrogate first: uv run python -m surrogate.training.train"
            )

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_cfg_dict = checkpoint.get("model_config", {})
        model_cfg = ModelConfig(**model_cfg_dict)
        self.grid = model_cfg.grid

        self.model = FNO3d(model_cfg).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(
            "Loaded FNO surrogate from %s (epoch %d, device=%s)",
            checkpoint_path, checkpoint.get("epoch", -1), self.device,
        )

        # Per-preset base occupancy (un-rotated, downsampled, before canonical fit)
        self._base_cache: dict[str, dict] = {}
        # Per-(preset, direction) rotated occ+SDF on device
        self._rotated_cache: dict[tuple[str, float], dict] = {}
        self._coord_grid: torch.Tensor | None = None

    def _ensure_coord_grid(self) -> torch.Tensor:
        """Lazily create and cache the normalised spatial coordinate grid."""
        if self._coord_grid is None:
            g = self.grid
            zc = np.linspace(0, 1, g.depth, dtype=np.float32)
            yc = np.linspace(0, 1, g.height, dtype=np.float32)
            xc = np.linspace(0, 1, g.width, dtype=np.float32)
            Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
            coords = np.stack([Z, Y, X], axis=0)  # [3, D, H, W]
            self._coord_grid = torch.from_numpy(coords).to(self.device)
        return self._coord_grid

    def _ensure_base(self, preset_id: str) -> dict:
        """Lazily load and cache the un-rotated base occupancy for a preset."""
        if preset_id not in self._base_cache:
            from lbm.preprocess import downsample_block_max

            h5_path = ASSETS_DIR / f"{preset_id}.h5"
            if not h5_path.exists():
                raise FileNotFoundError(f"Preset HDF5 not found: {h5_path}")

            with h5py.File(str(h5_path), "r") as hf:
                occ = hf["occupancy"][:]
                domain_size = list(int(v) for v in hf["occupancy"].attrs["domain_size"])

            # Downsample to match training resolution (8m pitch)
            native_pitch = domain_size[0] / occ.shape[2]
            factor = max(1, round(8.0 / native_pitch))
            occ = downsample_block_max(occ, factor)
            # Do NOT add ground plane or fit to canonical grid yet — rotation
            # must happen on the raw downsampled geometry first (matching the
            # data generation pipeline in surrogate/data/generate.py).

            # Compute the un-rotated output shape: how many voxels of the
            # downsampled grid fit inside the canonical grid.  This is used to
            # unpad the model output (which is in the original spatial frame).
            g = self.grid
            orig_shape = (
                min(occ.shape[0], g.depth),
                min(occ.shape[1], g.height),
                min(occ.shape[2], g.width),
            )

            self._base_cache[preset_id] = {
                "occ_ds": occ,              # [NZ, NY, NX] uint8, no ground
                "domain_size": domain_size,
                "orig_shape": orig_shape,   # un-rotated shape clamped to canonical
            }

        return self._base_cache[preset_id]

    def _ensure_rotated(self, preset_id: str, direction: float) -> dict:
        """Lazily rotate, pad, and cache occ + SDF for a (preset, direction) pair.

        Mirrors the rotation logic in ``surrogate/data/generate.py``:
        ``_build_rotated_occ_cache`` rotates the downsampled occupancy by
        ``(direction + 180) % 360`` before adding the ground plane and fitting
        to the canonical grid.  The model was trained on these rotated grids,
        so inference must match.
        """
        # Snap direction to the nearest training increment (5°) for cache reuse
        snap = round(direction / 5.0) * 5.0 % 360
        cache_key = (preset_id, snap)

        if cache_key not in self._rotated_cache:
            from lbm.preprocess import rotate_voxels

            base = self._ensure_base(preset_id)
            occ_ds = base["occ_ds"].copy()

            # Rotate geometry — same formula as data generation & LBM solver
            voxel_rotation = (snap + 180) % 360
            occ_rot = rotate_voxels(occ_ds, voxel_rotation)
            occ_rot[0, :, :] = 1  # ground plane

            # Fit to canonical grid: clip then pad
            g = self.grid
            occ_rot = occ_rot[: g.depth, : g.height, : g.width]

            pad_z = g.depth - occ_rot.shape[0]
            pad_y = g.height - occ_rot.shape[1]
            pad_x = g.width - occ_rot.shape[2]
            occ_padded = np.pad(occ_rot, ((0, pad_z), (0, pad_y), (0, pad_x)),
                                mode="constant")

            sdf = _compute_sdf(occ_padded)

            occ_t = torch.from_numpy(occ_padded.astype(np.float32)).unsqueeze(0).to(self.device)
            sdf_t = torch.from_numpy(sdf).unsqueeze(0).to(self.device)

            self._rotated_cache[cache_key] = {
                "occ": occ_t,
                "sdf": sdf_t,
            }

        return self._rotated_cache[cache_key]

    @torch.no_grad()
    def predict(
        self,
        preset_id: str,
        direction: float,
        speed: float,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Predict a wind velocity field for given conditions.

        Returns:
            vel:     float32 ndarray, shape [Nz, Ny, Nx, 3]
            comfort: float32 ndarray, shape [Ny, Nx]
            domain:  [domX, domY, domZ] metres

        Matches the signature of ``api.simulation.generate()``.
        """
        base = self._ensure_base(preset_id)
        rotated = self._ensure_rotated(preset_id, direction)
        coords = self._ensure_coord_grid()
        g = self.grid

        # Build parameter channels
        direction_rad = math.radians(direction)
        sin_dir = torch.full((1, g.depth, g.height, g.width), math.sin(direction_rad),
                             dtype=torch.float32, device=self.device)
        cos_dir = torch.full((1, g.depth, g.height, g.width), math.cos(direction_rad),
                             dtype=torch.float32, device=self.device)

        # Stack input: [1, 7, D, H, W]
        # Uses the ROTATED occupancy and SDF — matching training data layout
        x = torch.cat(
            [
                rotated["occ"],   # Ch 0: rotated occupancy
                rotated["sdf"],   # Ch 1: SDF of rotated occupancy
                sin_dir,          # Ch 2
                cos_dir,          # Ch 3
                coords,           # Ch 4-6
            ],
            dim=0,
        ).unsqueeze(0)  # add batch dim

        # Inference — use rotated occupancy for the hard solid mask too
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            pred = self.model(x, occupancy=rotated["occ"].unsqueeze(0))

        # pred shape: [1, 3, D, H, W] -> numpy [D, H, W, 3]
        # The model predicts a unit-speed flow pattern (targets were normalised
        # by training_speed during training: target = vel_physical / 7.0).
        # To recover physical m/s at the requested wind speed, multiply by
        # `speed` (not `speed / training_speed` — that would divide by 7 twice).
        pred_np = pred[0].permute(1, 2, 3, 0).cpu().numpy() * speed

        # Unpad to the un-rotated domain shape.  The model was trained on
        # velocity targets that were spatially rotated back to the original
        # (un-rotated) grid frame, so the active region in the canonical
        # output corresponds to the un-rotated domain dimensions.
        nz, ny, nx = base["orig_shape"]
        vel = pred_np[:nz, :ny, :nx, :].astype(np.float32)

        # Comfort map: horizontal wind speed at z=0 (pedestrian level)
        comfort = np.sqrt(vel[0, :, :, 0] ** 2 + vel[0, :, :, 1] ** 2).astype(np.float32)

        domain = base["domain_size"]

        return vel, comfort, domain
