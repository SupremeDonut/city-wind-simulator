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

from surrogate.config import ASSETS_DIR, CHECKPOINT_DIR, CanonicalGrid, ModelConfig
from surrogate.model.fno3d import FNO3d

logger = logging.getLogger(__name__)


class WindSurrogate:
    """Loads a trained FNO checkpoint and provides fast wind field predictions.

    Caches preset occupancy grids, SDFs, and coordinate grids on the target
    device to minimise per-prediction overhead.
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

        # Pre-compute and cache per-preset data
        self._preset_cache: dict[str, dict] = {}
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

    def _ensure_preset(self, preset_id: str) -> dict:
        """Lazily load and cache occupancy + SDF for a preset."""
        if preset_id not in self._preset_cache:
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
            occ[0, :, :] = 1  # ground plane

            # Fit to canonical grid: clip excess (upper Z is free-stream atmosphere),
            # then pad any shortfall with zeros.
            g = self.grid
            occ = occ[: g.depth, : g.height, : g.width]  # clip
            orig_shape = occ.shape  # (NZ, NY, NX) — already clamped to canonical

            pad_z = g.depth - occ.shape[0]
            pad_y = g.height - occ.shape[1]
            pad_x = g.width - occ.shape[2]
            occ_padded = np.pad(occ, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")

            # SDF
            air = occ_padded == 0
            solid = occ_padded == 1
            dist_air = distance_transform_edt(air).astype(np.float32)
            dist_solid = distance_transform_edt(solid).astype(np.float32)
            sdf = dist_air - dist_solid
            max_dist = max(dist_air.max(), dist_solid.max(), 1.0)
            sdf = sdf / max_dist

            # To tensors on device
            occ_t = torch.from_numpy(occ_padded.astype(np.float32)).unsqueeze(0).to(self.device)  # [1, D, H, W]
            sdf_t = torch.from_numpy(sdf).unsqueeze(0).to(self.device)  # [1, D, H, W]

            self._preset_cache[preset_id] = {
                "occ": occ_t,
                "sdf": sdf_t,
                "domain_size": domain_size,
                "orig_shape": orig_shape,
            }

        return self._preset_cache[preset_id]

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
        cache = self._ensure_preset(preset_id)
        coords = self._ensure_coord_grid()
        g = self.grid

        # Build parameter channels
        direction_rad = math.radians(direction)
        sin_dir = torch.full((1, g.depth, g.height, g.width), math.sin(direction_rad),
                             dtype=torch.float32, device=self.device)
        cos_dir = torch.full((1, g.depth, g.height, g.width), math.cos(direction_rad),
                             dtype=torch.float32, device=self.device)
        norm_speed = torch.full((1, g.depth, g.height, g.width), speed / 15.0,
                                dtype=torch.float32, device=self.device)

        # Stack input: [1, 8, D, H, W]
        x = torch.cat(
            [
                cache["occ"],   # Ch 0
                cache["sdf"],   # Ch 1
                sin_dir,        # Ch 2
                cos_dir,        # Ch 3
                norm_speed,     # Ch 4
                coords,         # Ch 5-7
            ],
            dim=0,
        ).unsqueeze(0)  # add batch dim

        # Inference
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            pred = self.model(x, occupancy=cache["occ"].unsqueeze(0))

        # pred shape: [1, 3, D, H, W] -> numpy [D, H, W, 3]
        pred_np = pred[0].permute(1, 2, 3, 0).cpu().numpy()

        # Unpad to original grid size
        nz, ny, nx = cache["orig_shape"]
        vel = pred_np[:nz, :ny, :nx, :].astype(np.float32)

        # Comfort map: horizontal wind speed at z=0 (pedestrian level)
        comfort = np.sqrt(vel[0, :, :, 0] ** 2 + vel[0, :, :, 1] ** 2).astype(np.float32)

        domain = cache["domain_size"]

        return vel, comfort, domain
