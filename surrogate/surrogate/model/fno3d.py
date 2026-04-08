"""3D Fourier Neural Operator for urban wind field prediction.

Architecture:
    Lifting (Conv3d 1x1x1) -> N x FourierLayer -> Projection MLP -> Hard solid mask

Input channels (7):
    0: Occupancy grid (binary)
    1: Signed distance field from buildings
    2: sin(direction)   (broadcast)
    3: cos(direction)   (broadcast)
    4: z / D  (spatial coordinate)
    5: y / H  (spatial coordinate)
    6: x / W  (spatial coordinate)

Output: velocity field [B, 3, D, H, W]
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from surrogate.config import ModelConfig
from surrogate.model.layers import FourierLayer


class FNO3d(nn.Module):
    """3D Fourier Neural Operator for wind field prediction."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # Lifting: pointwise projection from input channels to hidden width
        self.lifting = nn.Conv3d(cfg.in_channels, cfg.width, kernel_size=1)

        # Fourier layers
        self.fourier_layers = nn.ModuleList(
            [
                FourierLayer(cfg.width, cfg.modes_z, cfg.modes_y, cfg.modes_x)
                for _ in range(cfg.n_layers)
            ]
        )

        # Projection: width -> hidden -> out_channels
        self.projection = nn.Sequential(
            nn.Conv3d(cfg.width, cfg.projection_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(cfg.projection_hidden, cfg.out_channels, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        occupancy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).
            occupancy: Optional binary occupancy (B, 1, D, H, W) for hard masking.
                       If None, no solid mask is applied.

        Returns:
            Velocity field (B, 3, D, H, W).
        """
        # Lifting
        h = self.lifting(x)

        # Fourier layers — gradient checkpointing trades recompute for memory,
        # preventing the ~4 GB activation spike that causes 11-13 s steps on
        # 12 GB GPUs.  Disable with ModelConfig(use_checkpoint=False) if you
        # have enough VRAM and want to measure raw speed without recompute.
        for layer in self.fourier_layers:
            if self.cfg.use_checkpoint and self.training:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)

        # Projection
        out = self.projection(h)

        # Hard solid mask: enforce zero velocity inside buildings
        if occupancy is not None:
            out = out * (1.0 - occupancy)

        return out
