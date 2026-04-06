"""Loss functions for training the FNO surrogate model.

Implements:
    - Relative L2 loss (data fidelity) with optional pedestrian-level weighting
    - Divergence penalty (incompressibility regularisation)
"""

import torch


def relative_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_air: torch.Tensor,
    pedestrian_weight: float = 2.0,
) -> torch.Tensor:
    """Relative L2 loss over air voxels with pedestrian-layer emphasis.

    Args:
        pred:   (B, 3, D, H, W) predicted velocity.
        target: (B, 3, D, H, W) ground truth velocity.
        mask_air: (B, 1, D, H, W) binary mask (1 = air, 0 = solid).
        pedestrian_weight: Extra weight for z=0..2 (pedestrian level).

    Returns:
        Scalar loss.
    """
    diff = (pred - target) * mask_air  # zero out solid cells

    # Build per-voxel weight: pedestrian layers (z=0,1,2) get extra weight
    weight = torch.ones_like(mask_air)
    weight[:, :, :3, :, :] = pedestrian_weight
    weight = weight * mask_air  # only air cells

    weighted_diff = diff * weight
    numerator = torch.sqrt((weighted_diff ** 2).sum())
    denominator = torch.sqrt(((target * weight) ** 2).sum()) + 1e-8

    return numerator / denominator


def divergence_loss(
    pred: torch.Tensor,
    mask_air: torch.Tensor,
) -> torch.Tensor:
    """Divergence penalty: mean(div(u)^2) over air voxels.

    Enforces the incompressibility constraint div(u) = 0 using central
    finite differences.

    Args:
        pred:     (B, 3, D, H, W) predicted velocity (channels: ux, uy, uz).
        mask_air: (B, 1, D, H, W) binary mask (1 = air).

    Returns:
        Scalar loss.
    """
    # Central differences (interior points only)
    # du_x/dx  (channel 0, varies along W=dim4)
    dudx = (pred[:, 0:1, :, :, 2:] - pred[:, 0:1, :, :, :-2]) / 2.0
    # du_y/dy  (channel 1, varies along H=dim3)
    dvdy = (pred[:, 1:2, :, 2:, :] - pred[:, 1:2, :, :-2, :]) / 2.0
    # du_z/dz  (channel 2, varies along D=dim2)
    dwdz = (pred[:, 2:3, 2:, :, :] - pred[:, 2:3, :-2, :, :]) / 2.0

    # Crop to common interior region
    D, H, W = pred.shape[2], pred.shape[3], pred.shape[4]
    dudx = dudx[:, :, 1:-1, 1:-1, :]     # [B, 1, D-2, H-2, W-2]
    dvdy = dvdy[:, :, 1:-1, :, 1:-1]     # [B, 1, D-2, H-2, W-2]
    dwdz = dwdz[:, :, :, 1:-1, 1:-1]     # [B, 1, D-2, H-2, W-2]

    div = dudx + dvdy + dwdz  # [B, 1, D-2, H-2, W-2]

    # Apply air mask (cropped to interior)
    mask_interior = mask_air[:, :, 1:-1, 1:-1, 1:-1]
    div_masked = div * mask_interior

    n_air = mask_interior.sum() + 1e-8
    return (div_masked ** 2).sum() / n_air


def surrogate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    occupancy: torch.Tensor,
    lambda_div: float = 0.1,
    pedestrian_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss: relative L2 + divergence penalty.

    Args:
        pred:       (B, 3, D, H, W) predicted velocity.
        target:     (B, 3, D, H, W) ground truth velocity.
        occupancy:  (B, 1, D, H, W) binary occupancy (1 = solid).
        lambda_div: Weight for divergence penalty.
        pedestrian_weight: Extra weight for pedestrian-level accuracy.

    Returns:
        (total_loss, {"l_data": ..., "l_div": ..., "total": ...})
    """
    mask_air = 1.0 - occupancy

    l_data = relative_l2_loss(pred, target, mask_air, pedestrian_weight)
    l_div = divergence_loss(pred, mask_air)

    total = l_data + lambda_div * l_div

    components = {
        "l_data": l_data.item(),
        "l_div": l_div.item(),
        "total": total.item(),
    }
    return total, components
