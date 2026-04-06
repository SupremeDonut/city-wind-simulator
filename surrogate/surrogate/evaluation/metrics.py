"""Evaluation metrics for comparing surrogate predictions against LBM ground truth."""

from __future__ import annotations

import numpy as np


def relative_l2(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Relative L2 error: ||pred - target||_2 / ||target||_2.

    Args:
        pred:   Predicted velocity [Nz, Ny, Nx, 3].
        target: Ground truth velocity [Nz, Ny, Nx, 3].
        mask:   Optional air mask [Nz, Ny, Nx] (1 = air, 0 = solid).

    Returns:
        Relative L2 error (scalar).
    """
    diff = pred - target
    if mask is not None:
        mask_4d = mask[..., np.newaxis]
        diff = diff * mask_4d
        target_masked = target * mask_4d
    else:
        target_masked = target

    return float(np.linalg.norm(diff) / (np.linalg.norm(target_masked) + 1e-8))


def mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean squared error over air voxels.

    Args:
        pred:   [Nz, Ny, Nx, 3]
        target: [Nz, Ny, Nx, 3]
        mask:   Optional [Nz, Ny, Nx] air mask.

    Returns:
        MSE (scalar).
    """
    diff = pred - target
    if mask is not None:
        mask_4d = mask[..., np.newaxis]
        n = mask.sum() * 3 + 1e-8
        return float((diff ** 2 * mask_4d).sum() / n)
    return float(np.mean(diff ** 2))


def max_pointwise_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Maximum absolute pointwise error across the field.

    Returns:
        Max |pred - target| (scalar, m/s).
    """
    return float(np.max(np.abs(pred - target)))


def divergence_field(vel: np.ndarray) -> np.ndarray:
    """Compute divergence of a velocity field using central differences.

    Args:
        vel: [Nz, Ny, Nx, 3] velocity field.

    Returns:
        div: [Nz-2, Ny-2, Nx-2] divergence at interior points.
    """
    dudx = (vel[1:-1, 1:-1, 2:, 0] - vel[1:-1, 1:-1, :-2, 0]) / 2.0
    dvdy = (vel[1:-1, 2:, 1:-1, 1] - vel[1:-1, :-2, 1:-1, 1]) / 2.0
    dwdz = (vel[2:, 1:-1, 1:-1, 2] - vel[:-2, 1:-1, 1:-1, 2]) / 2.0
    return dudx + dvdy + dwdz


def mean_abs_divergence(vel: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean absolute divergence over air voxels.

    Args:
        vel:  [Nz, Ny, Nx, 3] velocity field.
        mask: Optional [Nz, Ny, Nx] air mask.

    Returns:
        mean(|div(u)|) at interior points.
    """
    div = divergence_field(vel)
    if mask is not None:
        mask_interior = mask[1:-1, 1:-1, 1:-1]
        n = mask_interior.sum() + 1e-8
        return float(np.sum(np.abs(div) * mask_interior) / n)
    return float(np.mean(np.abs(div)))


def comfort_mae(
    pred_vel: np.ndarray, target_vel: np.ndarray, z_level: int = 0
) -> float:
    """Mean absolute error of comfort map (horizontal speed at pedestrian level).

    Args:
        pred_vel:   [Nz, Ny, Nx, 3] predicted velocity.
        target_vel: [Nz, Ny, Nx, 3] ground truth velocity.
        z_level:    Z index for pedestrian level (default 0).

    Returns:
        MAE of horizontal speed (m/s).
    """
    pred_comfort = np.sqrt(
        pred_vel[z_level, :, :, 0] ** 2 + pred_vel[z_level, :, :, 1] ** 2
    )
    target_comfort = np.sqrt(
        target_vel[z_level, :, :, 0] ** 2 + target_vel[z_level, :, :, 1] ** 2
    )
    return float(np.mean(np.abs(pred_comfort - target_comfort)))


def lawson_category(speed: np.ndarray) -> np.ndarray:
    """Classify wind speed into Lawson comfort categories.

    Categories:
        0: < 2.5 m/s  (comfortable sitting)
        1: 2.5-4 m/s  (comfortable standing)
        2: 4-6 m/s    (acceptable walking)
        3: 6-8 m/s    (uncomfortable)
        4: > 8 m/s    (dangerous)
    """
    cats = np.zeros_like(speed, dtype=np.int32)
    cats[speed >= 2.5] = 1
    cats[speed >= 4.0] = 2
    cats[speed >= 6.0] = 3
    cats[speed >= 8.0] = 4
    return cats


def lawson_agreement(
    pred_vel: np.ndarray, target_vel: np.ndarray, z_level: int = 0
) -> float:
    """Percentage of cells where Lawson comfort category matches.

    Args:
        pred_vel:   [Nz, Ny, Nx, 3]
        target_vel: [Nz, Ny, Nx, 3]
        z_level:    Z index for pedestrian level.

    Returns:
        Agreement fraction (0..1).
    """
    pred_speed = np.sqrt(
        pred_vel[z_level, :, :, 0] ** 2 + pred_vel[z_level, :, :, 1] ** 2
    )
    target_speed = np.sqrt(
        target_vel[z_level, :, :, 0] ** 2 + target_vel[z_level, :, :, 1] ** 2
    )
    pred_cats = lawson_category(pred_speed)
    target_cats = lawson_category(target_speed)
    return float(np.mean(pred_cats == target_cats))
