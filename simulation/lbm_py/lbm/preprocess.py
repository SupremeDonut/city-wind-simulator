"""Preprocessing utilities for the CUDA LBM solver.

Handles voxel rotation (wind direction) and downsampling.
"""

import numpy as np
from scipy.ndimage import rotate


def rotate_voxels(occ: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate occupancy grid in the XY plane by `angle_deg` degrees.

    The LBM solver always pushes wind in the +Y direction (inlet at y=0,
    outlet at y=NY-1).  Rotating the geometry lets us simulate wind from
    any compass direction.

    Uses reshape=True so corners of the domain are not clipped.  The caller
    must rotate the output velocity field back and crop to original size.

    Parameters
    ----------
    occ : np.ndarray
        Binary occupancy grid, shape [NZ, NY, NX].
    angle_deg : float
        Rotation angle in degrees (positive = CCW in XY plane).

    Returns
    -------
    np.ndarray
        Rotated occupancy grid (binary uint8).  Shape may be larger than input.
    """
    if angle_deg == 0.0:
        return occ

    # axes=(2,1) rotates in the X-Y plane (index 2=X, index 1=Y in ZYX layout)
    # order=0 = nearest-neighbour to preserve binary occupancy
    # reshape=True avoids clipping corners
    rotated = rotate(occ.astype(np.float32), angle_deg, axes=(2, 1),
                     reshape=True, order=0)
    return (rotated > 0.5).astype(np.uint8)


def rotate_field_back(field: np.ndarray, angle_deg: float,
                      target_shape: tuple) -> np.ndarray:
    """Spatially rotate a 3D scalar field back by -angle_deg and center-crop.

    Parameters
    ----------
    field : np.ndarray
        3D array [NZ, NY, NX] on the rotated grid.
    angle_deg : float
        The angle that was used to rotate voxels (will rotate back by -angle_deg).
    target_shape : tuple
        (NZ, NY, NX) of the original (unrotated, downsampled) grid.

    Returns
    -------
    np.ndarray
        Field on the original grid, shape == target_shape.
    """
    if angle_deg == 0.0:
        return field

    # Rotate back (linear interpolation for smooth velocity)
    back = rotate(field.astype(np.float32), -angle_deg, axes=(2, 1),
                  reshape=False, order=1)

    # Center-crop to target shape (Z unchanged by XY rotation)
    tz, ty, tx = target_shape
    cz, cy, cx = back.shape
    y0 = (cy - ty) // 2
    x0 = (cx - tx) // 2
    return back[:tz, y0:y0 + ty, x0:x0 + tx]


def downsample_block_max(occ: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 3D occupancy grid using block-max pooling.

    A block is solid if ANY voxel in it is solid — conservative for
    preserving building geometry.

    Parameters
    ----------
    occ : np.ndarray
        Binary occupancy grid, shape [NZ, NY, NX].
    factor : int
        Downsampling factor (e.g. 4 means 2m → 8m).

    Returns
    -------
    np.ndarray
        Downsampled occupancy grid, shape [NZ//f, NY//f, NX//f], uint8.
    """
    if factor <= 1:
        return occ.astype(np.uint8)

    NZ, NY, NX = occ.shape
    nz, ny, nx = NZ // factor, NY // factor, NX // factor
    trimmed = occ[:nz * factor, :ny * factor, :nx * factor]
    blocks = trimmed.reshape(nz, factor, ny, factor, nx, factor)
    return blocks.max(axis=(1, 3, 5)).astype(np.uint8)
