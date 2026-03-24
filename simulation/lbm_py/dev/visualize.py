"""
PyVista visualizer for D3Q19 wind simulation output.

Usage:
    uv run python visualize.py                     # geometry only
    uv run python visualize.py wind_field.npz      # geometry + vector field
"""

import sys

import h5py
import numpy as np
import pyvista as pv

VOXEL_PATH = "../../../assets/presets/chicago.h5"
DOWNSAMPLE = 5


def load_voxels(voxel_path: str) -> tuple[np.ndarray, float]:
    with h5py.File(voxel_path, "r") as hf:
        s = slice(None, None, DOWNSAMPLE)
        occ = hf["occupancy"][s, s, s]
        domain_size = hf["occupancy"].attrs["domain_size"]
    solid = (occ > 0).astype(np.uint8)
    solid[0, :, :] = 1  # ground floor
    dx = float(domain_size[0]) / solid.shape[2]  # metres per cell
    return solid, dx


def vtk_order(arr: np.ndarray) -> np.ndarray:
    """Reorder numpy [NZ, NY, NX] array to VTK flat order (x varies fastest).

    solid[k, j, i] → VTK flat index i + j*NX + k*NX*NY.
    C-order ravel of [NZ, NY, NX] gives exactly that since x (last axis) varies fastest.
    """
    return arr.ravel()


def make_grid(NX: int, NY: int, NZ: int, dx: float) -> pv.ImageData:
    grid = pv.ImageData()
    grid.dimensions = (NX + 1, NY + 1, NZ + 1)
    grid.spacing = (dx, dx, dx)
    grid.origin = (0.0, 0.0, 0.0)
    return grid


def visualize(voxel_path: str, field_path: str | None) -> None:
    solid, dx = load_voxels(voxel_path)
    NZ, NY, NX = solid.shape
    print(f"Grid {NX}×{NY}×{NZ}, cell size {dx:.1f} m")

    grid = make_grid(NX, NY, NZ, dx)
    grid.cell_data["solid"] = vtk_order(solid)

    has_field = field_path is not None
    if has_field:
        data = np.load(field_path)
        ux, uy, uz = data["ux"], data["uy"], data["uz"]
        dx = float(data["dx"]) if "dx" in data else dx
        speed = np.sqrt(ux**2 + uy**2 + uz**2)
        grid.cell_data["speed"] = vtk_order(speed)
        grid.cell_data["velocity"] = np.column_stack(
            [vtk_order(ux), vtk_order(uy), vtk_order(uz)]
        )

    pl = pv.Plotter(window_size=(1400, 900))
    pl.set_background("#1a1a2e")

    # --- Buildings: threshold solid cells, extract surface ---
    buildings = grid.threshold(0.5, scalars="solid").extract_surface()
    pl.add_mesh(
        buildings, color="#d4d4d4", opacity=1.0, smooth_shading=True, label="Buildings"
    )

    if has_field:
        # Convert cell data to point data for smooth slice coloring
        point_grid = grid.cell_data_to_point_data()
        speed_max = float(speed[~solid.astype(bool)].max())
        clim = [0.0, speed_max]

        # Horizontal slice at pedestrian height (z index 2)
        z_ped = 2.5 * dx
        sl_h = point_grid.slice(normal="z", origin=(0.0, 0.0, z_ped))
        pl.add_mesh(
            sl_h,
            scalars="speed",
            cmap="inferno",
            clim=clim,
            opacity=0.9,
            label="Speed — plan",
        )

        # Vertical slice at domain midplane (y = NY/2)
        y_mid = (NY / 2) * dx
        sl_v = point_grid.slice(normal="y", origin=(0.0, y_mid, 0.0))
        pl.add_mesh(
            sl_v,
            scalars="speed",
            cmap="inferno",
            clim=clim,
            opacity=0.9,
            label="Speed — elevation",
        )

        # --- Arrow glyphs: subsample fluid cells ---
        step = 4
        iz, iy, ix = np.mgrid[1:NZ:step, 0:NY:step, 0:NX:step]
        fluid_mask = ~solid[iz, iy, ix].astype(bool)
        pts = np.column_stack(
            [
                (ix[fluid_mask] + 0.5) * dx,
                (iy[fluid_mask] + 0.5) * dx,
                (iz[fluid_mask] + 0.5) * dx,
            ]
        )
        vels = np.column_stack(
            [
                ux[iz[fluid_mask], iy[fluid_mask], ix[fluid_mask]],
                uy[iz[fluid_mask], iy[fluid_mask], ix[fluid_mask]],
                uz[iz[fluid_mask], iy[fluid_mask], ix[fluid_mask]],
            ]
        )
        spd = np.linalg.norm(vels, axis=1)

        arrows_pd = pv.PolyData(pts)
        arrows_pd["velocity"] = vels
        arrows_pd["speed"] = spd
        glyphs = arrows_pd.glyph(
            orient="velocity",
            scale="speed",
            factor=dx * 8,
            geom=pv.Arrow(),
            tolerance=0.0,
        )
        pl.add_mesh(
            glyphs, scalars="speed", cmap="plasma", clim=clim, label="Velocity vectors"
        )

    pl.add_axes(line_width=3)
    pl.show_grid(color="white", font_size=10)
    pl.show()


if __name__ == "__main__":
    field = sys.argv[1] if len(sys.argv) > 1 else "wind_field.npz"
    import os

    if not os.path.exists(field):
        print(f"No field file found at '{field}' — showing geometry only.")
        field = None
    visualize(VOXEL_PATH, field)
