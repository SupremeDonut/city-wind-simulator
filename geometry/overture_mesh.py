# Exports GLB building meshes sourced from Overture Maps, voxelises footprints
# into an HDF5 occupancy grid, and stitches an OSM map texture aligned to the
# same Web Mercator coordinate system used by the GLB geometry.

from typing import cast
import pyproj
import trimesh
import numpy as np
import shapely
import h5py
from shapely.geometry import box, Polygon as ShapelyPolygon
from geopandas import GeoDataFrame
from overturemaps import core

from osm_tiles import fetch_map_png


def export_preset_glb(
    lat, lon, dist, output_path, preset_id=None, domain_z=200.0, pitch=2.0
):
    hdf5_path = str(output_path).rsplit(".", 1)[0] + ".h5"

    # 1. Project centre to Web Mercator and compute clip box
    proj = pyproj.Proj("epsg:3857")
    cx_m, cy_m = proj(lon, lat)

    # Convert clip corners back to WGS84 for the Overture bbox parameter
    transformer = pyproj.Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(cx_m - dist, cy_m - dist)
    max_lon, max_lat = transformer.transform(cx_m + dist, cy_m + dist)

    # 2. Fetch buildings from Overture Maps (bbox is lon-first)
    buildings = core.geodataframe("building", bbox=(min_lon, min_lat, max_lon, max_lat))

    # 3. Filter to simple polygons and reproject to metres
    buildings = buildings[buildings.geometry.geom_type == "Polygon"].copy()
    buildings = buildings.set_crs(epsg=4326)
    buildings = cast(GeoDataFrame, buildings.to_crs(epsg=3857))

    clip_box = box(cx_m - dist, cy_m - dist, cx_m + dist, cy_m + dist)

    # 4. Extrude each building for GLB; collect shifted polygons for voxelisation
    meshes = []
    buildings_data = []  # (polygon_shifted, height)

    for _, row in buildings.iterrows():
        clipped = row.geometry.intersection(clip_box)
        if clipped.is_empty or clipped.geom_type != "Polygon":
            continue

        # Height chain: explicit metres → num_floors * 3.5 → default 10m
        height = row.get("height")
        if height is None:
            num_floors = row.get("num_floors")
            height = float(num_floors) * 3.5 if num_floors is not None else 10.0
        else:
            try:
                height = float(height)
            except (TypeError, ValueError):
                height = 10.0
        if not np.isfinite(height):
            height = 10.0
        height = max(height, 3.0)

        # GLB mesh — pass the full clipped Shapely polygon (holes included)
        # translated to domain-centred coordinates
        def _shift_ring(ring, dx, dy):
            a = np.array(ring.coords)
            a[:, 0] += dx
            a[:, 1] += dy
            return a

        glb_exterior = _shift_ring(clipped.exterior, -cx_m, -cy_m)
        glb_interiors = [_shift_ring(r, -cx_m, -cy_m) for r in clipped.interiors]
        glb_polygon = ShapelyPolygon(glb_exterior, glb_interiors)
        mesh = trimesh.creation.extrude_polygon(glb_polygon, height)
        meshes.append(mesh)

        # Voxelisation polygon — shift to [0, 2*dist] domain; preserve holes
        vox_exterior = _shift_ring(clipped.exterior, -cx_m + dist, -cy_m + dist)
        vox_interiors = [
            _shift_ring(r, -cx_m + dist, -cy_m + dist) for r in clipped.interiors
        ]
        polygon_shifted = ShapelyPolygon(vox_exterior, vox_interiors)
        buildings_data.append((polygon_shifted, height))

    scene = trimesh.Scene(meshes)
    scene.export(output_path)
    print(f"Exported {len(meshes)} buildings to {output_path}")

    # Generate aligned OSM map texture into geometry/assets/{preset_id}.png.
    # domain_x_wm / domain_y_wm are in Web Mercator pseudo-metres (= 2 * dist),
    # matching the ±dist shift applied to building coordinates above.
    if preset_id is not None:
        from pathlib import Path as _Path

        assets_dir = _Path(__file__).parent / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        png = fetch_map_png(lat, lon, 2.0 * dist, 2.0 * dist)
        map_path = assets_dir / f"{preset_id}.png"
        map_path.write_bytes(png)
        print(f"Saved map texture to {map_path}")

    # 5. Extend domain_z to provide clearance above tallest building
    max_building_height = max((h for _, h in buildings_data), default=10.0)
    domain_z = max(domain_z, max_building_height * 1.5, max_building_height + 50.0)
    print(f"Max building height: {max_building_height:.1f}m, domain_z: {domain_z:.1f}m")

    # 6. Voxelise footprints into occupancy grid
    Nx = round(2 * dist / pitch)
    Ny = Nx
    Nz = round(domain_z / pitch)
    occupancy = np.zeros((Nz, Ny, Nx), dtype=np.uint8)

    for polygon_shifted, height in buildings_data:
        minx, miny, maxx, maxy = polygon_shifted.bounds
        ix0 = max(0, int(minx / pitch))
        ix1 = min(Nx, int(np.ceil(maxx / pitch)))
        iy0 = max(0, int(miny / pitch))
        iy1 = min(Ny, int(np.ceil(maxy / pitch)))
        iz1 = min(Nz, int(np.ceil(height / pitch)))

        if ix0 >= ix1 or iy0 >= iy1 or iz1 <= 0:
            continue

        xs = (np.arange(ix0, ix1) + 0.5) * pitch  # voxel-centre X coords
        ys = (np.arange(iy0, iy1) + 0.5) * pitch  # voxel-centre Y coords
        XX, YY = np.meshgrid(xs, ys, indexing="xy")  # shape [Ny_local, Nx_local]
        mask = shapely.contains_xy(polygon_shifted, XX.ravel(), YY.ravel())
        mask = mask.reshape(XX.shape).astype(np.uint8)

        occupancy[:iz1, iy0:iy1, ix0:ix1] |= mask[np.newaxis, :, :]

    solid_count = int(occupancy.sum())
    print(
        f"Voxelised occupancy grid shape: {occupancy.shape}, solid voxels: {solid_count}"
    )

    with h5py.File(hdf5_path, "w") as f:
        ds = f.create_dataset("occupancy", data=occupancy, compression="gzip")
        ds.attrs["resolution"] = float(pitch)
        ds.attrs["pitch"] = float(pitch)
        ds.attrs["domain_size"] = np.array(
            [round(2 * dist), round(2 * dist), round(domain_z)], dtype=np.int32
        )
    print(f"Saved occupancy grid to {hdf5_path}")


if __name__ == "__main__":
    # export_preset_glb(
    #     lat=41.8827,
    #     lon=-87.6233,
    #     dist=500,
    #     output_path="../assets/presets/chicago.glb",
    #     preset_id="chicago",
    # )
    # export_preset_glb(
    #     lat=40.7682,
    #     lon=-73.9807,
    #     dist=500,
    #     output_path="../assets/presets/manhattan.glb",
    #     preset_id="manhattan",
    # )
    # export_preset_glb(
    #     lat=35.6938,
    #     lon=139.7034,
    #     dist=500,
    #     output_path="../assets/presets/tokyo.glb",
    #     preset_id="tokyo",
    # )
    # export_preset_glb(
    #     lat=48.8698,
    #     lon=2.3078,
    #     dist=500,
    #     output_path="../assets/presets/paris.glb",
    #     preset_id="paris",
    # )
    export_preset_glb(
        lat=31.2397,
        lon=121.4988,
        dist=500,
        output_path="../assets/presets/shanghai.glb",
        preset_id="shanghai",
    )
