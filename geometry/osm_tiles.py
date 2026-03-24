"""
OSM tile fetching and stitching utilities.

Downloads tiles from OpenStreetMap, stitches them, and crops to the exact
domain bounding box expressed in Web Mercator pseudo-metres (EPSG:3857).

This script lives in the geometry pipeline so map images are produced at the
same time as GLB meshes.  All coordinates use the same Web Mercator projection
that overture_mesh.py uses for building geometry, ensuring pixel-perfect
alignment without any cos(lat) correction.
"""

import io
import math
import urllib.request
from pathlib import Path

from PIL import Image

TILE_SIZE = 256  # px per OSM tile
OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "WindSimulator/1.0"

# Web Mercator semi-circumference — same constant used by EPSG:3857
_WM_CIRC = 40_075_016.686  # metres (full equatorial circumference)


def _deg2tile_float(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Fractional tile (x, y) for a WGS84 lat/lon at the given zoom level.

    The tile formula is the standard OSM/Web-Mercator Slippy Map convention:
      tile_x is linear in WM X  (no cos factor)
      tile_y is linear in WM Y  (same scale as tile_x, because Mercator is conformal)
    """
    lat_r = math.radians(lat)
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _tile_size_wm(zoom: int) -> float:
    """Width of one tile in Web Mercator pseudo-metres.

    Tile coordinates are a linear function of WM (X, Y) with identical scale
    in both axes — the conformal property of Web Mercator means:

        d(tile_x) / d(WM_x) = d(tile_y) / d(WM_y) = 2^zoom / _WM_CIRC

    So one tile spans _WM_CIRC / 2^zoom WM metres in both X and Y.
    There is NO cos(lat) factor here.
    """
    return _WM_CIRC / (2**zoom)


def _fetch_tile(x: int, y: int, zoom: int, cache_dir: Path) -> Image.Image:
    cache_path = cache_dir / f"{zoom}_{x}_{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    url = OSM_URL.format(z=zoom, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()

    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.save(cache_path)
    return img


def fetch_map_png(
    center_lat: float,
    center_lon: float,
    domain_x_wm: float,
    domain_y_wm: float,
    zoom: int = 17,
    cache_dir: Path | None = None,
) -> bytes:
    """
    Return a PNG of the OSM map covering the given domain.

    Parameters
    ----------
    center_lat, center_lon
        WGS84 geographic centre — must be the SAME point used as the origin
        when the companion GLB was built (cx_m, cy_m in overture_mesh.py).
    domain_x_wm, domain_y_wm
        Domain width and height in **Web Mercator pseudo-metres** (EPSG:3857).
        For a preset built with dist=500, pass 1000.0 for both axes.
    zoom
        OSM zoom level.  17 gives ~306 WM-m per tile → ~836 px output for a
        1 km domain, which is a good balance of detail vs tile count.
    cache_dir
        Directory for individual tile PNGs and the stitched result.  Defaults
        to <this file's directory>/tile_cache/.

    Returns
    -------
    bytes
        PNG image data.  North is at the top.  One pixel represents
        (domain_x_wm / image_width) WM metres, matching the GLB coordinate
        system exactly.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "tile_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    stitch_name = (
        f"map_{center_lat:.5f}_{center_lon:.5f}"
        f"_{int(domain_x_wm)}x{int(domain_y_wm)}_z{zoom}.png"
    )
    stitch_path = cache_dir / stitch_name
    if stitch_path.exists():
        return stitch_path.read_bytes()

    tsw = _tile_size_wm(zoom)  # WM metres per tile

    # Fractional tile position of the domain centre
    fx, fy = _deg2tile_float(center_lat, center_lon, zoom)

    # Integer tile range with a 1-tile safety buffer on every side
    half_tx = math.ceil(domain_x_wm / tsw / 2) + 1
    half_ty = math.ceil(domain_y_wm / tsw / 2) + 1

    tx0 = int(fx) - half_tx
    ty0 = int(fy) - half_ty
    tx1 = int(fx) + half_tx
    ty1 = int(fy) + half_ty

    n_tiles_x = tx1 - tx0 + 1
    n_tiles_y = ty1 - ty0 + 1

    # Fetch and stitch
    canvas = Image.new("RGB", (n_tiles_x * TILE_SIZE, n_tiles_y * TILE_SIZE))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            tile = _fetch_tile(tx, ty, zoom, cache_dir)
            canvas.paste(tile, ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE))

    # Pixel-per-WM-metre scale (identical for X and Y — conformal projection)
    px_per_wm = TILE_SIZE / tsw

    # Centre of the domain in canvas pixel coordinates
    cx_px = (fx - tx0) * TILE_SIZE
    cy_px = (fy - ty0) * TILE_SIZE

    hw = domain_x_wm / 2 * px_per_wm
    hh = domain_y_wm / 2 * px_per_wm

    cropped = canvas.crop(
        (
            round(cx_px - hw),
            round(cy_px - hh),
            round(cx_px + hw),
            round(cy_px + hh),
        )
    )

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    stitch_path.write_bytes(png_bytes)
    return png_bytes
