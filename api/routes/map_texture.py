from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from simulation import PRESETS

router = APIRouter()

GEOMETRY_ASSETS = Path(__file__).parent.parent.parent / "geometry" / "assets"


@router.get("/presets/{preset_id}/map-texture")
def get_map_texture(preset_id: str) -> FileResponse:
    if preset_id not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
    path = GEOMETRY_ASSETS / f"{preset_id}.png"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Map texture not found — run the geometry pipeline first",
        )
    return FileResponse(str(path), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
