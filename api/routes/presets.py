from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from simulation import PRESETS

router = APIRouter()

ASSETS = Path(__file__).parent.parent.parent / "assets" / "presets"
BASE_URL = "http://localhost:8000"


@router.get("/presets")
def list_presets() -> list[dict]:
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "description": p["description"],
            "glbPath": f"{BASE_URL}/presets/{p['id']}/geometry",
            "domainSize": p["domainSize"],
            "resolution": p["resolution"],
        }
        for p in PRESETS.values()
    ]


@router.get("/presets/{preset_id}/geometry")
def get_geometry(preset_id: str) -> FileResponse:
    if preset_id not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
    path = ASSETS / f"{preset_id}.glb"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Geometry file not found on disk")
    return FileResponse(str(path), media_type="model/gltf-binary")
