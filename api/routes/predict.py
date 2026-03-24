import base64

import numpy as np
from fastapi import APIRouter, HTTPException

from models import PredictRequest
from simulation import PRESETS, generate

router = APIRouter()


@router.post("/predict")
def predict(req: PredictRequest) -> dict:
    if req.preset_id not in PRESETS:
        raise HTTPException(
            status_code=404, detail=f"Preset '{req.preset_id}' not found"
        )

    p = req.wind_params
    vel, comfort, domain = generate(
        req.preset_id,
        p.direction,
        p.speed,
        p.roughness,
    )

    # vel shape: [Nz, Ny, Nx, 3] — use float16 to halve payload size
    Nz, Ny, Nx, _ = vel.shape

    vel_b64 = base64.b64encode(vel.astype(np.float16).tobytes()).decode()
    comfort_b64 = base64.b64encode(comfort.astype(np.float16).tobytes()).decode()

    vel_flat = vel.reshape(-1)
    v_min = float(np.min(vel_flat))
    v_max = float(np.max(np.abs(vel_flat)))

    return {
        "velocityField": {
            "data": vel_b64,
            "shape": [Nx, Ny, Nz],
            "min": v_min,
            "max": v_max,
            "domain": domain,
        },
        "comfortMap": {
            "data": comfort_b64,
            "shape": [Nx, Ny],
        },
    }
