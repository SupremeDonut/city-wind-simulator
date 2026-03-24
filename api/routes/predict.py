import base64

import lz4.frame
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

    # vel shape: [Nz, Ny, Nx, 3] — float16 halves size, lz4 compresses 3-5x further
    Nz, Ny, Nx, _ = vel.shape

    vel_b64 = base64.b64encode(
        lz4.frame.compress(vel.astype(np.float16).tobytes(), compression_level=0)
    ).decode()
    comfort_b64 = base64.b64encode(
        lz4.frame.compress(comfort.astype(np.float16).tobytes(), compression_level=0)
    ).decode()

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
            "encoding": "lz4+float16",
        },
        "comfortMap": {
            "data": comfort_b64,
            "shape": [Nx, Ny],
            "encoding": "lz4+float16",
        },
    }
