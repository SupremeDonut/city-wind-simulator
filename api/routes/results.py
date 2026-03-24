from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from jobs import JobStatus, store

router = APIRouter()

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _load_npz(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Job is {job.status}, not done yet")
    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return dict(np.load(str(path)))


@router.get("/results/{job_id}/velocity")
def get_velocity(job_id: str) -> Response:
    data = _load_npz(job_id)
    vel: np.ndarray = data["vel"]        # [Nz, Ny, Nx, 3]
    domain: np.ndarray = data["domain"]  # [domX, domY, domZ]

    Nz, Ny, Nx, _ = vel.shape
    vel_f32 = vel.astype(np.float32)
    v_min = float(np.min(vel_f32))
    v_max = float(np.max(np.abs(vel_f32)))

    dom = domain.tolist()

    return Response(
        content=vel_f32.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Shape": f"{Nx},{Ny},{Nz}",
            "X-Domain": f"{dom[0]},{dom[1]},{dom[2]}",
            "X-Min": str(v_min),
            "X-Max": str(v_max),
            "Access-Control-Expose-Headers": "X-Shape,X-Domain,X-Min,X-Max",
        },
    )


@router.get("/results/{job_id}/comfort-map")
def get_comfort_map(job_id: str) -> Response:
    data = _load_npz(job_id)
    comfort: np.ndarray = data["comfort"]  # [Ny, Nx]

    Ny, Nx = comfort.shape
    comfort_f32 = comfort.astype(np.float32)

    return Response(
        content=comfort_f32.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Shape": f"{Nx},{Ny}",
            "Access-Control-Expose-Headers": "X-Shape",
        },
    )
