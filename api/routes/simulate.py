import asyncio
import base64
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
import h5py

from jobs import Job, JobStatus, store
from models import PredictRequest
from simulation import PRESETS, generate, _downsample, _downsample_2d

logger = logging.getLogger(__name__)

# Add simulation package to path so `lbm` is importable even outside uv
_SIM_LBM_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "simulation" / "lbm_py"
)
if _SIM_LBM_DIR not in sys.path:
    sys.path.insert(0, _SIM_LBM_DIR)

from lbm.run_cuda import run_cuda_simulation

router = APIRouter()

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _build_snapshot_msg(
    step: int, vel: np.ndarray, comfort: np.ndarray, domain: list
) -> dict:
    """Build a WebSocket snapshot message matching the predict response format."""
    Nz, Ny, Nx, _ = vel.shape
    vel_f16 = vel.astype(np.float16)
    comfort_f16 = comfort.astype(np.float16)

    vel_b64 = base64.b64encode(vel_f16.tobytes()).decode()
    comfort_b64 = base64.b64encode(comfort_f16.tobytes()).decode()

    vel_flat = vel.reshape(-1)
    v_min = float(np.min(vel_flat))
    v_max = float(np.max(np.abs(vel_flat)))

    return {
        "type": "snapshot",
        "step": int(step),
        "velocityField": {
            "data": vel_b64,
            "shape": [int(Nx), int(Ny), int(Nz)],
            "min": v_min,
            "max": v_max,
            "domain": [float(d) for d in domain],
        },
        "comfortMap": {
            "data": comfort_b64,
            "shape": [int(Nx), int(Ny)],
        },
    }


async def _run_job(
    job: Job,
    preset_id: str,
    direction: float,
    speed: float,
    roughness: float,
    resolution: float,
    num_steps: int = 3000,
    snapshot_interval: int = 200,
) -> None:
    try:
        loop = asyncio.get_event_loop()

        def progress_cb(val: float):
            job.progress = val
            # Schedule WebSocket broadcast on the event loop
            for q in list(job.subscribers):
                asyncio.run_coroutine_threadsafe(
                    q.put({"type": "progress", "value": val}), loop
                )

        # Retrieve domain for snapshot messages

        _repo_root = Path(__file__).resolve().parent.parent.parent
        voxel_path = _repo_root / "assets" / "presets" / f"{preset_id}.h5"
        with h5py.File(str(voxel_path), "r") as hf:
            snap_domain = list(hf["occupancy"].attrs["domain_size"])

        def snapshot_cb(step: int, vel: np.ndarray, comfort: np.ndarray):
            ds_vel = _downsample(vel, 2)
            ds_comfort = _downsample_2d(comfort, 2)
            msg = _build_snapshot_msg(step, ds_vel, ds_comfort, snap_domain)
            for q in list(job.subscribers):
                asyncio.run_coroutine_threadsafe(q.put(msg), loop)

        vel, comfort, domain = await loop.run_in_executor(
            None,
            run_cuda_simulation,
            preset_id,
            direction,
            speed,
            roughness,
            resolution,
            num_steps,
            progress_cb,
            snapshot_interval,
            snapshot_cb,
        )

        result_path = RESULTS_DIR / f"{job.id}.npz"
        np.savez(str(result_path), vel=vel, comfort=comfort, domain=domain)
        job.result_path = str(result_path)
        job.status = JobStatus.DONE

        done_msg = _build_snapshot_msg(num_steps, vel, comfort, domain)
        done_msg["type"] = "done"
        for q in list(job.subscribers):
            await q.put(done_msg)

    except Exception as exc:
        job.status = JobStatus.ERROR
        job.error = str(exc)
        err_msg = {"type": "error", "message": str(exc)}
        for q in list(job.subscribers):
            await q.put(err_msg)


@router.post("/simulate")
async def start_simulation(req: PredictRequest) -> dict:
    if req.preset_id not in PRESETS:
        raise HTTPException(
            status_code=404, detail=f"Preset '{req.preset_id}' not found"
        )

    job_id = str(uuid.uuid4())
    job = Job(id=job_id)
    store[job_id] = job

    p = req.wind_params
    asyncio.create_task(
        _run_job(
            job,
            req.preset_id,
            p.direction,
            p.speed,
            p.roughness,
            req.resolution,
            req.num_steps,
            req.snapshot_interval,
        )
    )

    return {"job_id": job_id}


@router.websocket("/ws/simulation/{job_id}")
async def ws_simulation(websocket: WebSocket, job_id: str) -> None:
    job = store.get(job_id)
    if job is None:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    queue = asyncio.Queue()
    job.subscribers.append(queue)

    async def _ping_loop() -> None:
        while True:
            await asyncio.sleep(20)
            await websocket.send_json({"type": "ping"})

    ping_task: asyncio.Task | None = None
    try:
        # If job already done/errored, send final state immediately
        if job.status == JobStatus.DONE:
            await websocket.send_json({"type": "done"})
            return
        if job.status == JobStatus.ERROR:
            await websocket.send_json(
                {"type": "error", "message": job.error or "unknown error"}
            )
            return

        ping_task = asyncio.create_task(_ping_loop())
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg["type"] in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if ping_task is not None:
            ping_task.cancel()
        if queue in job.subscribers:
            job.subscribers.remove(queue)
