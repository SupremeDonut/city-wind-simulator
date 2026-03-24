from pydantic import BaseModel


class WindParams(BaseModel):
    direction: float   # degrees, 0=north clockwise
    speed: float       # m/s
    roughness: float   # z0 metres


class PredictRequest(BaseModel):
    preset_id: str
    wind_params: WindParams
    resolution: float = 8.0  # desired output resolution in metres
    num_steps: int = 3000
    snapshot_interval: int = 200
