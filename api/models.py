from pydantic import BaseModel

# Fixed aerodynamic roughness length z0 (m) used throughout the system
DEFAULT_ROUGHNESS: float = 0.3


class WindParams(BaseModel):
    direction: float   # degrees, 0=north clockwise
    speed: float       # m/s


class PredictRequest(BaseModel):
    preset_id: str
    wind_params: WindParams
    resolution: float = 8.0  # desired output resolution in metres
    num_steps: int = 3000
    snapshot_interval: int = 200
