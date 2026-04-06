"""Hyperparameters and paths for the FNO surrogate model."""

from dataclasses import dataclass, field
from pathlib import Path

# Repo root: surrogate/ -> wind_sim/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "presets"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Preset geometries available for training
PRESET_IDS = ["chicago", "manhattan", "tokyo", "paris", "shanghai"]

# Fixed aerodynamic roughness length z0 (m) — urban/suburban default
DEFAULT_ROUGHNESS: float = 0.3


@dataclass
class CanonicalGrid:
    """Fixed tensor dimensions for the FNO (powers of 2 for FFT efficiency)."""

    depth: int = 32  # Z (vertical)
    height: int = 64  # Y (north-south)
    width: int = 64  # X (east-west)


@dataclass
class DataGenConfig:
    """Configuration for the batch LBM data generation pipeline."""

    preset_ids: list[str] = field(default_factory=lambda: list(PRESET_IDS))
    # Wind direction: 36 angles at 10-degree increments
    n_directions: int = 36
    # Wind speed: 8 values linearly spaced in [1, 15] m/s
    speeds: list[float] = field(
        default_factory=lambda: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    )
    # Roughness: single fixed value (urban/suburban default)
    roughnesses: list[float] = field(default_factory=lambda: [DEFAULT_ROUGHNESS])
    pitch: float = 8.0  # metres per cell (matches LBM default)
    num_steps: int = 500  # LBM iterations per sample
    grid: CanonicalGrid = field(default_factory=CanonicalGrid)


@dataclass
class ModelConfig:
    """FNO architecture hyperparameters."""

    in_channels: int = 8  # occ, sdf, sin/cos dir, speed, z/y/x coords
    out_channels: int = 3  # ux, uy, uz
    width: int = 32  # hidden channel width in Fourier layers
    n_layers: int = 4  # number of Fourier layers
    modes_z: int = 6  # Fourier modes retained in Z (out of 32)
    modes_y: int = 6  # Fourier modes retained in Y (out of 128)
    modes_x: int = 6  # Fourier modes retained in X (out of 128)
    projection_hidden: int = 128  # hidden dim in final projection MLP
    use_checkpoint: bool = (
        True  # gradient checkpointing on Fourier layers (saves ~4 GB VRAM)
    )
    grid: CanonicalGrid = field(default_factory=CanonicalGrid)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Optimiser
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Schedule
    epochs: int = 150
    warmup_epochs: int = 5
    # Batch
    batch_size: int = 4
    num_workers: int = 0  # Windows: shared memory file mapping fails with >0 workers
    # Loss weights
    lambda_div: float = 0.1
    pedestrian_weight: float = 2.0  # extra weight on z=0..2 layers
    # Early stopping
    patience: int = 15
    # Data split
    train_frac: float = 0.8
    val_frac: float = 0.1
    # test_frac = 1 - train_frac - val_frac
    # Mixed precision
    use_amp: bool = True
    # Speed options
    use_compile: bool = (
        False  # torch.compile — 10-30% faster, ~50-100 MB extra VRAM for compiled graph
    )
    cache_in_ram: bool = (
        False  # load full dataset into CPU RAM — eliminates HDF5 I/O; ~7 GB RAM per preset at 64×128×128
    )
    # Paths
    checkpoint_dir: Path = CHECKPOINT_DIR
    runs_dir: Path = RUNS_DIR
    data_dir: Path = DATA_DIR
