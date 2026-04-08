# Urban Wind Flow Simulator

A GPU-accelerated urban wind simulation platform combining Lattice Boltzmann fluid dynamics, a deep learning surrogate, and real-time 3D visualization. Select a preset city neighborhood, choose wind direction and speed, and get an interactive visualization of wind flow — 3D streamlines and a pedestrian comfort map at street level.

---

## How It Works

The system operates in two modes:

- **Surrogate mode** — a trained Fourier Neural Operator predicts the velocity field in ~50 ms for interactive exploration.
- **Full simulation** — the CUDA LBM solver runs on the GPU for high-fidelity results, streaming progress over WebSocket.

### Preset City Neighborhoods

| Preset             | Description                                        |
| ------------------ | -------------------------------------------------- |
| Chicago Loop       | Dense urban canyon grid                             |
| Manhattan Midtown  | High-rise corridor with street canyons              |
| Tokyo Shinjuku     | Irregular dense urban fabric                        |
| Haussmann Paris    | Radial boulevard layout                             |
| Shanghai Lujiazui  | Clustered supertalls with open river exposure        |

Each preset ships as a pre-computed voxel grid (HDF5) and 3D mesh (GLB) derived from Overture Maps building data.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Frontend (Browser)               │
│    Preset Selector │ 3D Streamlines │ Comfort Map │
└────────────────────┬─────────────────────────────┘
                     │ HTTP / WebSocket
┌────────────────────▼─────────────────────────────┐
│               Backend API (FastAPI)               │
└────────┬──────────────────┬──────────────────────┘
         │                  │
┌────────▼────────┐  ┌──────▼──────────┐
│  LBM Simulator  │  │  ML Surrogate   │
│  (CUDA C++)     │  │  (PyTorch/FNO)  │
└─────────────────┘  └─────────────────┘
```

| Layer             | Language   | Key Libraries                       |
| ----------------- | ---------- | ----------------------------------- |
| LBM Simulator     | CUDA C++   | cnpy, zlib                          |
| Geometry Pipeline | Python     | Overture Maps, Trimesh, Shapely     |
| ML Surrogate      | Python     | PyTorch, TensorBoard                |
| Backend API       | Python     | FastAPI, uvicorn, lz4               |
| Frontend          | TypeScript | React, Three.js, R3F, deck.gl       |

---

## Running

### 1. Build the CUDA Solver

Prerequisites: CUDA Toolkit 12.6+, MSVC 2022, CMake 3.24+, Ninja.

```bash
cd simulation/src
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
```

### 2. Start the Backend

```bash
cd api
uv run uvicorn main:app --reload
```

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

### Running the LBM Solver Directly

```bash
cd simulation/lbm_py
uv run python -m lbm.run_cuda assets/presets/chicago.h5 \
    --angle 0 --speed 2 --roughness 0.5 --pitch 8 --steps 500
```

Outputs `cuda_wind_field.npz` with keys `ux, uy, uz, solid, dx`.

---

## API Endpoints

| Method | Path                                 | Description                                    |
| ------ | ------------------------------------ | ---------------------------------------------- |
| GET    | `/presets`                           | List available city presets                     |
| GET    | `/presets/{preset_id}/geometry`      | GLB mesh for a preset                          |
| GET    | `/presets/{preset_id}/map-texture`   | OSM map texture (PNG)                          |
| POST   | `/predict`                           | Surrogate inference (lz4+float16 response)     |
| POST   | `/simulate`                          | Enqueue full LBM job, returns job_id           |
| GET    | `/results/{job_id}/velocity`         | Velocity field (raw float32 bytes)             |
| GET    | `/results/{job_id}/comfort-map`      | Pedestrian comfort map (raw float32 bytes)     |
| WS     | `/ws/simulation/{job_id}`            | Real-time progress + snapshots during LBM run  |

---

## Project Structure

```
wind_sim/
├── api/                  # FastAPI backend
│   ├── main.py           # App setup, CORS, middleware
│   ├── models.py         # Pydantic request/response models
│   ├── simulation.py     # Surrogate loading + procedural fallback
│   ├── routes/           # Endpoint handlers
│   └── results/          # Job output cache (.npz)
├── frontend/             # React + Three.js + Tailwind
│   └── src/
│       ├── components/   # PresetSelector, WindControls, Viewport3D, ComfortMap
│       ├── store/        # Zustand state management
│       └── api/          # API client with lz4 decompression
├── simulation/
│   ├── src/              # CUDA C++ LBM solver (D3Q19, BGK collision)
│   └── lbm_py/           # Python wrapper (preprocessing, subprocess management)
├── surrogate/            # FNO surrogate model
│   ├── surrogate/
│   │   ├── model/        # FNO3d architecture (4 Fourier layers, 32 width)
│   │   ├── training/     # Training loop, loss functions
│   │   ├── inference/    # Checkpoint loading, caching, prediction
│   │   └── data/         # Dataset loading, LBM data generation
│   ├── checkpoints/      # Model weights
│   └── data/             # Training data (HDF5 per preset)
├── geometry/             # Overture Maps → voxel grid + GLB mesh pipeline
└── assets/presets/       # Pre-computed .glb meshes and .h5 voxel grids
```

---

## Visualization

**3D Wind Particles** — 16,384 GPU-advected tracer particles rendered as color-coded point sprites (blue → red by speed). Runs entirely on the GPU via a GPGPU ping-pong scheme with zero CPU particle state.

**Pedestrian Comfort Map** — velocity magnitude at pedestrian height classified by Lawson criteria:

```
< 2.5 m/s  →  Comfortable for sitting    (green)
2.5-4 m/s  →  Comfortable for standing   (yellow-green)
4-6 m/s    →  Acceptable for walking     (yellow)
6-8 m/s    →  Uncomfortable              (orange)
> 8 m/s    →  Dangerous                  (red)
```
