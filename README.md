# Urban Wind Flow Simulator

A GPU-accelerated urban wind simulation platform combining Lattice Boltzmann fluid dynamics, a deep learning surrogate, and real-time 3D visualization. The user selects a preset city neighborhood, chooses a wind direction and speed, and the system delivers an interactive visualization of wind flow through the built environment — streamlines in 3D and a pedestrian comfort map at street level. A trained Fourier Neural Operator surrogate responds quickly for interactive exploration; the full LBM solver runs on demand for high-fidelity results.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [LBM Simulator (CUDA C++)](#lbm-simulator-cuda-c)
5. [Geometry Pipeline (Python)](#geometry-pipeline-python)
6. [ML Surrogate (PyTorch)](#ml-surrogate-pytorch)
7. [Backend API (FastAPI)](#backend-api-fastapi)
8. [Frontend (React + Three.js)](#frontend-react--threejs)
9. [Visualization](#visualization)
10. [Physics Reference](#physics-reference)
11. [Full Stack Summary](#full-stack-summary)

---

## Project Overview

Wind at street level affects pedestrian comfort, pollutant dispersal, and building energy loads. Existing simulation tools are too slow for interactive use — full CFD runs take hours. This project bridges that gap with a GPU-accelerated LBM solver backed by a real-time 3D frontend.

### Dual-Mode Operation

```
Surrogate mode:  User selects preset → POST /predict → FNO inference (50ms) → visualization updates

Full simulation: User triggers LBM  → asyncio background task → run_cuda.py subprocess → lbm_cuda.exe on GPU
                                      → WebSocket streams progress live
                                      → final field replaces surrogate on completion
```

### Preset City Neighborhoods

Rather than a freeform map editor, the frontend offers a curated set of pre-voxelized city configurations, each capturing a distinct urban morphology:

| Preset          | Description                                                             |
| --------------- | ----------------------------------------------------------------------- |
| Manhattan Grid  | Dense uniform street grid, tall towers, strong channeling effects       |
| Chicago Loop    | Mixed heights, river boundary, exposed lakefront fetch                  |
| Haussmann Paris | Radial boulevard layout, uniform cornice height, courtyard blocks       |
| Tokyo Shinjuku  | Irregular dense urban fabric, tower clusters, narrow pedestrian streets |

Each preset ships as a pre-computed voxel grid and GLB mesh derived from Overture Maps. The user adjusts wind direction (compass rose), reference wind speed, and terrain roughness class — these feed directly into the LBM inlet boundary condition.

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
│     Job orchestration, geometry serving,          │
│     result streaming                              │
└────────┬──────────────────┬──────────────────────┘
         │                  │
┌────────▼────────┐  ┌──────▼──────────┐
│  LBM Simulator  │  │  ML Surrogate   │
│  (CUDA C++)     │  │  (PyTorch/FNO)  │
└─────────────────┘  └─────────────────┘
         │                  │
┌────────▼──────────────────▼──────────┐
│             Storage Layer             │
│  HDF5 (fields) │ NumPy .npz │ in-mem  │
└──────────────────────────────────────┘
```

---

## LBM Simulator (CUDA C++)

LBM simulates fluid by tracking the statistical distribution of particle velocities on a lattice rather than solving Navier-Stokes directly. Each node updates independently — collision and streaming are both local operations with no global dependencies — making it exceptionally well-suited to GPU parallelism.

### D3Q19 Lattice

The standard 3D stencil: 19 discrete velocity directions (1 rest, 6 face neighbors, 12 edge neighbors), each carrying a population value f_i. Macroscopic density and velocity are recovered by moment sums over all populations at each node.

### BGK Collision

Populations relax toward a local equilibrium at rate 1/τ, where τ encodes kinematic viscosity. The equilibrium is the second-order Maxwell-Boltzmann distribution. MRT (Multiple Relaxation Time) collision is a drop-in replacement that improves stability at higher Reynolds numbers by relaxing each moment independently.

### Boundary Conditions

**Buildings** use bounce-back: populations streaming into a solid node reflect back in the opposite direction, producing a no-slip wall. Interpolated bounce-back (Bouzidi) gives second-order accuracy at the exact wall position.

**Inlet** imposes an atmospheric boundary layer velocity profile — a logarithmic wind profile u(z) = u_ref · ln(z/z0) / ln(z_ref/z0) — matched to the reference wind speed at 10 m height.

**Outlet** uses zero-gradient extrapolation. Top and side faces use free-slip.

### Performance Notes

Memory layout uses Structure of Arrays (SoA) — `f[q * N + linear_idx]` — so all threads in a warp access the same population array with stride-1 (fully coalesced). Two-lattice (A/B pointer swap) provides race-free streaming. Thread block is `(16, 8, 2)` = 256 threads.

### Building

Prerequisites: CUDA Toolkit 12.6+, MSVC 2022, CMake 3.24+, Ninja.

```bash
# Open a Visual Studio Developer Command Prompt (or PowerShell with VS dev shell)
cd simulation/src
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
# Produces: build/Release/lbm_cuda.exe
```

zlib is fetched automatically via CMake FetchContent if not found on the system. The `build.bat` helper script sets up the VS environment and runs both commands.

### Python Wrapper

The CUDA solver is invoked via subprocess from `simulation/lbm_py/lbm/run_cuda.py`. The Python side handles HDF5 loading, voxel rotation (wind direction), downsampling, and input/output `.npz` serialisation. Progress is reported via stdout `PROGRESS <float>` lines.

The CUDA executable path defaults to `simulation/src/build/Release/lbm_cuda.exe` and can be overridden with the `LBM_EXE` environment variable.

```bash
cd simulation/lbm_py
uv run python -m lbm.run_cuda assets/presets/chicago.h5 \
    --angle 0 --speed 2 --roughness 0.5 --pitch 8 --steps 500
```

| Flag          | Default | Description                               |
| ------------- | ------- | ----------------------------------------- |
| `--angle`     | 0       | Wind direction in degrees (0 = north, CW) |
| `--speed`     | 2.0     | Reference wind speed at 10 m height (m/s) |
| `--roughness` | 0.5     | Aerodynamic roughness length z0 (m)       |
| `--pitch`     | 8.0     | Target grid spacing in metres             |
| `--steps`     | 500     | Number of LBM iterations                  |

Outputs `cuda_wind_field.npz` with keys `ux, uy, uz, solid, dx`.

### Preprocessing

Wind direction is implemented by rotating the voxel grid before simulation (the inlet always faces +Y). Downsampling uses block-max pooling so any voxel that is solid makes the entire coarsened block solid — conservative for preserving building geometry.

```
Native 2 m pitch → --pitch 8 → factor 4 → Chicago 500×500×100 → 125×125×25 (390 K cells)
```

### Benchmarking

```bash
cd simulation/lbm_py
uv run python dev/benchmark.py --preset chicago --pitch 8 --steps 500
```

Runs the CUDA and Python solvers on the same preprocessed input and prints a comparison table with wall-clock time, throughput (MLUPS), and max absolute velocity difference per component.

---

## Geometry Pipeline (Python)

Each preset is processed once offline and cached as a voxel grid:

```
Overture Maps building footprints + heights
  → coordinate transform (WGS84 → Web Mercator)    [pyproj]
  → footprint cleaning and clipping                 [Shapely]
  → 3D mesh extrusion → GLB export                 [Trimesh]
  → voxelization → uint8 occupancy grid (Nz×Ny×Nx)
  → saved as .h5
  → OSM map texture stitched and saved as .png      [osm_tiles.py]
```

Voxel grids and GLB meshes ship pre-computed with the project under `assets/presets/`. The pipeline only runs when adding a new preset:

```bash
cd geometry
uv run python overture_mesh.py
```

**Key libraries:** OSMnx, Shapely, Trimesh, pyproj, NumPy, h5py, Pillow.

---

## ML Surrogate (PyTorch)

### Learning Problem

Given a voxel occupancy grid G and wind parameters (speed, direction, roughness), predict the steady-state velocity field U ∈ ℝ^(3×Nx×Ny×Nz).

### Fourier Neural Operator (FNO)

FNO operates in Fourier space — at each layer the input is FFT'd, multiplied by learned complex weights in the low-frequency modes, and inverse-FFT'd. A pointwise linear transform is added in real space. This gives global receptive field from the first layer and resolution invariance (train at 128³, infer at 256³).

### Physics-Informed Loss

```
L = L_data + λ_div · ||∇·u||² + λ_solid · ||u · mask_solid||²
```

The divergence penalty enforces incompressibility; the solid penalty enforces zero velocity inside buildings.

### Training Data

An automated pipeline runs the LBM simulator on procedurally generated city configurations and saves (geometry, wind params, velocity field) pairs to HDF5. Target: 2000–5000 pairs at 128³. Wind direction augmentation (8 compass rotations per geometry) multiplies effective dataset size by 8.

### Uncertainty

An ensemble of 5–10 models provides per-voxel uncertainty estimates, surfaced in the frontend as reduced streamline opacity in low-confidence regions.

---

## Backend API (FastAPI)

### Key Endpoints

| Method | Path                        | Description                                           |
| ------ | --------------------------- | ----------------------------------------------------- |
| GET    | `/presets`                  | List available city presets                           |
| GET    | `/presets/{id}/geometry`    | GLB mesh for a preset                                 |
| POST   | `/predict`                  | Surrogate inference — returns velocity field in ~50ms |
| POST   | `/simulate`                 | Enqueue full LBM job, returns job_id                  |
| GET    | `/results/{id}/velocity`    | Full velocity field (compressed bytes)                |
| GET    | `/results/{id}/comfort-map` | Pedestrian comfort 2D array                           |
| WS     | `/ws/simulation/{id}`       | Real-time progress during LBM run                     |

Long-running LBM jobs run as asyncio background tasks. The API returns a job_id immediately; the frontend opens a WebSocket for progress updates. Job state is held in an in-memory dict and results are written to disk as NumPy `.npz` files under `api/results/`.

**Key libraries:** FastAPI, uvicorn, Pydantic, h5py, NumPy.

### Running

```bash
cd api
uv run uvicorn main:app --reload
```

---

## Frontend (React + Three.js)

### Views

**Preset Selector** — a grid of neighborhood cards with thumbnail, name, and urban morphology description. User picks a preset, sets wind direction via a compass rose and speed via a slider, then hits Run.

**3D Particle Viewport** — Three.js scene showing city geometry and the wind field as 16 384 GPU-simulated particles, colored by local velocity magnitude (blue → green → yellow → red). A 1 m × 1 m reference grid at ground level provides scale context.

**Comfort Map** — 2D top-down view with a continuous color field at pedestrian height showing Lawson comfort categories.

### GPGPU Particle System

All particle simulation runs on the GPU via a ping-pong render target scheme — the main thread issues a single draw call per frame and never touches particle state.

**Data structures**

| Resource             | Format         | Size               | Purpose                                  |
| -------------------- | -------------- | ------------------ | ---------------------------------------- |
| Position texture A/B | RGBA32F RT     | 128 × 128          | Ping-pong particle state (x, y, z, age)  |
| Velocity texture     | RGBA32F 3D     | Nx/4 × Ny/4 × Nz/4 | Subsampled wind field for GPU sampling   |
| Points geometry      | vec2 attribute | 16 384 vertices    | Per-particle texel UV — no position data |

**Per-frame pipeline**

```
Compute pass  — fragment shader reads position RT, samples velocity 3D texture,
                advances each particle, respawns expired/OOB ones → writes next RT
Render pass   — vertex shader reads position RT via aUV, converts grid → world coords,
                samples velocity texture for colour → gl_PointSize = 5px sprite
```

**Compute fragment shader (key logic)**

```glsl
vec3 normPos = clamp(gpos / uGridSize, 0.0, 1.0);
vec3 vel     = texture(uVelocity, normPos).rgb;

if (expired || outOfBounds || stalled) {
  gpos = uniformRandom(uv, uNow) * uGridSize;  // respawn
  age  = 0.0;
} else {
  gpos += vel * uDelta * STEP_SCALE;
  age  += uDelta;
}
```

The velocity field is subsampled at stride 4 before upload (500 × 500 × 100 → 125 × 125 × 25, ~6 MB instead of ~400 MB), keeping GPU memory flat while preserving all large-scale flow features.

**Key libraries:** React, TypeScript, Three.js, React Three Fiber, Zustand, Tailwind CSS.

### Running

```bash
cd frontend
npm install
npm run dev
```

---

## Visualization

### Wind Particles (GPGPU)

16 384 massless tracer particles advected in real time entirely on the GPU. Each particle stores its grid-space position and age in a floating-point render target; a compute fragment shader advances every particle each frame by sampling the 3D velocity texture and integrating with a forward-Euler step. Particles are respawned at uniformly random positions when they expire, leave the domain, or stall in a zero-velocity cell. Rendered as additive soft-edged point sprites (5 px) colored by local wind speed on a blue → green → yellow → red scale. They reveal channeling through street canyons, recirculation zones in building wakes, and corner acceleration — all with zero CPU overhead beyond a single `postMessage` per frame.

### Pedestrian Comfort Map (Lawson Criteria)

Time-averaged velocity magnitude sampled at z = 1.5m, classified by the Lawson wind comfort criteria:

```
< 2.5 m/s   → Comfortable for sitting       (green)
2.5–4 m/s   → Comfortable for standing      (yellow-green)
4–6 m/s     → Acceptable for walking        (yellow)
6–8 m/s     → Uncomfortable                 (orange)
> 8 m/s     → Dangerous                     (red)
```

---

## Physics Reference

```
Reynolds number:  Re = U·L / ν    Urban flow: Re ~ 10⁴–10⁷ (turbulent)
Mach number:      Ma = U / cs     Keep Ma < 0.1 for incompressibility
Strouhal number:  St = f·L / U    Vortex shedding: St ≈ 0.2 for bluff bodies
```

LBM recovers the incompressible Navier-Stokes equations in the limit Ma → 0 via Chapman-Enskog expansion. At urban Reynolds numbers, DNS is infeasible; the Smagorinsky LES subgrid model augments effective viscosity (νt = (Cs·Δ)²·|S|) to represent unresolved small-scale turbulence.

---

## Full Stack Summary

| Layer             | Language   | Key Libraries                      | Role                        |
| ----------------- | ---------- | ---------------------------------- | --------------------------- |
| LBM Simulator     | CUDA C++   | cnpy, zlib                         | GPU fluid simulation        |
| Geometry Pipeline | Python     | Overture Maps, Trimesh, Shapely    | Maps → voxel grid (offline) |
| ML Surrogate      | Python     | PyTorch, neuraloperator, Lightning | FNO fast inference          |
| Backend API       | Python     | FastAPI, uvicorn                   | Orchestration, streaming    |
| Storage           | —          | HDF5, NumPy .npz, in-memory dict   | Fields, job results, state  |
| Frontend          | TypeScript | React, Three.js, R3F               | Streamlines, comfort map    |

### Data Flow

```
User selects preset + wind params
  → POST /predict → FNO inference (50ms) → velocity field
  → velocity field subsampled → uploaded as RGBA32F 3D texture to GPU
  → GPGPU particle system initialised (16 384 particles, 128×128 RT)
  → comfort map extracted at z=1.5m → overlay
  → full visualization update in ~200ms

User triggers full LBM run
  → POST /simulate → asyncio background task → lbm.run_cuda subprocess → lbm_cuda.exe on GPU
  → WebSocket streams progress (parsed from stdout PROGRESS lines)
  → on completion: velocity field replaces surrogate prediction
```

### Infrastructure

Runs locally: FastAPI server (`uvicorn`) and frontend dev server (`vite`) started directly.

---

## References

- Succi, S. (2001). _The Lattice Boltzmann Equation for Fluid Dynamics and Beyond_. Oxford University Press.
- Li, Z. et al. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. _ICLR 2021_.
- Lawson, T.V. (1990). _Building Aerodynamics_. Imperial College Press.
- Martinuzzi & Tropea (1993). Flow around surface-mounted prismatic obstacles. _Journal of Fluids Engineering_.
- Klein et al. (2003). Digital filter based generation of inflow data for LES. _Journal of Computational Physics_.
