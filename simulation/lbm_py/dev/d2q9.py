import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# D2Q9 constants
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])  # opposite direction indices

CS2 = 1 / 3
TAU = 0.7  # gives kinematic viscosity ν = cs²(τ - 0.5) = 0.333 * 0.2 ≈ 0.067

NX, NY = 300, 100  # width x height
U_INLET = 0.1  # inlet velocity in lattice units (keep << cs = 0.577)


def equilibrium(rho, ux, uy):
    # rho: (NY, NX), ux/uy: (NY, NX) -> feq: (9, NY, NX)
    eu = EX[:, None, None] * ux[None] + EY[:, None, None] * uy[None]
    u2 = ux**2 + uy**2
    return (
        W[:, None, None]
        * rho[None]
        * (1 + eu / CS2 + eu**2 / (2 * CS2**2) - u2[None] / (2 * CS2))
    )


# --- Geometry: a rectangular building obstacle ---
solid = np.zeros((NY, NX), dtype=bool)
solid[30:70, 80:120] = True  # building block

# --- Initialise to uniform inlet flow ---
rho = np.ones((NY, NX))
ux = np.full((NY, NX), U_INLET)
uy = np.zeros((NY, NX))
ux[solid] = 0.0
uy[solid] = 0.0

f = equilibrium(rho, ux, uy)  # shape: (9, NY, NX)


def stream_and_bounce(f, solid):
    f_streamed = np.zeros_like(f)
    for v in range(9):
        # stream
        shifted = np.roll(np.roll(f[v], EX[v], axis=1), EY[v], axis=0)
        # bounce-back: where the destination is solid, reflect
        f_streamed[v] = np.where(solid, f[OPP[v]], shifted)
    return f_streamed


def apply_inlet(f, rho, ux_in, uy_in=0.0):
    # overwrite left column with equilibrium at inlet velocity
    rho_in = np.ones(NY)
    ux_col = np.full(NY, ux_in)
    uy_col = np.zeros(NY)
    feq_in = equilibrium(rho_in, ux_col, uy_col)  # (9, NY)
    f[:, :, 0] = feq_in[:, :, 0] if feq_in.ndim == 3 else feq_in
    return f


def apply_outlet(f):
    # zero-gradient: copy second-to-last column to last
    f[:, :, -1] = f[:, :, -2]
    return f


fig, ax = plt.subplots(figsize=(10, 4))
speed_plot = ax.imshow(
    np.zeros((NY, NX)), origin="lower", cmap="inferno", vmin=0, vmax=U_INLET * 2.0
)
plt.colorbar(speed_plot, ax=ax, label="speed")
plt.tight_layout()

frames: list[Image.Image] = []


def capture_frame(step: int) -> None:
    speed = np.sqrt(ux**2 + uy**2)
    speed_plot.set_data(speed)
    ax.set_title(f"step {step}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    print(f"step {step}, max speed: {speed.max():.4f}")


for step in range(3001):
    # Collision
    feq = equilibrium(rho, ux, uy)
    f += (feq - f) / TAU
    f[:, solid] = f[OPP][:, solid]  # zero velocity inside solid (re-zero)

    # Streaming + bounce-back
    f = stream_and_bounce(f, solid)

    # Boundary conditions
    apply_inlet(f, rho, U_INLET)
    apply_outlet(f)

    # Macroscopic variables
    rho = f.sum(axis=0)
    ux = (f * EX[:, None, None]).sum(axis=0) / rho
    uy = (f * EY[:, None, None]).sum(axis=0) / rho

    # enforce zero velocity in solid
    ux[solid] = 0.0
    uy[solid] = 0.0

    if step % 50 == 0:
        capture_frame(step)

plt.close(fig)

frames[0].save(
    "simulation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=200,  # ms per frame
    loop=0,
)
print(f"Saved simulation.gif ({len(frames)} frames)")
