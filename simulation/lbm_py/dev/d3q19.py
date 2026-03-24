import io
import math

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# D3Q19 lattice constants
EX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0])
EY = np.array([0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1])
EZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1])
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)

CS2 = 1 / 3
TAU = 0.55


def log_wind_profile(z: float, z_ref: float, u_ref: float, z0: float) -> float:
    """
    Returns wind speed at height z using the logarithmic wind profile.

    z     : height above ground (m)
    z_ref : reference height (m) — typically 10 m
    u_ref : wind speed at z_ref (m/s)
    z0    : roughness length (m)
    """
    if z <= z0:
        return 0.0
    return u_ref * math.log(z / z0) / math.log(z_ref / z0)


def equilibrium(rho, ux, uy, uz):
    # rho: (NZ, NY, NX), ux/uy/uz: (NZ, NY, NX) -> feq: (19, NZ, NY, NX)
    eu = (
        EX[:, None, None, None] * ux[None]
        + EY[:, None, None, None] * uy[None]
        + EZ[:, None, None, None] * uz[None]
    )
    u2 = ux**2 + uy**2 + uz**2
    return (
        W[:, None, None, None]
        * rho[None]
        * (1 + eu / CS2 + eu**2 / (2 * CS2**2) - u2[None] / (2 * CS2))
    )


def stream_and_bounce(f, solid):
    f_new = np.zeros_like(f)
    for v in range(19):
        # stream: roll along x (axis=2), y (axis=1), z (axis=0) in the NZ,NY,NX sub-array
        shifted = np.roll(
            np.roll(np.roll(f[v], EX[v], axis=2), EY[v], axis=1), EZ[v], axis=0
        )
        # bounce-back: where destination is solid, reflect
        f_new[v] = np.where(solid, f[OPP[v]], shifted)
    return f_new


def apply_inlet(f, u_profile, NZ, NX):
    # u_profile: (NZ,) — log-law wind speed per height level
    rho_in = np.ones((NZ, NX))
    ux_in = np.zeros((NZ, NX))
    uy_in = np.tile(u_profile[:, None], (1, NX))   # wind now +Y
    uz_in = np.zeros((NZ, NX))
    # compute equilibrium for the XZ inlet face (y=0)
    eu = (
        EX[:, None, None] * ux_in[None]
        + EY[:, None, None] * uy_in[None]
        + EZ[:, None, None] * uz_in[None]
    )
    u2 = ux_in**2 + uy_in**2 + uz_in**2
    feq_in = (
        W[:, None, None]
        * rho_in[None]
        * (1 + eu / CS2 + eu**2 / (2 * CS2**2) - u2[None] / (2 * CS2))
    )
    f[:, :, 0, :] = feq_in   # XZ face (y=0)


def apply_outlet(f):
    f[:, :, -1, :] = f[:, :, -2, :]   # y outlet


def run_simulation(voxel_path: str, wind_speed: float, roughness: float):
    # --- Load geometry ---
    with h5py.File(voxel_path, "r") as hf:
        occ = hf["occupancy"][::5, ::5, ::5]  # downsample 5x
        domain_size = hf["occupancy"].attrs["domain_size"]
    solid = occ > 0
    NZ, NY, NX = solid.shape
    print(f"Grid: {NX}x{NY}x{NZ} = {NX*NY*NZ} cells")

    # Add ground floor
    solid[0, :, :] = True

    # --- Convert physical units to lattice units ---
    domain_z = domain_size[2]
    dx = domain_z / NZ  # physical metres per grid cell

    # Lattice conversion: choose u_ref_lattice to keep Mach number safe
    u_ref_lattice = 0.06
    # dt/dx ratio implied by this choice
    dt_over_dx = u_ref_lattice / wind_speed

    z0_lattice = roughness / dx  # roughness in grid cells
    z_ref_lattice = 10.0 / dx  # 10 m reference height in grid cells

    # Build inlet profile in lattice units (heights in grid cells)
    u_profile = np.array(
        [
            log_wind_profile(k + 0.5, z_ref_lattice, u_ref_lattice, z0_lattice)
            for k in range(NZ)
        ]
    )

    # --- Initialise ---
    rho = np.ones((NZ, NY, NX))
    ux = np.zeros((NZ, NY, NX))
    uy = np.tile(u_profile[:, None, None], (1, NY, NX))
    uz = np.zeros((NZ, NY, NX))
    uy[solid] = 0.0

    f = equilibrium(rho, ux, uy, uz)

    # --- Visualization setup (2-panel) ---
    fig, (ax_plan, ax_elev) = plt.subplots(1, 2, figsize=(14, 5))

    vmax = u_ref_lattice * 2.5
    plan_img = ax_plan.imshow(
        np.zeros((NY, NX)), origin="lower", cmap="inferno", vmin=0, vmax=vmax
    )
    ax_plan.set_title("Plan view (z=2)")
    ax_plan.set_xlabel("x")
    ax_plan.set_ylabel("y")
    plt.colorbar(plan_img, ax=ax_plan, label="speed")

    elev_img = ax_elev.imshow(
        np.zeros((NZ, NY)), origin="lower", cmap="inferno", vmin=0, vmax=vmax
    )
    ax_elev.set_title("Elevation (x=mid)")
    ax_elev.set_xlabel("y")
    ax_elev.set_ylabel("z")
    plt.colorbar(elev_img, ax=ax_elev, label="speed")

    plt.tight_layout()
    frames: list[Image.Image] = []

    def capture_frame(step):
        speed = np.sqrt(ux**2 + uy**2 + uz**2)
        plan_img.set_data(speed[2, :, :])
        elev_img.set_data(speed[:, :, NX // 2])   # YZ at x=mid
        ax_plan.set_title(f"Plan view (z=2) — step {step}")
        ax_elev.set_title(f"Elevation (x=mid) — step {step}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        print(f"step {step}, max speed: {speed.max():.4f}")

    # --- Main loop ---
    for step in range(1001):
        # Collision (BGK)
        feq = equilibrium(rho, ux, uy, uz)
        f += (feq - f) / TAU
        f[:, solid] = f[OPP][:, solid]  # zero velocity inside solid

        # Streaming + bounce-back
        f = stream_and_bounce(f, solid)

        # Boundary conditions
        apply_inlet(f, u_profile, NZ, NX)
        apply_outlet(f)

        # Macroscopic variables
        rho = f.sum(axis=0)
        ux = (f * EX[:, None, None, None]).sum(axis=0) / rho
        uy = (f * EY[:, None, None, None]).sum(axis=0) / rho
        uz = (f * EZ[:, None, None, None]).sum(axis=0) / rho

        # Enforce zero velocity in solid
        ux[solid] = 0.0
        uy[solid] = 0.0
        uz[solid] = 0.0

        if step % 100 == 0:
            capture_frame(step)

    plt.close(fig)

    frames[0].save(
        "simulation_3d.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    print(f"Saved simulation_3d.gif ({len(frames)} frames)")

    np.savez(
        "wind_field.npz",
        ux=ux,
        uy=uy,
        uz=uz,
        solid=solid.astype(np.uint8),
        dx=np.float64(dx),
    )
    print("Saved wind_field.npz")


if __name__ == "__main__":
    run_simulation("../../../assets/presets/chicago.h5", 2, 0.5)
