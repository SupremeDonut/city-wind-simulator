#include "lbm_kernels.cuh"
#include "lattice_constants.cuh"
#include <cstdio>

// Thread block dimensions
static constexpr int BX = 32;
static constexpr int BY = 4;
static constexpr int BZ = 2;

// Helper: linear index in ZYX order
__device__ __forceinline__ int idx3(int x, int y, int z, int NX, int NY) {
    return z * NY * NX + y * NX + x;
}

// ─── Kernel 1: Collide + Stream (BGK, two-lattice) ────────────────────────

__launch_bounds__(256, 2)
__global__ void collide_stream_kernel(
    const float* __restrict__ f_in,
    float*       __restrict__ f_out,
    const uint8_t* __restrict__ solid,
    int NX, int NY, int NZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    int N = NX * NY * NZ;
    int c = idx3(x, y, z, NX, NY);

    if (solid[c]) {
        // Bounce-back: reflect all populations in place
        for (int q = 0; q < Q; q++) {
            f_out[OPP[q] * N + c] = f_in[q * N + c];
        }
        return;
    }

    // ── Compute macroscopic quantities ──
    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    float fi[Q];
    for (int q = 0; q < Q; q++) {
        fi[q] = f_in[q * N + c];
        rho += fi[q];
        mx  += EX[q] * fi[q];
        my  += EY[q] * fi[q];
        mz  += EZ[q] * fi[q];
    }
    float inv_rho = 1.0f / rho;
    float vx = mx * inv_rho;
    float vy = my * inv_rho;
    float vz = mz * inv_rho;

    // ── BGK collision ──
    float u2 = vx * vx + vy * vy + vz * vz;
    float inv_tau = 1.0f / TAU;

    for (int q = 0; q < Q; q++) {
        float eu = EX[q] * vx + EY[q] * vy + EZ[q] * vz;
        float feq = W[q] * rho * (1.0f + eu / CS2 + eu * eu / (2.0f * CS2 * CS2) - u2 / (2.0f * CS2));
        float f_coll = fi[q] + (feq - fi[q]) * inv_tau;

        // ── Stream to neighbour ──
        // Periodic in X; Y and Z boundaries handled by BC kernels
        int xn = (x + EX[q] + NX) % NX;   // periodic X
        int yn =  y + EY[q];               // no wrap Y
        int zn =  z + EZ[q];               // no wrap Z

        // Clamp Y to domain (inlet/outlet BCs will overwrite faces)
        if (yn < 0)  yn = 0;
        if (yn >= NY) yn = NY - 1;

        // Clamp Z to domain (top BC will overwrite z=NZ-1 face)
        if (zn < 0)  zn = 0;
        if (zn >= NZ) zn = NZ - 1;

        int cn = idx3(xn, yn, zn, NX, NY);

        if (solid[cn]) {
            // Bounce-back: reflected population stays at current cell
            f_out[OPP[q] * N + c] = f_coll;
        } else {
            f_out[q * N + cn] = f_coll;
        }
    }
}

// ─── Kernel 1b: Collide + Stream with optional fused macro write ───────────
//
// WriteMacro=true  – also writes rho/ux/uy/uz from the collision-step values
//                    (computed before BC kernels overwrite the face populations,
//                    so BC face cells may lag by one step – acceptable for open BCs).
// WriteMacro=false – if constexpr eliminates all macro stores; output pointers
//                    are passed but never dereferenced.

template <bool WriteMacro>
__launch_bounds__(256, 2)
__global__ void collide_stream_macro_kernel(
    const float* __restrict__ f_in,
    float*       __restrict__ f_out,
    const uint8_t* __restrict__ solid,
    float* __restrict__ rho_out,
    float* __restrict__ ux_out,
    float* __restrict__ uy_out,
    float* __restrict__ uz_out,
    int NX, int NY, int NZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    int N = NX * NY * NZ;
    int c = idx3(x, y, z, NX, NY);

    if (solid[c]) {
        // Bounce-back: reflect all populations in place
        for (int q = 0; q < Q; q++)
            f_out[OPP[q] * N + c] = f_in[q * N + c];
        if constexpr (WriteMacro) {
            rho_out[c] = 0.0f;
            ux_out[c]  = 0.0f;
            uy_out[c]  = 0.0f;
            uz_out[c]  = 0.0f;
        }
        return;
    }

    // ── Compute macroscopic quantities ──
    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    float fi[Q];
    for (int q = 0; q < Q; q++) {
        fi[q] = f_in[q * N + c];
        rho += fi[q];
        mx  += EX[q] * fi[q];
        my  += EY[q] * fi[q];
        mz  += EZ[q] * fi[q];
    }
    float inv_rho = 1.0f / rho;
    float vx = mx * inv_rho;
    float vy = my * inv_rho;
    float vz = mz * inv_rho;

    // Write macro fields while we already have rho/vx/vy/vz in registers
    if constexpr (WriteMacro) {
        rho_out[c] = rho;
        ux_out[c]  = vx;
        uy_out[c]  = vy;
        uz_out[c]  = vz;
    }

    // ── BGK collision ──
    float u2      = vx * vx + vy * vy + vz * vz;
    float inv_tau = 1.0f / TAU;

    for (int q = 0; q < Q; q++) {
        float eu     = EX[q] * vx + EY[q] * vy + EZ[q] * vz;
        float feq    = W[q] * rho * (1.0f + eu / CS2
                                    + eu * eu / (2.0f * CS2 * CS2)
                                    - u2 / (2.0f * CS2));
        float f_coll = fi[q] + (feq - fi[q]) * inv_tau;

        // ── Stream to neighbour ──
        int xn = (x + EX[q] + NX) % NX;
        int yn =  y + EY[q];
        int zn =  z + EZ[q];
        if (yn < 0)   yn = 0;
        if (yn >= NY) yn = NY - 1;
        if (zn < 0)   zn = 0;
        if (zn >= NZ) zn = NZ - 1;

        int cn = idx3(xn, yn, zn, NX, NY);
        if (solid[cn])
            f_out[OPP[q] * N + c] = f_coll;
        else
            f_out[q * N + cn] = f_coll;
    }
}

// ─── Kernel 2: Macroscopic quantities ──────────────────────────────────────

__global__ void macroscopic_kernel(
    const float* __restrict__ f,
    float* __restrict__ rho_out,
    float* __restrict__ ux_out,
    float* __restrict__ uy_out,
    float* __restrict__ uz_out,
    const uint8_t* __restrict__ solid,
    int NX, int NY, int NZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    int N = NX * NY * NZ;
    int c = idx3(x, y, z, NX, NY);

    float r = 0.0f, mx = 0.0f, my = 0.0f, mz_val = 0.0f;
    for (int q = 0; q < Q; q++) {
        float fq = f[q * N + c];
        r     += fq;
        mx    += EX[q] * fq;
        my    += EY[q] * fq;
        mz_val += EZ[q] * fq;
    }

    rho_out[c] = r;
    if (solid[c]) {
        ux_out[c] = 0.0f;
        uy_out[c] = 0.0f;
        uz_out[c] = 0.0f;
    } else {
        float inv_r = 1.0f / r;
        ux_out[c] = mx * inv_r;
        uy_out[c] = my * inv_r;
        uz_out[c] = mz_val * inv_r;
    }
}

// ─── Kernel 3: Inlet BC (y=0 face, equilibrium with log-law profile) ──────

__global__ void inlet_bc_kernel(
    float* __restrict__ f,
    const float* __restrict__ u_profile,   // [NZ]
    int NX, int NY, int NZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || z >= NZ) return;

    int N = NX * NY * NZ;
    int c = idx3(x, 0, z, NX, NY);   // y = 0

    float vy = u_profile[z];
    float u2 = vy * vy;

    for (int q = 0; q < Q; q++) {
        float eu = EY[q] * vy;   // ux=uz=0 so only EY contributes
        float feq = W[q] * (1.0f + eu / CS2 + eu * eu / (2.0f * CS2 * CS2) - u2 / (2.0f * CS2));
        f[q * N + c] = feq;
    }
}

// ─── Kernel 4: Outlet BC (y=NY-1 face, zero-gradient copy from y=NY-2) ────

__global__ void outlet_bc_kernel(
    float* __restrict__ f,
    int NX, int NY, int NZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || z >= NZ) return;

    int N = NX * NY * NZ;
    int c_last = idx3(x, NY - 1, z, NX, NY);
    int c_prev = idx3(x, NY - 2, z, NX, NY);

    for (int q = 0; q < Q; q++) {
        f[q * N + c_last] = f[q * N + c_prev];
    }
}

// ─── Kernel 5: Top BC (z=NZ-1 face, zero-gradient copy from z=NZ-2) ───────

__global__ void top_bc_kernel(
    float* __restrict__ f,
    int NX, int NY, int NZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    int N = NX * NY * NZ;
    int c_top  = idx3(x, y, NZ - 1, NX, NY);
    int c_prev = idx3(x, y, NZ - 2, NX, NY);

    for (int q = 0; q < Q; q++) {
        f[q * N + c_top] = f[q * N + c_prev];
    }
}

// ─── Launch wrappers ───────────────────────────────────────────────────────

void launch_collide_stream(
    const float* f_in, float* f_out,
    const uint8_t* solid,
    int NX, int NY, int NZ)
{
    dim3 block(BX, BY, BZ);
    dim3 grid((NX + BX - 1) / BX, (NY + BY - 1) / BY, (NZ + BZ - 1) / BZ);
    collide_stream_kernel<<<grid, block>>>(f_in, f_out, solid, NX, NY, NZ);
}

void launch_collide_stream_fused(
    const float* f_in, float* f_out,
    const uint8_t* solid,
    float* rho, float* ux, float* uy, float* uz,
    bool write_macro,
    int NX, int NY, int NZ)
{
    dim3 block(BX, BY, BZ);
    dim3 grid((NX + BX - 1) / BX, (NY + BY - 1) / BY, (NZ + BZ - 1) / BZ);
    if (write_macro)
        collide_stream_macro_kernel<true><<<grid, block>>>(
            f_in, f_out, solid, rho, ux, uy, uz, NX, NY, NZ);
    else
        collide_stream_macro_kernel<false><<<grid, block>>>(
            f_in, f_out, solid, rho, ux, uy, uz, NX, NY, NZ);
}

void launch_macroscopic(
    const float* f,
    float* rho, float* ux, float* uy, float* uz,
    const uint8_t* solid,
    int NX, int NY, int NZ)
{
    dim3 block(BX, BY, BZ);
    dim3 grid((NX + BX - 1) / BX, (NY + BY - 1) / BY, (NZ + BZ - 1) / BZ);
    macroscopic_kernel<<<grid, block>>>(f, rho, ux, uy, uz, solid, NX, NY, NZ);
}

void launch_inlet_bc(
    float* f,
    const float* u_profile,
    int NX, int NY, int NZ)
{
    dim3 block(16, 16);
    dim3 grid((NX + 15) / 16, (NZ + 15) / 16);
    inlet_bc_kernel<<<grid, block>>>(f, u_profile, NX, NY, NZ);
}

void launch_outlet_bc(
    float* f,
    int NX, int NY, int NZ)
{
    dim3 block(16, 16);
    dim3 grid((NX + 15) / 16, (NZ + 15) / 16);
    outlet_bc_kernel<<<grid, block>>>(f, NX, NY, NZ);
}

void launch_top_bc(
    float* f,
    int NX, int NY, int NZ)
{
    dim3 block(16, 16);
    dim3 grid((NX + 15) / 16, (NY + 15) / 16);
    top_bc_kernel<<<grid, block>>>(f, NX, NY, NZ);
}
