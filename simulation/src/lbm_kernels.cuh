#pragma once
#include <cstdint>

// SoA memory layout: f[q * N + idx], idx = z*NY*NX + y*NX + x
// Two-lattice (A/B swap) for race-free streaming.

void launch_collide_stream(
    const float* f_in, float* f_out,
    const uint8_t* solid,
    int NX, int NY, int NZ);

// Fused variant: performs collision+stream and optionally writes rho/ux/uy/uz
// in the same kernel pass (write_macro=false compiles away all macro stores).
void launch_collide_stream_fused(
    const float* f_in, float* f_out,
    const uint8_t* solid,
    float* rho, float* ux, float* uy, float* uz,
    bool write_macro,
    int NX, int NY, int NZ);

void launch_macroscopic(
    const float* f,
    float* rho, float* ux, float* uy, float* uz,
    const uint8_t* solid,
    int NX, int NY, int NZ);

void launch_inlet_bc(
    float* f,
    const float* u_profile,   // device array [NZ]
    int NX, int NY, int NZ);

void launch_outlet_bc(
    float* f,
    int NX, int NY, int NZ);

void launch_top_bc(
    float* f,
    int NX, int NY, int NZ);
