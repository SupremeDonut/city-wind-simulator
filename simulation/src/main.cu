#include "lbm_kernels.cuh"
#include "third_party/cnpy/cnpy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static constexpr int Q = 19;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ─── Log-law wind profile (matches d3q19.py) ──────────────────────────────

static float log_wind_profile(float z, float z_ref, float u_ref, float z0) {
  if (z <= z0)
    return 0.0f;
  return u_ref * logf(z / z0) / logf(z_ref / z0);
}

// ─── Compute equilibrium on host and upload ────────────────────────────────

static void init_equilibrium_host(float *f_host, int N, int NX, int NY, int NZ,
                                  const float *u_profile,
                                  const uint8_t *solid) {
  // Host-side lattice constants (same as __constant__ arrays)
  const int ex[19] = {0,  1, -1, 0, 0,  0, 0, 1, -1, 1,
                      -1, 1, -1, 1, -1, 0, 0, 0, 0};
  const int ey[19] = {0, 0, 0, 1, -1, 0, 0,  1, -1, -1,
                      1, 0, 0, 0, 0,  1, -1, 1, -1};
  const int ez[19] = {0, 0, 0,  0,  0, 1, -1, 0,  0, 0,
                      0, 1, -1, -1, 1, 1, -1, -1, 1};
  const float w[19] = {1.0f / 3.0f,  1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
                       1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 36.0f,
                       1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
                       1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
                       1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};
  const float cs2 = 1.0f / 3.0f;

  for (int z = 0; z < NZ; z++) {
    float vy = u_profile[z];
    for (int y = 0; y < NY; y++) {
      for (int x = 0; x < NX; x++) {
        int c = z * NY * NX + y * NX + x;
        float vx = 0.0f, vz = 0.0f;
        float uy_cell = solid[c] ? 0.0f : vy;
        float u2 = uy_cell * uy_cell;

        for (int q = 0; q < 19; q++) {
          float eu =
              (float)ex[q] * vx + (float)ey[q] * uy_cell + (float)ez[q] * vz;
          float feq = w[q] * 1.0f *
                      (1.0f + eu / cs2 + eu * eu / (2.0f * cs2 * cs2) -
                       u2 / (2.0f * cs2));
          f_host[q * N + c] = feq;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  const char *snapshot_dir = (argc >= 2) ? argv[1] : nullptr;

#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  // ─── 1. Read binary input from stdin ──────────────────────────────────
  // Header layout (little-endian, 32 bytes):
  //   [0] NX          int32
  //   [1] NY          int32
  //   [2] NZ          int32
  //   [3] wind_speed  float32
  //   [4] roughness   float32
  //   [5] domain_z    float32
  //   [6] num_steps   int32
  //   [7] snapshot_interval int32
  // Body: solid[NZ*NY*NX] uint8

  int32_t NX, NY, NZ, num_steps, snapshot_interval;
  float wind_speed, roughness, domain_z;

  auto must_read = [](void *buf, size_t n, const char *what) {
    if (fread(buf, 1, n, stdin) != n) {
      fprintf(stderr, "stdin read error: expected %zu bytes for %s\n", n, what);
      exit(1);
    }
  };

  must_read(&NX, 4, "NX");
  must_read(&NY, 4, "NY");
  must_read(&NZ, 4, "NZ");
  must_read(&wind_speed, 4, "wind_speed");
  must_read(&roughness, 4, "roughness");
  must_read(&domain_z, 4, "domain_z");
  must_read(&num_steps, 4, "num_steps");
  must_read(&snapshot_interval, 4, "snapshot_interval");

  int N = NX * NY * NZ;
  std::vector<uint8_t> solid_vec(N);
  must_read(solid_vec.data(), N, "solid");
  uint8_t *solid_h = solid_vec.data();

  // Snapshot only active when a snapshot_dir was provided
  if (!snapshot_dir)
    snapshot_interval = 0;

  float dx = domain_z / (float)NZ;

  fprintf(
      stderr,
      "Grid: %dx%dx%d = %d cells, dx=%.2f m, steps=%d, snapshot_interval=%d\n",
      NX, NY, NZ, N, dx, num_steps, snapshot_interval);

  // ─── 2. Build inlet profile (log-law, lattice units) ──────────────────
  float u_ref_lattice = 0.06f;
  float z0_lattice = roughness / dx;
  float z_ref_lattice = 10.0f / dx;

  std::vector<float> u_profile_h(NZ);
  for (int k = 0; k < NZ; k++) {
    u_profile_h[k] =
        log_wind_profile(k + 0.5f, z_ref_lattice, u_ref_lattice, z0_lattice);
  }

  // ─── 3. Init equilibrium on host ──────────────────────────────────────
  size_t f_bytes = (size_t)Q * N * sizeof(float);
  size_t field_bytes = (size_t)N * sizeof(float);
  size_t solid_bytes = (size_t)N * sizeof(uint8_t);

  std::vector<float> f_host(Q * N);
  init_equilibrium_host(f_host.data(), N, NX, NY, NZ, u_profile_h.data(),
                        solid_h);

  // ─── 4. Allocate device memory ────────────────────────────────────────
  float *d_fA, *d_fB;
  float *d_rho, *d_ux, *d_uy, *d_uz;
  uint8_t *d_solid;
  float *d_u_profile;

  CUDA_CHECK(cudaMalloc(&d_fA, f_bytes));
  CUDA_CHECK(cudaMalloc(&d_fB, f_bytes));
  CUDA_CHECK(cudaMalloc(&d_rho, field_bytes));
  CUDA_CHECK(cudaMalloc(&d_ux, field_bytes));
  CUDA_CHECK(cudaMalloc(&d_uy, field_bytes));
  CUDA_CHECK(cudaMalloc(&d_uz, field_bytes));
  CUDA_CHECK(cudaMalloc(&d_solid, solid_bytes));
  CUDA_CHECK(cudaMalloc(&d_u_profile, NZ * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_fA, f_host.data(), f_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_solid, solid_h, solid_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_u_profile, u_profile_h.data(), NZ * sizeof(float),
                        cudaMemcpyHostToDevice));

  fprintf(stderr, "GPU memory: %.1f MB allocated\n",
          (2.0 * f_bytes + 4 * field_bytes + solid_bytes) / 1e6);

  // ─── 5. Main loop ─────────────────────────────────────────────────────
  int progress_interval = num_steps < 100 ? 1 : num_steps / 100;

  // Snapshot host buffers (allocated once, reused)
  std::vector<float> snap_ux, snap_uy, snap_uz;
  if (snapshot_interval > 0) {
    snap_ux.resize(N);
    snap_uy.resize(N);
    snap_uz.resize(N);
  }

  for (int step = 0; step < num_steps; step++) {
    // Macro fields needed only for snapshots and the final output step
    bool need_macro = (step == num_steps - 1) ||
        (snapshot_interval > 0 && (step + 1) % snapshot_interval == 0);

    launch_collide_stream_fused(d_fA, d_fB, d_solid,
                                d_rho, d_ux, d_uy, d_uz,
                                need_macro, NX, NY, NZ);
    launch_inlet_bc(d_fB, d_u_profile, NX, NY, NZ);
    launch_outlet_bc(d_fB, NX, NY, NZ);
    launch_top_bc(d_fB, NX, NY, NZ);

    // Swap A <-> B (pointer swap)
    float *tmp = d_fA;
    d_fA = d_fB;
    d_fB = tmp;

    // Snapshot writing
    if (snapshot_interval > 0 && (step + 1) % snapshot_interval == 0) {
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(snap_ux.data(), d_ux, field_bytes,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(snap_uy.data(), d_uy, field_bytes,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(snap_uz.data(), d_uz, field_bytes,
                            cudaMemcpyDeviceToHost));

      char snap_name[64];
      snprintf(snap_name, sizeof(snap_name), "snapshot_%04d.npz", step + 1);
      std::string snap_path = std::string(snapshot_dir) + "/" + snap_name;

      unsigned int shape3[3] = {(unsigned)NZ, (unsigned)NY, (unsigned)NX};
      unsigned int shape1[1] = {1};
      int step_val = step + 1;

      cnpy::npz_save(snap_path, "ux", snap_ux.data(), shape3, 3, "w");
      cnpy::npz_save(snap_path, "uy", snap_uy.data(), shape3, 3, "a");
      cnpy::npz_save(snap_path, "uz", snap_uz.data(), shape3, 3, "a");
      cnpy::npz_save(snap_path, "step", &step_val, shape1, 1, "a");

      fprintf(stderr, "SNAPSHOT %s\n", snap_name);
      fflush(stderr);
    }

    // Progress reporting
    if (step % progress_interval == 0 || step == num_steps - 1) {
      fprintf(stderr, "PROGRESS %.4f\n", (float)(step + 1) / (float)num_steps);
      fflush(stderr);
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  // ─── 6. Copy results back ─────────────────────────────────────────────
  std::vector<float> ux_h(N), uy_h(N), uz_h(N);
  CUDA_CHECK(
      cudaMemcpy(ux_h.data(), d_ux, field_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(uy_h.data(), d_uy, field_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(uz_h.data(), d_uz, field_bytes, cudaMemcpyDeviceToHost));

  // Also copy solid back (it wasn't modified but we include it in output)
  std::vector<uint8_t> solid_out(N);
  CUDA_CHECK(cudaMemcpy(solid_out.data(), d_solid, solid_bytes,
                        cudaMemcpyDeviceToHost));

  // ─── 7. Write binary results to stdout ───────────────────────────────
  // Layout: [dx: f32] [ux: N×f32] [uy: N×f32] [uz: N×f32] [solid: N×u8]
  fwrite(&dx, sizeof(float), 1, stdout);
  fwrite(ux_h.data(), sizeof(float), N, stdout);
  fwrite(uy_h.data(), sizeof(float), N, stdout);
  fwrite(uz_h.data(), sizeof(float), N, stdout);
  fwrite(solid_out.data(), sizeof(uint8_t), N, stdout);
  fflush(stdout);

  fprintf(stderr, "Done. Written %zu bytes to stdout.\n",
          sizeof(float) + (size_t)N * (3 * sizeof(float) + sizeof(uint8_t)));

  // ─── Cleanup ──────────────────────────────────────────────────────────
  cudaFree(d_fA);
  cudaFree(d_fB);
  cudaFree(d_rho);
  cudaFree(d_ux);
  cudaFree(d_uy);
  cudaFree(d_uz);
  cudaFree(d_solid);
  cudaFree(d_u_profile);

  return 0;
}
