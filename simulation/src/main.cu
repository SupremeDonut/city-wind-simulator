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

// ─── Persistent GPU state (reused across simulations) ─────────────────────

struct GpuState {
  float    *d_fA       = nullptr;
  float    *d_fB       = nullptr;
  float    *d_rho      = nullptr;
  float    *d_ux       = nullptr;
  float    *d_uy       = nullptr;
  float    *d_uz       = nullptr;
  uint8_t  *d_solid    = nullptr;
  float    *d_u_profile = nullptr;
  int       allocated_N  = 0;
  int       allocated_NZ = 0;

  void ensure(int N, int NZ) {
    if (N == allocated_N && NZ == allocated_NZ)
      return;
    free_all();
    size_t f_bytes     = (size_t)Q * N * sizeof(float);
    size_t field_bytes = (size_t)N * sizeof(float);
    size_t solid_bytes = (size_t)N * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_fA,        f_bytes));
    CUDA_CHECK(cudaMalloc(&d_fB,        f_bytes));
    CUDA_CHECK(cudaMalloc(&d_rho,       field_bytes));
    CUDA_CHECK(cudaMalloc(&d_ux,        field_bytes));
    CUDA_CHECK(cudaMalloc(&d_uy,        field_bytes));
    CUDA_CHECK(cudaMalloc(&d_uz,        field_bytes));
    CUDA_CHECK(cudaMalloc(&d_solid,     solid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_profile, NZ * sizeof(float)));
    allocated_N  = N;
    allocated_NZ = NZ;
  }

  void free_all() {
    if (d_fA)        { cudaFree(d_fA);        d_fA        = nullptr; }
    if (d_fB)        { cudaFree(d_fB);        d_fB        = nullptr; }
    if (d_rho)       { cudaFree(d_rho);       d_rho       = nullptr; }
    if (d_ux)        { cudaFree(d_ux);        d_ux        = nullptr; }
    if (d_uy)        { cudaFree(d_uy);        d_uy        = nullptr; }
    if (d_uz)        { cudaFree(d_uz);        d_uz        = nullptr; }
    if (d_solid)     { cudaFree(d_solid);     d_solid     = nullptr; }
    if (d_u_profile) { cudaFree(d_u_profile); d_u_profile = nullptr; }
    allocated_N  = 0;
    allocated_NZ = 0;
  }
};

// ─── Run one simulation ────────────────────────────────────────────────────
// Returns false if stdin closed unexpectedly (persistent mode only).

static bool run_simulation(GpuState &gpu, const char *snapshot_dir,
                           int32_t NX, int32_t NY, int32_t NZ,
                           float wind_speed, float roughness, float domain_z,
                           int32_t num_steps, int32_t snapshot_interval,
                           const uint8_t *solid_h) {
  int N = NX * NY * NZ;
  float dx = domain_z / (float)NZ;

  fprintf(stderr,
          "Grid: %dx%dx%d = %d cells, dx=%.2f m, steps=%d\n",
          NX, NY, NZ, N, dx, num_steps);

  // ── Build inlet profile ────────────────────────────────────────────────
  float u_ref_lattice  = 0.06f;
  float z0_lattice     = roughness / dx;
  float z_ref_lattice  = 10.0f / dx;

  std::vector<float> u_profile_h(NZ);
  for (int k = 0; k < NZ; k++)
    u_profile_h[k] = log_wind_profile(k + 0.5f, z_ref_lattice,
                                       u_ref_lattice, z0_lattice);

  // ── Init equilibrium on host ───────────────────────────────────────────
  std::vector<float> f_host((size_t)Q * N);
  init_equilibrium_host(f_host.data(), N, NX, NY, NZ,
                        u_profile_h.data(), solid_h);

  // ── Ensure device buffers (no-op if same grid size) ───────────────────
  gpu.ensure(N, NZ);

  size_t f_bytes     = (size_t)Q * N * sizeof(float);
  size_t field_bytes = (size_t)N * sizeof(float);
  size_t solid_bytes = (size_t)N * sizeof(uint8_t);

  CUDA_CHECK(cudaMemcpy(gpu.d_fA,        f_host.data(),       f_bytes,     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gpu.d_solid,     solid_h,             solid_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gpu.d_u_profile, u_profile_h.data(),  NZ * sizeof(float), cudaMemcpyHostToDevice));

  // ── Main simulation loop ───────────────────────────────────────────────
  int progress_interval = num_steps < 100 ? 1 : num_steps / 100;

  std::vector<float> snap_ux, snap_uy, snap_uz;
  if (snapshot_interval > 0 && snapshot_dir) {
    snap_ux.resize(N); snap_uy.resize(N); snap_uz.resize(N);
  }

  for (int step = 0; step < num_steps; step++) {
    bool need_macro = (step == num_steps - 1) ||
        (snapshot_interval > 0 && (step + 1) % snapshot_interval == 0);

    launch_collide_stream_fused(gpu.d_fA, gpu.d_fB, gpu.d_solid,
                                gpu.d_rho, gpu.d_ux, gpu.d_uy, gpu.d_uz,
                                need_macro, NX, NY, NZ);
    launch_inlet_bc(gpu.d_fB, gpu.d_u_profile, NX, NY, NZ);
    launch_outlet_bc(gpu.d_fB, NX, NY, NZ);
    launch_top_bc(gpu.d_fB, NX, NY, NZ);

    float *tmp = gpu.d_fA; gpu.d_fA = gpu.d_fB; gpu.d_fB = tmp;

    if (snapshot_interval > 0 && snapshot_dir &&
        (step + 1) % snapshot_interval == 0) {
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(snap_ux.data(), gpu.d_ux, field_bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(snap_uy.data(), gpu.d_uy, field_bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(snap_uz.data(), gpu.d_uz, field_bytes, cudaMemcpyDeviceToHost));

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

    if (step % progress_interval == 0 || step == num_steps - 1) {
      fprintf(stderr, "PROGRESS %.4f\n", (float)(step + 1) / (float)num_steps);
      fflush(stderr);
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  // ── Copy results back ──────────────────────────────────────────────────
  std::vector<float>   ux_h(N), uy_h(N), uz_h(N);
  std::vector<uint8_t> solid_out(N);

  CUDA_CHECK(cudaMemcpy(ux_h.data(),    gpu.d_ux,    field_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(uy_h.data(),    gpu.d_uy,    field_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(uz_h.data(),    gpu.d_uz,    field_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(solid_out.data(), gpu.d_solid, solid_bytes, cudaMemcpyDeviceToHost));

  // ── Write binary results to stdout ────────────────────────────────────
  // Layout: [dx: f32] [ux: N×f32] [uy: N×f32] [uz: N×f32] [solid: N×u8]
  fwrite(&dx,             sizeof(float),   1, stdout);
  fwrite(ux_h.data(),     sizeof(float),   N, stdout);
  fwrite(uy_h.data(),     sizeof(float),   N, stdout);
  fwrite(uz_h.data(),     sizeof(float),   N, stdout);
  fwrite(solid_out.data(), sizeof(uint8_t), N, stdout);
  fflush(stdout);

  fprintf(stderr, "Done. Written %zu bytes to stdout.\n",
          sizeof(float) + (size_t)N * (3 * sizeof(float) + sizeof(uint8_t)));

  return true;
}

// ─── Read exactly n bytes or return false on EOF ───────────────────────────

static bool read_exact(void *buf, size_t n, const char *what) {
  size_t got = fread(buf, 1, n, stdin);
  if (got == n) return true;
  if (got == 0 && feof(stdin)) return false;  // clean EOF (persistent mode)
  fprintf(stderr, "stdin read error: got %zu / %zu bytes for %s\n", got, n, what);
  exit(1);
}

// ─── Entry point ──────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  bool persistent = false;
  const char *snapshot_dir = nullptr;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--persistent") == 0)
      persistent = true;
    else
      snapshot_dir = argv[i];  // treat any other arg as snapshot dir path
  }

#ifdef _WIN32
  _setmode(_fileno(stdin),  _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  GpuState gpu;

  // ── Persistent mode: loop over multiple simulations ───────────────────
  // Protocol: send header (NX=0 as sentinel to exit), then solid grid.
  // CUDA context and device allocations are reused across calls as long
  // as grid dimensions stay the same (typical for data generation).
  do {
    int32_t NX, NY, NZ, num_steps, snapshot_interval;
    float   wind_speed, roughness, domain_z;

    if (!read_exact(&NX,               4, "NX"))          break;
    if (!read_exact(&NY,               4, "NY"))          break;
    if (!read_exact(&NZ,               4, "NZ"))          break;
    if (!read_exact(&wind_speed,       4, "wind_speed"))  break;
    if (!read_exact(&roughness,        4, "roughness"))   break;
    if (!read_exact(&domain_z,         4, "domain_z"))    break;
    if (!read_exact(&num_steps,        4, "num_steps"))   break;
    if (!read_exact(&snapshot_interval,4, "snap_interval")) break;

    // NX == 0 is the sentinel: clean shutdown
    if (NX == 0) break;

    int N = NX * NY * NZ;
    std::vector<uint8_t> solid_vec(N);
    if (!read_exact(solid_vec.data(), N, "solid")) break;

    if (!snapshot_dir)
      snapshot_interval = 0;

    run_simulation(gpu, snapshot_dir,
                   NX, NY, NZ, wind_speed, roughness, domain_z,
                   num_steps, snapshot_interval, solid_vec.data());

  } while (persistent);  // single-shot mode exits after one iteration

  gpu.free_all();
  return 0;
}
