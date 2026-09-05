// Does a GPU actually help Kraken 2's hash probe?
//
// This is the question the whole offload rests on.  The probe is a random
// gather into a multi-gigabyte table with linear probing: one MurmurHash3 of
// the minimizer, one modulo, then a walk forward until an empty cell or a key
// match.  GPUs are weak at uncoalesced random access relative to their peak
// bandwidth, but they have far more of both bandwidth and latency-hiding
// parallelism than a CPU does, and which effect dominates at this table size
// is not something to guess at.
//
// The table here is synthetic but the access pattern is faithful: cells are
// 4 bytes laid out as Kraken 2 lays them out, occupancy is the 70% the builder
// targets, and probing is the same linear walk with the same early exit.
// Consecutive minimizers in a read are unrelated, so uniform random starts are
// the right model; there is no locality to reproduce.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <vector>
#include <cuda_runtime_api.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "cuda error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static const int KEY_BITS = 16;
static const int MAX_WALK = 64;

__host__ __device__ static inline uint64_t mix64(uint64_t k) {
  k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33; return k;
}

// Fills the table to the target occupancy with a value of 0 marking empty,
// exactly as the real table does.
__global__ void fill(uint32_t *t, size_t cap, uint64_t seed, uint32_t occ_num,
                     uint32_t occ_den) {
  size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t) gridDim.x * blockDim.x;
  for (; i < cap; i += stride) {
    uint64_t h = mix64(seed ^ i);
    if ((uint32_t) (h % occ_den) < occ_num) {
      uint32_t key = (uint32_t) (h >> 48);
      uint32_t val = (uint32_t) ((h >> 16) & 0xffff);
      if (val == 0) val = 1;
      t[i] = (key << 16) | val;
    } else {
      t[i] = 0;
    }
  }
}

__global__ void probe(const uint32_t *__restrict__ t, size_t cap,
                      uint64_t n, uint64_t seed,
                      unsigned long long *hits, unsigned long long *steps) {
  uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t) gridDim.x * blockDim.x;
  unsigned long long h = 0, s = 0;
  for (; i < n; i += stride) {
    uint64_t hc = mix64(seed + i);
    uint32_t ck = (uint32_t) (hc >> (64 - KEY_BITS));
    size_t idx = hc % cap;
    for (int p = 0; p < MAX_WALK; p++) {
      uint32_t cell = t[idx];
      s++;
      if ((cell & 0xffffu) == 0) break;
      if ((cell >> 16) == ck) { h++; break; }
      if (++idx == cap) idx = 0;
    }
  }
  atomicAdd(hits, h);
  atomicAdd(steps, s);
}

static void cpu_probe(const uint32_t *t, size_t cap, uint64_t n, uint64_t seed,
                      uint64_t *hits, uint64_t *steps, int threads) {
  uint64_t H = 0, S = 0;
  const int PF = 16;   // matches the prefetch distance classify uses
  #pragma omp parallel for num_threads(threads) reduction(+:H,S) schedule(static)
  for (long long b = 0; b < (long long) n; b++) {
    uint64_t hc = mix64(seed + (uint64_t) b);
    if (b + PF < (long long) n)
      __builtin_prefetch(&t[mix64(seed + (uint64_t) b + PF) % cap], 0, 1);
    uint32_t ck = (uint32_t) (hc >> (64 - KEY_BITS));
    size_t idx = hc % cap;
    for (int p = 0; p < MAX_WALK; p++) {
      uint32_t cell = t[idx];
      S++;
      if ((cell & 0xffffu) == 0) break;
      if ((cell >> 16) == ck) { H++; break; }
      if (++idx == cap) idx = 0;
    }
  }
  *hits = H; *steps = S;
}

int main(int argc, char **argv) {
  double gb = argc > 1 ? atof(argv[1]) : 15.0;
  uint64_t nprobe = argc > 2 ? strtoull(argv[2], 0, 10) : 500000000ULL;
  int threads = argc > 3 ? atoi(argv[3]) : 32;
  size_t cap = (size_t) (gb * 1e9 / 4);

  printf("table %.1f GB, %zu cells, 70%% occupancy, %" PRIu64 " probes\n", gb, cap, nprobe);

  // ---------------- GPU ----------------
  uint32_t *d_t = nullptr;
  CK(cudaMalloc(&d_t, cap * 4));
  fill<<<4096, 256>>>(d_t, cap, 12345, 70, 100);
  CK(cudaDeviceSynchronize());

  unsigned long long *d_h, *d_s;
  CK(cudaMalloc(&d_h, 8)); CK(cudaMalloc(&d_s, 8));
  CK(cudaMemset(d_h, 0, 8)); CK(cudaMemset(d_s, 0, 8));

  probe<<<8192, 256>>>(d_t, cap, nprobe / 20, 999, d_h, d_s);   // warm
  CK(cudaDeviceSynchronize());
  CK(cudaMemset(d_h, 0, 8)); CK(cudaMemset(d_s, 0, 8));

  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  probe<<<8192, 256>>>(d_t, cap, nprobe, 4242, d_h, d_s);
  CK(cudaEventRecord(b));
  CK(cudaDeviceSynchronize());
  float ms = 0; cudaEventElapsedTime(&ms, a, b);
  unsigned long long gh = 0, gs = 0;
  CK(cudaMemcpy(&gh, d_h, 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&gs, d_s, 8, cudaMemcpyDeviceToHost));
  double gpu_pps = nprobe / (ms / 1000.0);
  printf("GPU   %8.1f ms   %7.2f Mprobe/s   %.2f steps/probe   %5.2f GB/s random\n",
         ms, gpu_pps / 1e6, (double) gs / nprobe, gs * 4.0 / (ms / 1000.0) / 1e9);

  // ---------------- CPU ----------------
  // Host copy of the same table, built identically on the host.
  std::vector<uint32_t> h_t;
  bool have_host = true;
  try { h_t.resize(cap); } catch (...) { have_host = false; }
  if (have_host) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (long long i = 0; i < (long long) cap; i++) {
      uint64_t h = mix64(12345 ^ (uint64_t) i);
      if ((uint32_t) (h % 100) < 70) {
        uint32_t key = (uint32_t) (h >> 48);
        uint32_t val = (uint32_t) ((h >> 16) & 0xffff);
        if (val == 0) val = 1;
        h_t[i] = (key << 16) | val;
      } else h_t[i] = 0;
    }
    uint64_t ch = 0, cs = 0;
    uint64_t cpu_n = nprobe / 10;
    cpu_probe(&h_t[0], cap, cpu_n / 10, 999, &ch, &cs, threads);   // warm
    double t0 = omp_get_wtime();
    cpu_probe(&h_t[0], cap, cpu_n, 4242, &ch, &cs, threads);
    double el = omp_get_wtime() - t0;
    double cpu_pps = cpu_n / el;
    printf("CPU   %8.1f ms   %7.2f Mprobe/s   %.2f steps/probe   %5.2f GB/s random  (%d threads)\n",
           el * 1000, cpu_pps / 1e6, (double) cs / cpu_n, cs * 4.0 / el / 1e9, threads);
    printf("\nGPU / CPU probe throughput: %.2fx\n", gpu_pps / cpu_pps);
  } else {
    printf("CPU   skipped, host allocation of %.1f GB failed\n", gb);
  }
  return 0;
}
