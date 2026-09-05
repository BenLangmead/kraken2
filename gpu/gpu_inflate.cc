// Decompress record-aligned BGZF blocks on the GPU with nvCOMP, verify the
// output against the original FASTQ, and measure throughput.
//
// Each BGZF block is a complete gzip member, so the batched gzip decompressor
// takes them exactly as they sit in the file: no header stripping, no
// reframing.  The index supplies each block's offset and uncompressed size, so
// the batch can be built without inspecting the data.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nvcomp/gzip.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "cuda error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
  exit(1); } } while (0)
#define NK(x) do { nvcompStatus_t s = (x); if (s != nvcompSuccess) { \
  fprintf(stderr, "nvcomp error %d at line %d\n", (int) s, __LINE__); exit(1); } } while (0)

struct IdxHeader {
  char magic[8]; uint32_t version, block_target;
  uint64_t n_blocks, n_records, total_uncompressed;
};
struct IdxEnt { uint64_t coff, uoff, rec; };

static double now_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: gpu_inflate file.k2bgz file.k2i original.fq "
                    "[blocks_per_batch] [backend 0=default 2=cuda]\n");
    return 1;
  }
  const char *bgz = argv[1], *idx = argv[2], *orig = argv[3];
  size_t batch = argc > 4 ? (size_t) atoll(argv[4]) : 8192;
  int backend = argc > 5 ? atoi(argv[5]) : 0;

  // ---- index ----
  FILE *f = fopen(idx, "rb");
  if (!f) { perror("index"); return 1; }
  IdxHeader h;
  if (fread(&h, sizeof h, 1, f) != 1) return 1;
  std::vector<IdxEnt> e(h.n_blocks + 1);
  if (fread(&e[0], sizeof(IdxEnt), e.size(), f) != e.size()) return 1;
  fclose(f);
  printf("blocks %llu, records %llu, uncompressed %.2f GB, compressed %.2f GB\n",
         (unsigned long long) h.n_blocks, (unsigned long long) h.n_records,
         h.total_uncompressed / 1e9, e[h.n_blocks].coff / 1e9);

  int fd = open(bgz, O_RDONLY);
  int fo = open(orig, O_RDONLY);
  if (fd < 0 || fo < 0) { perror("open"); return 1; }

  nvcompBatchedGzipDecompressOpts_t opts = nvcompBatchedGzipDecompressDefaultOpts;
  opts.backend = (nvcompDecompressBackend_t) backend;

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));
  cudaEvent_t t0, t1, t2;
  CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1)); CK(cudaEventCreate(&t2));

  size_t max_ublock = 0;
  for (size_t i = 0; i < h.n_blocks; i++)
    max_ublock = std::max(max_ublock, (size_t) (e[i + 1].uoff - e[i].uoff));

  double h2d_ms = 0, dec_ms = 0;
  size_t total_out = 0, total_in = 0, nbatch = 0, mismatches = 0;
  std::vector<char> hin, hout, horig;

  for (size_t b0 = 0; b0 < h.n_blocks; b0 += batch) {
    size_t b1 = std::min(b0 + batch, (size_t) h.n_blocks);
    size_t n = b1 - b0;
    size_t clen = e[b1].coff - e[b0].coff;
    size_t ulen = e[b1].uoff - e[b0].uoff;

    hin.resize(clen);
    size_t got = 0;
    while (got < clen) {
      ssize_t r = pread(fd, &hin[0] + got, clen - got, e[b0].coff + got);
      if (r <= 0) break;
      got += r;
    }

    void *d_in = nullptr, *d_out = nullptr, *d_temp = nullptr;
    CK(cudaMalloc(&d_in, clen));
    CK(cudaMalloc(&d_out, ulen));

    std::vector<void *> h_cptr(n), h_uptr(n);
    std::vector<size_t> h_cbytes(n), h_ubytes(n);
    for (size_t i = 0; i < n; i++) {
      h_cptr[i] = (char *) d_in + (e[b0 + i].coff - e[b0].coff);
      h_uptr[i] = (char *) d_out + (e[b0 + i].uoff - e[b0].uoff);
      h_cbytes[i] = e[b0 + i + 1].coff - e[b0 + i].coff;
      h_ubytes[i] = e[b0 + i + 1].uoff - e[b0 + i].uoff;
    }
    void **d_cptr, **d_uptr; size_t *d_cbytes, *d_ubytes, *d_actual;
    nvcompStatus_t *d_stat;
    CK(cudaMalloc(&d_cptr, n * sizeof(void *)));
    CK(cudaMalloc(&d_uptr, n * sizeof(void *)));
    CK(cudaMalloc(&d_cbytes, n * sizeof(size_t)));
    CK(cudaMalloc(&d_ubytes, n * sizeof(size_t)));
    CK(cudaMalloc(&d_actual, n * sizeof(size_t)));
    CK(cudaMalloc(&d_stat, n * sizeof(nvcompStatus_t)));

    CK(cudaEventRecord(t0, stream));
    CK(cudaMemcpyAsync(d_in, &hin[0], clen, cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_cptr, &h_cptr[0], n * sizeof(void *), cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_uptr, &h_uptr[0], n * sizeof(void *), cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_cbytes, &h_cbytes[0], n * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_ubytes, &h_ubytes[0], n * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CK(cudaEventRecord(t1, stream));

    size_t temp_bytes = 0;
    NK(nvcompBatchedGzipDecompressGetTempSizeAsync(n, max_ublock, opts, &temp_bytes, ulen));
    if (temp_bytes) CK(cudaMalloc(&d_temp, temp_bytes));

    NK(nvcompBatchedGzipDecompressAsync(
        (const void *const *) d_cptr, d_cbytes, d_ubytes, d_actual, n,
        d_temp, temp_bytes, (void *const *) d_uptr, opts, d_stat, stream));
    CK(cudaEventRecord(t2, stream));
    CK(cudaStreamSynchronize(stream));

    std::vector<nvcompStatus_t> stat(n);
    CK(cudaMemcpy(&stat[0], d_stat, n * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++)
      if (stat[i] != nvcompSuccess) { fprintf(stderr, "chunk %zu status %d\n", b0 + i, (int) stat[i]); mismatches++; }

    // Verify against the original FASTQ over the same span.
    hout.resize(ulen); horig.resize(ulen);
    CK(cudaMemcpy(&hout[0], d_out, ulen, cudaMemcpyDeviceToHost));
    got = 0;
    while (got < ulen) {
      ssize_t r = pread(fo, &horig[0] + got, ulen - got, e[b0].uoff + got);
      if (r <= 0) break;
      got += r;
    }
    if (memcmp(&hout[0], &horig[0], ulen) != 0) { mismatches++; fprintf(stderr, "batch %zu CONTENT MISMATCH\n", nbatch); }

    h2d_ms += now_ms(t0, t1);
    dec_ms += now_ms(t1, t2);
    total_out += ulen; total_in += clen; nbatch++;

    cudaFree(d_in); cudaFree(d_out); if (d_temp) cudaFree(d_temp);
    cudaFree(d_cptr); cudaFree(d_uptr); cudaFree(d_cbytes);
    cudaFree(d_ubytes); cudaFree(d_actual); cudaFree(d_stat);
  }

  printf("backend %d, %zu blocks/batch, %zu batches\n", backend, batch, nbatch);
  printf("  H2D          %8.1f ms   %6.2f GB/s (compressed in)\n",
         h2d_ms, total_in / (h2d_ms / 1000.0) / 1e9);
  printf("  decompress   %8.1f ms   %6.2f GB/s (uncompressed out)\n",
         dec_ms, total_out / (dec_ms / 1000.0) / 1e9);
  printf("  H2D+decomp   %8.1f ms   %6.2f GB/s (uncompressed out)\n",
         h2d_ms + dec_ms, total_out / ((h2d_ms + dec_ms) / 1000.0) / 1e9);
  printf("  verification: %s\n", mismatches == 0 ? "ALL BLOCKS MATCH ORIGINAL FASTQ" : "FAILED");
  return mismatches ? 2 : 0;
}
