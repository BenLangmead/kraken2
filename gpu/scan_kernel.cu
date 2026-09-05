// Minimizer scan on the GPU, checked against the CPU scanner's own output.
//
// The CPU keeps a monotonic deque because it is a streaming algorithm that sees
// one l-mer at a time.  On a device that structure is unnecessary and actively
// unhelpful: with k=35 and l=31 the window is k-l+1 = 5 l-mers, so every k-mer
// position can take the minimum of five adjacent candidates with no dependence
// on any other position.  The scan is therefore embarrassingly parallel, and
// the sequential-looking deque in the CPU code is an artifact of streaming, not
// of the problem.
//
// Each block handles one read: the bases are packed two bits each into shared
// memory, every thread builds the l-mer candidates at its own positions, and a
// second pass takes the five-wide window minimum.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <vector>
#include <cuda_runtime_api.h>

#define CK(x) do { cudaError_t e=(x); if (e!=cudaSuccess) { \
  fprintf(stderr,"cuda %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

static const int MAX_LEN = 320;          // bases per read this kernel handles
static const int MAX_LMERS = MAX_LEN;
static const int THREADS = 128;

__device__ __forceinline__ uint64_t revcomp(uint64_t kmer, int n) {
  kmer = ((kmer & 0xCCCCCCCCCCCCCCCCULL) >> 2) | ((kmer & 0x3333333333333333ULL) << 2);
  kmer = ((kmer & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | ((kmer & 0x0F0F0F0F0F0F0F0FULL) << 4);
  kmer = ((kmer & 0xFF00FF00FF00FF00ULL) >> 8) | ((kmer & 0x00FF00FF00FF00FFULL) << 8);
  kmer = ((kmer & 0xFFFF0000FFFF0000ULL) >> 16) | ((kmer & 0x0000FFFF0000FFFFULL) << 16);
  kmer = (kmer >> 32) | (kmer << 32);
  // revcom_version 1: complement after the shift, matching mmscanner.cc
  return ((~kmer) >> (64 - n * 2)) & ((1ULL << (n * 2)) - 1);
}

// One block per read.  Emits, per k-mer position, the minimizer, and a validity
// flag that is false wherever the window overlaps an ambiguous base.
__global__ void scan(const char *__restrict__ bases,
                     const uint32_t *__restrict__ offsets,
                     const uint32_t *__restrict__ lengths,
                     uint32_t n_reads, int k, int l,
                     uint64_t spaced_seed_mask, uint64_t toggle_mask,
                     uint64_t *__restrict__ out_min,
                     uint8_t *__restrict__ out_valid,
                     const uint32_t *__restrict__ out_offsets) {
  __shared__ uint64_t cand[MAX_LMERS];
  __shared__ uint8_t  code[MAX_LEN];

  for (uint32_t r = blockIdx.x; r < n_reads; r += gridDim.x) {
    const char *seq = bases + offsets[r];
    int len = (int) lengths[r];
    if (len > MAX_LEN) len = MAX_LEN;

    // Encode bases; 4 marks ambiguous.
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
      char c = seq[i];
      uint8_t v;
      switch (c) {
        case 'A': case 'a': v = 0; break;
        case 'C': case 'c': v = 1; break;
        case 'G': case 'g': v = 2; break;
        case 'T': case 't': v = 3; break;
        default:  v = 4; break;
      }
      code[i] = v;
    }
    __syncthreads();

    int n_lmer = len - l + 1;
    if (n_lmer < 1) n_lmer = 0;
    const uint64_t lmask = (l * 2 >= 64) ? ~0ULL : ((1ULL << (l * 2)) - 1);

    for (int i = threadIdx.x; i < n_lmer; i += blockDim.x) {
      uint64_t lmer = 0;
      bool amb = false;
      for (int j = 0; j < l; j++) {
        uint8_t v = code[i + j];
        if (v == 4) { amb = true; break; }
        lmer = (lmer << 2) | v;
      }
      if (amb) { cand[i] = ~0ULL; continue; }
      lmer &= lmask;
      uint64_t rc = revcomp(lmer, l);
      uint64_t canon = lmer < rc ? lmer : rc;
      if (spaced_seed_mask) canon &= spaced_seed_mask;
      cand[i] = canon ^ toggle_mask;
    }
    __syncthreads();

    int n_kmer = len - k + 1;
    if (n_kmer < 1) n_kmer = 0;
    int win = k - l + 1;
    uint32_t obase = out_offsets[r];
    // The CPU pops its deque with a strict >, so among equal candidates the
    // earliest position stays at the front.  A new minimizer is reported when
    // that position expires, which in a tandem repeat can emit the same value
    // twice running.  Matching means tracking which position wins, not which
    // value does.
    for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) {
      int arg = -1; uint64_t m = ~0ULL; bool ok = true;
      for (int j = 0; j < win; j++) {
        uint64_t c = cand[p + j];
        if (c == ~0ULL) { ok = false; break; }
        if (arg < 0 || c < m) { m = c; arg = p + j; }
      }
      int parg = -1; bool pok = false;
      if (p > 0) {
        uint64_t pm = ~0ULL; pok = true;
        for (int j = 0; j < win; j++) {
          uint64_t c = cand[p - 1 + j];
          if (c == ~0ULL) { pok = false; break; }
          if (parg < 0 || c < pm) { pm = c; parg = p - 1 + j; }
        }
      }
      out_min[obase + p] = ok ? (m ^ toggle_mask) : 0;
      out_valid[obase + p] = ok ? ((!pok || arg != parg) ? 1 : 2) : 0;
    }
    __syncthreads();
  }
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: scan_kernel reads.fq ref.k2mm [reads]\n"); return 1; }
  size_t limit = argc > 3 ? (size_t) atoll(argv[3]) : 200000;
  int k = 35, l = 31;
  uint64_t ssm = 0x3ffffffff3333333ULL, tog = 0xe37e28c4271b5a2dULL;

  // ---- load reads ----
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror("reads"); return 1; }
  std::vector<char> bases; std::vector<uint32_t> offs, lens;
  char line[4096];
  size_t nr = 0;
  while (nr < limit && fgets(line, sizeof line, f)) {
    if (!fgets(line, sizeof line, f)) break;
    size_t L = strlen(line); while (L && (line[L-1]=='\n'||line[L-1]=='\r')) L--;
    offs.push_back((uint32_t) bases.size());
    lens.push_back((uint32_t) L);
    bases.insert(bases.end(), line, line + L);
    fgets(line, sizeof line, f); fgets(line, sizeof line, f);
    nr++;
  }
  fclose(f);
  printf("loaded %zu reads, %.2f MB of bases\n", nr, bases.size()/1e6);

  std::vector<uint32_t> ooff(nr + 1); uint64_t tot = 0;
  for (size_t i = 0; i < nr; i++) {
    ooff[i] = (uint32_t) tot;
    int nk = (int) lens[i] - k + 1; if (nk < 0) nk = 0;
    tot += nk;
  }
  ooff[nr] = (uint32_t) tot;
  printf("k-mer positions: %" PRIu64 "\n", tot);

  char *d_b; uint32_t *d_o, *d_l, *d_oo; uint64_t *d_m; uint8_t *d_v;
  CK(cudaMalloc(&d_b, bases.size())); CK(cudaMalloc(&d_o, nr*4));
  CK(cudaMalloc(&d_l, nr*4)); CK(cudaMalloc(&d_oo, (nr+1)*4));
  CK(cudaMalloc(&d_m, tot*8)); CK(cudaMalloc(&d_v, tot));
  CK(cudaMemcpy(d_b, bases.data(), bases.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_o, offs.data(), nr*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_l, lens.data(), nr*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_oo, ooff.data(), (nr+1)*4, cudaMemcpyHostToDevice));

  int blocks = 4096;
  scan<<<blocks, THREADS>>>(d_b,d_o,d_l,(uint32_t)nr,k,l,ssm,tog,d_m,d_v,d_oo);
  CK(cudaDeviceSynchronize());

  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  int reps = 20;
  CK(cudaEventRecord(a));
  for (int i = 0; i < reps; i++)
    scan<<<blocks, THREADS>>>(d_b,d_o,d_l,(uint32_t)nr,k,l,ssm,tog,d_m,d_v,d_oo);
  CK(cudaEventRecord(b)); CK(cudaDeviceSynchronize());
  float ms=0; cudaEventElapsedTime(&ms,a,b); ms /= reps;
  printf("GPU scan   %.2f ms per pass   %.2f Mbase/s   %.2f Mkmer-pos/s\n",
         ms, bases.size()/(ms/1000.0)/1e6, tot/(ms/1000.0)/1e6);

  // ---- collapse consecutive duplicates and compare with the oracle ----
  std::vector<uint64_t> hm(tot); std::vector<uint8_t> hv(tot);
  CK(cudaMemcpy(hm.data(), d_m, tot*8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hv.data(), d_v, tot, cudaMemcpyDeviceToHost));

  FILE *rf = fopen(argv[2], "rb");
  if (!rf) { perror("ref"); return 1; }
  char magic[8]; size_t rd = fread(magic,1,8,rf); (void)rd;
  char flags; rd = fread(&flags,1,1,rf);
  uint64_t rn; rd = fread(&rn,8,1,rf);
  size_t cmp = nr < rn ? nr : rn, bad = 0, checked = 0;
  std::vector<uint64_t> ref;
  for (size_t i = 0; i < cmp; i++) {
    uint32_t c; if (fread(&c,4,1,rf)!=1) break;
    ref.resize(c); if (c && fread(ref.data(),8,c,rf)!=c) break;
    std::vector<uint64_t> got;
    bool gap = true;   // an ambiguous run forces the next minimizer to be
    for (uint32_t p = ooff[i]; p < ooff[i+1]; p++) {   // emitted even if equal
      if (!hv[p]) { gap = true; continue; }
      if (hv[p] == 1) got.push_back(hm[p]);   // 1 = the winning position changed
      gap = false;
    }
    checked++;
    if (got.size() != ref.size() || (c && memcmp(got.data(), ref.data(), c*8) != 0)) {
      if (bad < 3)
        fprintf(stderr, "read %zu: gpu %zu vs cpu %u minimizers\n", i, got.size(), c);
      bad++;
    }
  }
  fclose(rf);
  printf("verified %zu reads against the CPU scanner: %zu mismatched\n", checked, bad);
  return bad ? 2 : 0;
}
