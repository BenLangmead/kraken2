// Scan, subsample and probe, fused, against the real Kraken 2 hash table.
//
// The three stages have been measured apart; this is the first time they run
// together against a real 16 GB table, which is what exercises the parts that
// isolated benchmarks cannot: that the device MurmurHash3 agrees with the
// host's, that the compact cell decoding is right, that the subsampling filter
// selects the same minimizers, and that a table of this size can simply live on
// the device while reads stream past it.
//
// Correctness is checked against k2mindump's output rather than asserted: for
// every read, the sequence of (minimizer, taxon) pairs the device produces must
// equal the sequence the CPU produced, in order.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime_api.h>

#define CK(x) do { cudaError_t e=(x); if (e!=cudaSuccess) { \
  fprintf(stderr,"cuda %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

static const int MAX_LEN = 320;
static const int THREADS = 128;

__host__ __device__ __forceinline__ uint64_t murmur(uint64_t k) {
  k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33; return k;
}

__device__ __forceinline__ uint64_t revcomp(uint64_t kmer, int n) {
  kmer = ((kmer & 0xCCCCCCCCCCCCCCCCULL) >> 2) | ((kmer & 0x3333333333333333ULL) << 2);
  kmer = ((kmer & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | ((kmer & 0x0F0F0F0F0F0F0F0FULL) << 4);
  kmer = ((kmer & 0xFF00FF00FF00FF00ULL) >> 8) | ((kmer & 0x00FF00FF00FF00FFULL) << 8);
  kmer = ((kmer & 0xFFFF0000FFFF0000ULL) >> 16) | ((kmer & 0x0000FFFF0000FFFFULL) << 16);
  kmer = (kmer >> 32) | (kmer << 32);
  return ((~kmer) >> (64 - n * 2)) & ((1ULL << (n * 2)) - 1);
}

// Linear probing over 4-byte cells, key in the high bits, value in the low.
// A value of zero marks an empty cell, which is also the miss return.
__device__ __forceinline__ uint32_t probe(const uint32_t *__restrict__ table,
                                          uint64_t capacity, uint64_t key,
                                          int key_bits, int value_bits) {
  uint64_t hc = murmur(key);
  uint32_t compacted = (uint32_t) (hc >> (64 - key_bits));
  uint64_t idx = hc % capacity;
  uint64_t first = idx;
  const uint32_t vmask = (1u << value_bits) - 1;
  while (true) {
    uint32_t cell = table[idx];
    uint32_t val = cell & vmask;
    if (!val) return 0;
    if ((cell >> value_bits) == compacted) return val;
    if (++idx == capacity) idx = 0;
    if (idx == first) return 0;
  }
}

__global__ void fused(const char *__restrict__ bases,
                      const uint32_t *__restrict__ offsets,
                      const uint32_t *__restrict__ lengths,
                      uint32_t n_reads, int k, int l,
                      uint64_t spaced_seed_mask, uint64_t toggle_mask,
                      uint64_t min_acceptable,
                      const uint32_t *__restrict__ table, uint64_t capacity,
                      int key_bits, int value_bits,
                      uint64_t *__restrict__ out_key,
                      uint32_t *__restrict__ out_tax,
                      uint32_t *__restrict__ out_count,
                      const uint32_t *__restrict__ out_offsets) {
  __shared__ uint64_t cand[MAX_LEN];
  __shared__ uint8_t  code[MAX_LEN];
  __shared__ uint64_t val[MAX_LEN];
  __shared__ int      flag[MAX_LEN];
  __shared__ int      look[MAX_LEN];
  __shared__ int      scan[MAX_LEN];
  __shared__ int      tmp[MAX_LEN];

  for (uint32_t r = blockIdx.x; r < n_reads; r += gridDim.x) {
    const char *seq = bases + offsets[r];
    int len = (int) lengths[r];
    if (len > MAX_LEN) len = MAX_LEN;

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
      char c = seq[i]; uint8_t v;
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

    int n_lmer = len - l + 1; if (n_lmer < 1) n_lmer = 0;
    const uint64_t lmask = (l * 2 >= 64) ? ~0ULL : ((1ULL << (l * 2)) - 1);
    for (int i = threadIdx.x; i < n_lmer; i += blockDim.x) {
      uint64_t lmer = 0; bool amb = false;
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

    int n_kmer = len - k + 1; if (n_kmer < 1) n_kmer = 0;
    int win = k - l + 1;

    // A position reports a minimizer when the winning l-mer position changes,
    // which is what the CPU's deque does; the value alone is not enough,
    // because inside a tandem repeat the winner can change while the value
    // stays equal.
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
      val[p] = ok ? (m ^ toggle_mask) : 0;
      flag[p] = (ok && (!pok || arg != parg)) ? 1 : 0;
    }
    __syncthreads();

    // Consecutive equal minimizers are one lookup, and the subsampling filter
    // drops most of what is left.  These are two independent rejections and
    // have to stay in separate arrays: a minimizer the filter drops still
    // counts as the previous reported one for duplicate detection, because
    // ClassifySequence assigns last_minimizer before deciding to skip.  Writing
    // both into flag[] would also race, since other threads read flag[] while
    // walking back to find the previous reporting position.
    for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) {
      int keep = flag[p];
      if (keep) {
        int q = p - 1;
        while (q >= 0 && !flag[q]) q--;
        if (q >= 0 && val[q] == val[p]) keep = 0;
        else if (min_acceptable && murmur(val[p]) < min_acceptable) keep = 0;
      }
      look[p] = keep;
    }
    __syncthreads();

    // Stable compaction: output order has to match the CPU's, so an atomic
    // counter will not do.  Hillis-Steele over the flags gives each surviving
    // position its index.
    for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) scan[p] = look[p];
    __syncthreads();
    // Double buffered: reading and writing one array in place would race
    // between threads within a step.
    for (int off = 1; off < n_kmer; off <<= 1) {
      for (int p = threadIdx.x; p < n_kmer; p += blockDim.x)
        tmp[p] = (p >= off) ? scan[p] + scan[p - off] : scan[p];
      __syncthreads();
      for (int p = threadIdx.x; p < n_kmer; p += blockDim.x)
        scan[p] = tmp[p];
      __syncthreads();
    }

    uint32_t obase = out_offsets[r];
    for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) {
      if (!look[p]) continue;
      int slot = scan[p] - 1;
      uint64_t key = val[p];
      out_key[obase + slot] = key;
      out_tax[obase + slot] = probe(table, capacity, key, key_bits, value_bits);
    }
    if (threadIdx.x == 0)
      out_count[r] = n_kmer ? (uint32_t) scan[n_kmer - 1] : 0;
    __syncthreads();
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: fused reads.fq hash.k2d opts.k2d ref.k2mm [reads]\n");
    return 1;
  }
  size_t limit = argc > 5 ? (size_t) atoll(argv[5]) : 50000;

  // ---- index options ----
  uint64_t o[8];
  { int fd = open(argv[3], O_RDONLY);
    if (fd < 0 || read(fd, o, sizeof o) < (ssize_t) sizeof o) { perror("opts"); return 1; }
    close(fd); }
  int k = (int) o[0], l = (int) o[1];
  uint64_t ssm = o[2], tog = o[3], mahv = o[5];
  printf("k=%d l=%d ssm=0x%" PRIx64 " toggle=0x%" PRIx64 " min_hash=0x%" PRIx64 "\n",
         k, l, ssm, tog, mahv);

  // ---- hash table ----
  uint64_t hdr[4];
  int hf = open(argv[2], O_RDONLY);
  if (hf < 0 || read(hf, hdr, sizeof hdr) < (ssize_t) sizeof hdr) { perror("hash"); return 1; }
  uint64_t capacity = hdr[0], key_bits = hdr[2], value_bits = hdr[3];
  size_t tbytes = capacity * 4;
  printf("table capacity %" PRIu64 ", %.2f GB, key_bits %" PRIu64 ", value_bits %" PRIu64 "\n",
         capacity, tbytes / 1e9, key_bits, value_bits);

  uint32_t *d_table = nullptr;
  CK(cudaMalloc(&d_table, tbytes));
  { const size_t CH = 256u << 20;
    std::vector<char> buf(CH);
    size_t done = 0;
    while (done < tbytes) {
      size_t want = tbytes - done < CH ? tbytes - done : CH;
      size_t got = 0;
      while (got < want) {
        ssize_t n = pread(hf, &buf[0] + got, want - got, 32 + done + got);
        if (n <= 0) break;
        got += n;
      }
      CK(cudaMemcpy((char *) d_table + done, &buf[0], got, cudaMemcpyHostToDevice));
      done += got;
      if (got < want) break;
    }
    printf("table uploaded: %.2f GB\n", done / 1e9);
  }
  close(hf);

  // ---- reads ----
  FILE *f = fopen(argv[1], "rb");
  std::vector<char> bases; std::vector<uint32_t> offs, lens;
  char line[8192]; size_t nr = 0;
  while (nr < limit && fgets(line, sizeof line, f)) {
    if (!fgets(line, sizeof line, f)) break;
    size_t L = strlen(line); while (L && (line[L-1]=='\n'||line[L-1]=='\r')) L--;
    offs.push_back((uint32_t) bases.size()); lens.push_back((uint32_t) L);
    bases.insert(bases.end(), line, line + L);
    if (!fgets(line, sizeof line, f)) break;
    if (!fgets(line, sizeof line, f)) break;
    nr++;
  }
  fclose(f);
  std::vector<uint32_t> ooff(nr + 1); uint64_t tot = 0;
  for (size_t i = 0; i < nr; i++) {
    ooff[i] = (uint32_t) tot;
    int nk = (int) lens[i] - k + 1; if (nk < 0) nk = 0;
    tot += nk;
  }
  ooff[nr] = (uint32_t) tot;
  printf("reads %zu, %.2f MB bases, %" PRIu64 " k-mer positions\n", nr, bases.size()/1e6, tot);

  char *d_b; uint32_t *d_o, *d_l, *d_oo, *d_tax, *d_cnt; uint64_t *d_key;
  CK(cudaMalloc(&d_b, bases.size())); CK(cudaMalloc(&d_o, nr*4));
  CK(cudaMalloc(&d_l, nr*4)); CK(cudaMalloc(&d_oo, (nr+1)*4));
  CK(cudaMalloc(&d_key, tot*8)); CK(cudaMalloc(&d_tax, tot*4)); CK(cudaMalloc(&d_cnt, nr*4));
  CK(cudaMemcpy(d_b, bases.data(), bases.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_o, offs.data(), nr*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_l, lens.data(), nr*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_oo, ooff.data(), (nr+1)*4, cudaMemcpyHostToDevice));

  int blocks = 8192;
  fused<<<blocks, THREADS>>>(d_b,d_o,d_l,(uint32_t)nr,k,l,ssm,tog,mahv,
                             d_table,capacity,(int)key_bits,(int)value_bits,
                             d_key,d_tax,d_cnt,d_oo);
  CK(cudaDeviceSynchronize());

  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  int reps = 10;
  CK(cudaEventRecord(a));
  for (int i = 0; i < reps; i++)
    fused<<<blocks, THREADS>>>(d_b,d_o,d_l,(uint32_t)nr,k,l,ssm,tog,mahv,
                               d_table,capacity,(int)key_bits,(int)value_bits,
                               d_key,d_tax,d_cnt,d_oo);
  CK(cudaEventRecord(b)); CK(cudaDeviceSynchronize());
  float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=reps;
  printf("\nfused scan+filter+probe   %.2f ms   %.2f Mbase/s   %.1f Mread/s\n",
         ms, bases.size()/(ms/1000.0)/1e6, nr/(ms/1000.0)/1e6);

  // ---- verify ----
  std::vector<uint64_t> hk(tot); std::vector<uint32_t> ht(tot), hc(nr);
  CK(cudaMemcpy(hk.data(), d_key, tot*8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ht.data(), d_tax, tot*4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hc.data(), d_cnt, nr*4, cudaMemcpyDeviceToHost));

  FILE *rf = fopen(argv[4], "rb");
  char magic[8]; size_t z = fread(magic,1,8,rf); (void)z;
  char flags; z = fread(&flags,1,1,rf);
  uint64_t rn; z = fread(&rn,8,1,rf);
  if (flags != 2) { fprintf(stderr, "oracle was not built with -H\n"); return 1; }
  size_t cmp = nr < rn ? nr : rn, bad = 0, pairs = 0, hits = 0;
  for (size_t i = 0; i < cmp; i++) {
    uint32_t c; if (fread(&c,4,1,rf)!=1) break;
    bool ok = (c == hc[i]);
    for (uint32_t j = 0; j < c; j++) {
      uint64_t key; uint32_t tax;
      if (fread(&key,8,1,rf)!=1 || fread(&tax,4,1,rf)!=1) { ok=false; break; }
      pairs++; if (tax) hits++;
      if (ok && (hk[ooff[i]+j] != key || ht[ooff[i]+j] != tax)) ok = false;
    }
    if (!ok) { if (bad<3) fprintf(stderr,"read %zu differs (cpu %u, gpu %u pairs)\n",i,c,hc[i]); bad++; }
  }
  fclose(rf);
  printf("verified %zu reads, %zu lookups (%.1f%% hit): %zu mismatched\n",
         cmp, pairs, pairs?100.0*hits/pairs:0, bad);
  return bad ? 2 : 0;
}
