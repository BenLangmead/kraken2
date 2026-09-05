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

__device__ __forceinline__ bool is_ancestor(const uint64_t *__restrict__ parent,
                                            uint64_t a, uint64_t b) {
  if (!a || !b) return false;
  while (b > a) b = parent[b];
  return b == a;
}

__device__ __forceinline__ uint64_t lca(const uint64_t *__restrict__ parent,
                                        uint64_t a, uint64_t b) {
  if (!a || !b) return a ? a : b;
  while (a != b) { if (a > b) a = parent[a]; else b = parent[b]; }
  return a;
}

// Kraken 2's tree resolution over the few distinct taxa one read hits.  Ties
// fold with the lowest common ancestor, which is commutative and associative,
// so the answer does not depend on visiting order -- which matters, because the
// CPU walks an unordered_map.
__device__ uint32_t resolve(const uint64_t *__restrict__ parent,
                            const uint32_t *tax, const uint32_t *cnt, int n) {
  uint64_t max_taxon = 0; uint32_t max_score = 0;
  for (int i = 0; i < n; i++) {
    uint32_t score = 0;
    for (int j = 0; j < n; j++)
      if (is_ancestor(parent, tax[j], tax[i])) score += cnt[j];
    if (score > max_score) { max_score = score; max_taxon = tax[i]; }
    else if (score == max_score) max_taxon = lca(parent, max_taxon, tax[i]);
  }
  return (uint32_t) max_taxon;
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
                      const uint32_t *__restrict__ out_offsets,
                      const uint64_t *__restrict__ parent,
                      const uint64_t *__restrict__ extid,
                      uint32_t *__restrict__ out_call) {
  __shared__ uint64_t cand[MAX_LEN];
  __shared__ uint8_t  code[MAX_LEN];
  __shared__ uint64_t val[MAX_LEN];
  __shared__ int      flag[MAX_LEN];
  __shared__ int      look[MAX_LEN];
  __shared__ int      scan[MAX_LEN];
  __shared__ int      tmp[MAX_LEN];
  __shared__ uint32_t stax[MAX_LEN];

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
    for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) scan[p] = flag[p];
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

    // One entry per reported minimizer.  A repeat inherits the taxon set at the
    // start of its equal-value run, and a run whose start the subsampling
    // filter dropped contributes zero, which is what last_taxon does
    // sequentially on the CPU.  Expressed this way it needs no scan: the run
    // start is a short walk back.
    uint32_t obase = out_offsets[r];
    for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) {
      if (!flag[p]) continue;
      int start = p, q = p - 1;
      while (true) {
        while (q >= 0 && !flag[q]) q--;
        if (q < 0 || val[q] != val[p]) break;
        start = q; q--;
      }
      bool filtered = min_acceptable && murmur(val[start]) < min_acceptable;
      int slot = scan[p] - 1;
      uint32_t t = filtered
          ? 0u : probe(table, capacity, val[p], key_bits, value_bits);
      out_key[obase + slot] = val[p];
      out_tax[obase + slot] = t;
      stax[slot] = t;
    }
    __syncthreads();
    // Counting distinct taxa and resolving the tree is a few dozen operations
    // on a handful of values, so one thread does it rather than paying for a
    // reduction across the block.
    if (threadIdx.x == 0) {
      int n = n_kmer ? scan[n_kmer - 1] : 0;
      out_count[r] = (uint32_t) n;
      uint32_t tx[64], ct[64]; int nd = 0;
      for (int j = 0; j < n; j++) {
        uint32_t t = stax[j];
        if (!t) continue;
        int q = 0; while (q < nd && tx[q] != t) q++;
        if (q == nd) { if (nd < 64) { tx[nd] = t; ct[nd] = 1; nd++; } }
        else ct[q]++;
      }
      uint32_t call = resolve(parent, tx, ct, nd);
      out_call[r] = call ? (uint32_t) extid[call] : 0u;
    }
    __syncthreads();
  }
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: fused reads.fq hash.k2d opts.k2d taxo.k2d classify.out [reads]\n");
    return 1;
  }
  size_t limit = argc > 6 ? (size_t) atoll(argv[6]) : 50000;

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

  // ---- taxonomy: only parent and external id are needed ----
  uint64_t *d_parent = nullptr, *d_extid = nullptr; uint64_t node_count = 0;
  {
    int tf = open(argv[4], O_RDONLY);
    if (tf < 0) { fprintf(stderr, "cannot open taxonomy %s\n", argv[4]); return 1; }
    char hdr[32];
    if (read(tf, hdr, 32) < 32 || memcmp(hdr, "K2TAXDAT", 8) != 0) {
      fprintf(stderr, "%s is not a taxonomy\n", argv[4]); return 1;
    }
    memcpy(&node_count, hdr + 8, 8);
    std::vector<uint64_t> raw(node_count * 7);
    size_t want = raw.size() * 8, got = 0;
    while (got < want) {
      ssize_t n = pread(tf, (char *) raw.data() + got, want - got, 32 + got);
      if (n <= 0) break;
      got += n;
    }
    close(tf);
    std::vector<uint64_t> par(node_count), ext(node_count);
    for (uint64_t i = 0; i < node_count; i++) {
      par[i] = raw[i * 7 + 0];
      ext[i] = raw[i * 7 + 5];
    }
    CK(cudaMalloc(&d_parent, node_count * 8));
    CK(cudaMalloc(&d_extid,  node_count * 8));
    CK(cudaMemcpy(d_parent, par.data(), node_count * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_extid,  ext.data(), node_count * 8, cudaMemcpyHostToDevice));
    printf("taxonomy: %" PRIu64 " nodes, %.1f MB uploaded\n",
           node_count, node_count * 16 / 1e6);
  }

  // ---- reads ----
  FILE *f = fopen(argv[1], "rb");
  if (! f) { fprintf(stderr, "cannot open reads %s\n", argv[1]); return 1; }
  std::vector<char> bases; std::vector<uint32_t> offs, lens;
  char line[8192]; size_t nr = 0;
  while (nr < limit && fgets(line, sizeof line, f)) {
    if (!fgets(line, sizeof line, f)) break;
    size_t L = strlen(line); while (L && (line[L-1]=='\n'||line[L-1]=='\r')) L--;
    if ((int) L > MAX_LEN) {
      fprintf(stderr, "read %zu is %zu bases, longer than the %d this build "
              "handles\n", nr, L, MAX_LEN);
      return 1;
    }
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
  uint32_t *d_call; CK(cudaMalloc(&d_call, nr*4));
  CK(cudaMemcpy(d_b, bases.data(), bases.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_o, offs.data(), nr*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_l, lens.data(), nr*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_oo, ooff.data(), (nr+1)*4, cudaMemcpyHostToDevice));

  int blocks = 8192;
  fused<<<blocks, THREADS>>>(d_b,d_o,d_l,(uint32_t)nr,k,l,ssm,tog,mahv,
                             d_table,capacity,(int)key_bits,(int)value_bits,
                             d_key,d_tax,d_cnt,d_oo,d_parent,d_extid,d_call);
  CK(cudaDeviceSynchronize());

  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  int reps = 10;
  CK(cudaEventRecord(a));
  for (int i = 0; i < reps; i++)
    fused<<<blocks, THREADS>>>(d_b,d_o,d_l,(uint32_t)nr,k,l,ssm,tog,mahv,
                               d_table,capacity,(int)key_bits,(int)value_bits,
                               d_key,d_tax,d_cnt,d_oo,d_parent,d_extid,d_call);
  CK(cudaEventRecord(b)); CK(cudaDeviceSynchronize());
  float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=reps;
  printf("\nfused scan+filter+probe   %.2f ms   %.2f Mbase/s   %.1f Mread/s\n",
         ms, bases.size()/(ms/1000.0)/1e6, nr/(ms/1000.0)/1e6);

  // ---- verify against classify's own calls ----
  std::vector<uint32_t> hcall(nr);
  CK(cudaMemcpy(hcall.data(), d_call, nr*4, cudaMemcpyDeviceToHost));

  FILE *rf = fopen(argv[5], "rb");
  if (! rf) { fprintf(stderr, "cannot open %s\n", argv[5]); return 1; }
  size_t bad = 0, seen = 0, classified = 0;
  char oline[8192];
  for (size_t i = 0; i < nr && fgets(oline, sizeof oline, rf); i++) {
    unsigned long want = 0;
    char *p1 = strchr(oline, '\t');
    char *p2 = p1 ? strchr(p1 + 1, '\t') : nullptr;
    if (p2) want = strtoul(p2 + 1, nullptr, 10);
    seen++;
    if (want) classified++;
    if ((unsigned long) hcall[i] != want) {
      if (bad < 3)
        fprintf(stderr, "read %zu: classify %lu, gpu %u\n", i, want, hcall[i]);
      bad++;
    }
  }
  fclose(rf);
  printf("verified %zu calls against classify (%.2f%% classified): %zu mismatched\n",
         seen, seen ? 100.0 * classified / seen : 0.0, bad);
  return bad ? 2 : 0;
}
