// Device kernels shared by the verification harness (fused.cu) and the
// end-to-end pipeline (k2gpu.cu), so the two cannot drift apart.

#ifndef KRAKEN2_GPU_KERNELS_CUH_
#define KRAKEN2_GPU_KERNELS_CUH_

#include <cstdint>
#include <cuda_runtime_api.h>

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
    uint32_t obase = out_offsets ? out_offsets[r] : 0u;
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
      // The per-token arrays exist for the verification harness.  The pipeline
      // wants only the call, and passes null rather than sizing them.
      if (out_key) out_key[obase + slot] = val[p];
      if (out_tax) out_tax[obase + slot] = t;
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

// Splits a run of record-aligned BGZF blocks into per-read sequence spans.
//
// Every block begins at a record start, so a block's records are simply its
// lines in fours: record i's bases are the line after its header.  Finding the
// lines is a cooperative two-pass count-then-write over the block's bytes,
// which is cheaper than one thread walking 60 KB alone.
__global__ void parse_blocks(const char *__restrict__ buf,
                             const uint64_t *__restrict__ blk_uoff,
                             const uint64_t *__restrict__ blk_rec,
                             uint64_t base_uoff, uint32_t n_blocks,
                             uint64_t base_rec,
                             uint32_t *__restrict__ out_off,
                             uint32_t *__restrict__ out_len) {
  __shared__ int nl_count[256];
  __shared__ int nl_scan[256];
  extern __shared__ uint32_t nl[];

  for (uint32_t b = blockIdx.x; b < n_blocks; b += gridDim.x) {
    uint64_t lo = blk_uoff[b] - base_uoff;
    uint64_t hi = blk_uoff[b + 1] - base_uoff;
    uint32_t len = (uint32_t) (hi - lo);
    const char *p = buf + lo;
    uint32_t recs = (uint32_t) (blk_rec[b + 1] - blk_rec[b]);

    uint32_t per = (len + blockDim.x - 1) / blockDim.x;
    uint32_t s0 = threadIdx.x * per;
    uint32_t s1 = s0 + per < len ? s0 + per : len;

    int c = 0;
    for (uint32_t i = s0; i < s1; i++) if (p[i] == '\n') c++;
    nl_count[threadIdx.x] = c;
    __syncthreads();

    if (threadIdx.x == 0) {
      int run = 0;
      for (uint32_t t = 0; t < blockDim.x; t++) { nl_scan[t] = run; run += nl_count[t]; }
    }
    __syncthreads();

    int w = nl_scan[threadIdx.x];
    for (uint32_t i = s0; i < s1; i++) if (p[i] == '\n') nl[w++] = i;
    __syncthreads();

    // Record r: header is line 4r, bases are line 4r+1.
    for (uint32_t r = threadIdx.x; r < recs; r += blockDim.x) {
      uint32_t seq_beg = nl[4 * r] + 1;
      uint32_t seq_end = nl[4 * r + 1];
      uint64_t idx = blk_rec[b] - base_rec + r;
      out_off[idx] = (uint32_t) (lo + seq_beg);
      out_len[idx] = seq_end - seq_beg;
    }
    __syncthreads();
  }
}

#endif  // KRAKEN2_GPU_KERNELS_CUH_
