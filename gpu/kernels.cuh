// Device kernels shared by the verification harness (fused.cu) and the
// end-to-end pipeline (k2gpu.cu), so the two cannot drift apart.

#ifndef KRAKEN2_GPU_KERNELS_CUH_
#define KRAKEN2_GPU_KERNELS_CUH_

#include <cstdint>
#include <cuda_runtime_api.h>
#include "table.h"

static const int MAX_LEN = 320;
static const int THREADS = 128;

__host__ __device__ __forceinline__ uint64_t murmur(uint64_t k) { return k2t::murmur(k); }

__device__ __forceinline__ uint64_t revcomp(uint64_t kmer, int n) {
  kmer = ((kmer & 0xCCCCCCCCCCCCCCCCULL) >> 2) | ((kmer & 0x3333333333333333ULL) << 2);
  kmer = ((kmer & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | ((kmer & 0x0F0F0F0F0F0F0F0FULL) << 4);
  kmer = ((kmer & 0xFF00FF00FF00FF00ULL) >> 8) | ((kmer & 0x00FF00FF00FF00FFULL) << 8);
  kmer = ((kmer & 0xFFFF0000FFFF0000ULL) >> 16) | ((kmer & 0x0000FFFF0000FFFFULL) << 16);
  kmer = (kmer >> 32) | (kmer << 32);
  return ((~kmer) >> (64 - n * 2)) & ((1ULL << (n * 2)) - 1);
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

__device__ __forceinline__ int put_u64(char *p, uint64_t v) {
  if (v == 0) { p[0] = '0'; return 1; }
  char t[24]; int n = 0;
  while (v) { t[n++] = (char) ('0' + (v % 10)); v /= 10; }
  for (int i = 0; i < n; i++) p[i] = t[n - 1 - i];
  return n;
}

__global__ void fused(const char *__restrict__ bases,
                      const uint32_t *__restrict__ offsets,
                      const uint32_t *__restrict__ lengths,
                      uint32_t n_reads, int k, int l,
                      uint64_t spaced_seed_mask, uint64_t toggle_mask,
                      uint64_t min_acceptable,
                      const k2t::Table table, int skip_probe,
                      uint64_t *__restrict__ out_key,
                      uint32_t *__restrict__ out_tax,
                      uint32_t *__restrict__ out_count,
                      const uint32_t *__restrict__ out_offsets,
                      const uint64_t *__restrict__ parent,
                      const uint64_t *__restrict__ extid,
                      uint32_t *__restrict__ out_call,
                      const uint32_t *__restrict__ hdr_off,
                      const uint32_t *__restrict__ hdr_len,
                      char *__restrict__ out_text, uint32_t text_stride,
                      uint32_t *__restrict__ out_linelen,
                      uint32_t *__restrict__ out_overflow,
                      uint8_t *__restrict__ out_amb) {
  __shared__ uint64_t cand[MAX_LEN];
  __shared__ uint8_t  code[MAX_LEN];
  __shared__ uint64_t val[MAX_LEN];
  __shared__ int      flag[MAX_LEN];
  __shared__ int      look[MAX_LEN];
  __shared__ int      scan[MAX_LEN];
  __shared__ int      tmp[MAX_LEN];
  __shared__ uint32_t stax[MAX_LEN];
  __shared__ uint32_t ptax[MAX_LEN];
  __shared__ int amb_read;

  for (uint32_t r = blockIdx.x; r < n_reads; r += gridDim.x) {
    const char *seq = bases + offsets[r];
    int len = (int) lengths[r];
    if (len > MAX_LEN) len = MAX_LEN;

    if (threadIdx.x == 0) amb_read = 0;
    __syncthreads();
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
      if (v == 4) amb_read = 1;
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

    // Every k-mer whose window holds k - l + 1 unambiguous l-mers reports a
    // token, as MinimizerScanner::NextMinimizer does once per k-mer.
    for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) {
      uint64_t m = ~0ULL; bool ok = true;
      for (int j = 0; j < win; j++) {
        uint64_t c = cand[p + j];
        if (c == ~0ULL) { ok = false; break; }
        if (c < m) m = c;
      }
      val[p] = ok ? (m ^ toggle_mask) : 0;
      flag[p] = ok ? 1 : 0;
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

    // One table lookup per run of equal consecutive minimizers, at the run's
    // first k-mer, unless the subsampling filter dropped it; look[] marks
    // exactly those positions.  This is the CPU's rule: ClassifySequence looks
    // a minimizer up only when it differs from the previous one.
    for (int p = threadIdx.x; p < n_kmer; p += blockDim.x)
      // skip_probe exists for timing: it removes the table lookup and nothing
      // else, so the difference in kernel time is the lookup's cost.
      if (look[p]) ptax[p] = skip_probe ? 0u : table.get(val[p]);
    __syncthreads();

    // One entry per reported k-mer.  A repeat inherits the taxon found at the
    // start of its equal-value run, and a run whose start the subsampling
    // filter dropped contributes zero, which is what last_taxon does
    // sequentially on the CPU.  The run start is a short walk back.
    uint32_t obase = out_offsets ? out_offsets[r] : 0u;
    for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) {
      if (!flag[p]) continue;
      int start = p, q = p - 1;
      while (true) {
        while (q >= 0 && !flag[q]) q--;
        if (q < 0 || val[q] != val[p]) break;
        start = q; q--;
      }
      int slot = scan[p] - 1;
      uint32_t t = look[start] ? ptax[start] : 0u;
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
      if (out_count) out_count[r] = (uint32_t) n;
      uint32_t tx[64], ct[64]; int nd = 0;
      for (int j = 0; j < n; j++) {
        uint32_t t = stax[j];
        if (!t) continue;
        int q = 0; while (q < nd && tx[q] != t) q++;
        if (q == nd) { if (nd < 64) { tx[nd] = t; ct[nd] = 1; nd++; } }
        else ct[q]++;
      }
      uint32_t call = resolve(parent, tx, ct, nd);
      uint64_t ext_call = call ? extid[call] : 0;
      out_call[r] = (uint32_t) ext_call;

      // A read carrying an ambiguous base is handed back to the host.  Kraken 2
      // emits a token per character while its scanner refills after one, which
      // this kernel does not reproduce, and getting it wrong changes the
      // confidence denominator.  Rather than model that here for the roughly one
      // read in a million that needs it, the line is left empty and the host
      // reclassifies with the original scanner.
      if (amb_read) {
        if (out_linelen) out_linelen[r] = 0;
        if (out_amb) out_amb[r] = 1;
      }
      else if (out_text) {
        // A hit list has one run per reported minimizer, so a long read with
        // no repeats can produce a long line.  Rather than size the slot for
        // the worst case everywhere, stop short of the end and report it, so a
        // too-narrow slot shows up as a counted failure instead of a
        // neighbouring line quietly overwritten.
        char *w = out_text + (size_t) r * text_stride;
        char *w0 = w;
        char *wend = w0 + text_stride - 24;
        *w++ = call ? 'C' : 'U'; *w++ = '\t';
        const char *h = bases + hdr_off[r];
        for (uint32_t i = 0; i < hdr_len[r]; i++) *w++ = h[i];
        *w++ = '\t';
        w += put_u64(w, ext_call); *w++ = '\t';
        w += put_u64(w, (uint64_t) len); *w++ = '\t';
        if (n == 0) { *w++='0'; *w++=':'; *w++='0'; }
        else {
          uint32_t last = stax[0]; int run = 1;
          bool over = false;
          for (int j = 1; j < n; j++) {
            if (stax[j] == last) { run++; continue; }
            if (w >= wend) { over = true; break; }
            w += put_u64(w, last ? extid[last] : 0);
            *w++ = ':'; w += put_u64(w, (uint64_t) run); *w++ = ' ';
            last = stax[j]; run = 1;
          }
          if (over) atomicAdd(out_overflow, 1u);
          w += put_u64(w, last ? extid[last] : 0);
          *w++ = ':'; w += put_u64(w, (uint64_t) run);
        }
        *w++ = '\n';
        out_linelen[r] = (uint32_t) (w - w0);
      }
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
                             uint32_t *__restrict__ out_len,
                             uint32_t *__restrict__ out_hoff,
                             uint32_t *__restrict__ out_hlen) {
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

    // A record is four lines, so a well formed block holds exactly 4*recs
    // newlines.  Anything else means the bytes are not what the index claims --
    // most likely decompression did not produce them -- and reading on would
    // hand the classifier wild offsets.  Emit empty reads instead, which it
    // already ignores, so the failure shows up as a mismatch rather than a
    // fault with no explanation.
    int total_nl = nl_scan[blockDim.x - 1] + nl_count[blockDim.x - 1];
    bool sane = (total_nl == (int) (4 * recs));

    for (uint32_t r = threadIdx.x; r < recs; r += blockDim.x) {
      uint64_t idx = blk_rec[b] - base_rec + r;
      if (!sane) {
        out_off[idx] = 0; out_len[idx] = 0;
        if (out_hoff) { out_hoff[idx] = 0; out_hlen[idx] = 0; }
        continue;
      }
      uint32_t seq_beg = nl[4 * r] + 1;
      uint32_t seq_end = nl[4 * r + 1];
      out_off[idx] = (uint32_t) (lo + seq_beg);
      out_len[idx] = seq_end > seq_beg ? seq_end - seq_beg : 0;
      if (out_hoff) {
        // The identifier is the header line past its '@', up to whitespace.
        uint32_t h_beg = (r == 0) ? 0 : nl[4 * r - 1] + 1;
        uint32_t h_end = nl[4 * r];
        if (h_beg < h_end && p[h_beg] == '@') h_beg++;
        uint32_t e = h_beg;
        while (e < h_end && p[e] != ' ' && p[e] != '\t') e++;
        out_hoff[idx] = (uint32_t) (lo + h_beg);
        out_hlen[idx] = e - h_beg;
      }
    }
    __syncthreads();
  }
}

// Exclusive prefix sum over per-read line lengths, in three passes.  Only a
// few hundred thousand values, so a simple version is ample and avoids pulling
// in a library for it.
__global__ void scan_pass1(const uint32_t *__restrict__ in,
                           uint32_t *__restrict__ out,
                           uint32_t *__restrict__ sums, uint32_t n) {
  __shared__ uint32_t a[256], b[256];
  uint32_t i = blockIdx.x * 256 + threadIdx.x;
  uint32_t v = i < n ? in[i] : 0u;
  a[threadIdx.x] = v;
  __syncthreads();
  for (int off = 1; off < 256; off <<= 1) {
    b[threadIdx.x] = threadIdx.x >= (uint32_t) off
        ? a[threadIdx.x] + a[threadIdx.x - off] : a[threadIdx.x];
    __syncthreads();
    a[threadIdx.x] = b[threadIdx.x];
    __syncthreads();
  }
  if (i < n) out[i] = a[threadIdx.x] - v;          // exclusive within the block
  if (threadIdx.x == 255) sums[blockIdx.x] = a[255];
}

__global__ void scan_pass2(uint32_t *__restrict__ sums, uint32_t nblk) {
  if (threadIdx.x || blockIdx.x) return;
  uint32_t run = 0;
  for (uint32_t i = 0; i < nblk; i++) { uint32_t v = sums[i]; sums[i] = run; run += v; }
}

__global__ void scan_pass3(uint32_t *__restrict__ out,
                           const uint32_t *__restrict__ sums, uint32_t n) {
  uint32_t i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) out[i] += sums[blockIdx.x];
}

// Gathers the formatted lines out of their fixed-width slots into one run of
// bytes, so the copy back to the host carries text and nothing else.
__global__ void compact_text(const char *__restrict__ src, uint32_t stride,
                             const uint32_t *__restrict__ len,
                             const uint32_t *__restrict__ off,
                             char *__restrict__ dst, uint32_t n) {
  for (uint32_t r = blockIdx.x; r < n; r += gridDim.x) {
    const char *s = src + (size_t) r * stride;
    char *d = dst + off[r];
    for (uint32_t i = threadIdx.x; i < len[r]; i += blockDim.x) d[i] = s[i];
  }
}

#endif  // KRAKEN2_GPU_KERNELS_CUH_
