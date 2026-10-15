// Device kernels shared by the verification harness (fused.cu) and the
// end-to-end pipeline (k2gpu.cu), so the two cannot drift apart.

#ifndef KRAKEN2_GPU_KERNELS_CUH_
#define KRAKEN2_GPU_KERNELS_CUH_

#include <cstdint>
#include <cuda_runtime_api.h>
#include "table.h"
#include "classify_core.h"

static const int MAX_LEN = 320;
static const int THREADS = 128;

__host__ __device__ __forceinline__ uint64_t murmur(uint64_t k) { return k2t::murmur(k); }

// What one launch of the fused kernel does.  CLASSIFY is the whole job for a
// table that is resident.  The other three serve a blocked table that is
// streamed in groups of whole partitions:
//   PROBE_GROUP   recompute the minimizers, look up those whose partition is
//                 in [part_lo, part_hi), and store each result in tok_tax
//   BIN           compute the minimizers once and append each lookup's hash
//                 to its partition's bin, recording in tok_ref which bin
//                 entry will hold its result, for probe_bins
//   FINISH        recompute the minimizers, take each lookup's taxon from
//                 tok_tax (or through tok_ref from bin_v), and resolve and
//                 format as CLASSIFY does
// Every key lives in exactly one partition and a lookup never leaves it, so
// after all groups every lookup has the value a resident table gives.
enum Phase { CLASSIFY = 0, PROBE_GROUP = 1, BIN = 2, FINISH = 3 };

struct StreamArgs {
  int phase;
  // Per-token state: token t of mate m of unit r of this launch is entry
  // tok_base + r * tok_stride + m * mate_stride + t.  Only tokens that are
  // lookups are written and read.  tok_tax holds taxa (rescan); tok_ref holds
  // bin entry indices (bins).
  uint16_t *tok_tax;
  uint32_t *tok_ref;
  uint64_t tok_base;
  uint32_t tok_stride, mate_stride;
  uint32_t part_lo, part_hi;
  // Bins for BIN: partition b of this launch's chunk holds bin_cnt[b]
  // entries at bin_base + b * bin_cap + i; probe_bins stores each result at
  // the same index of bin_v.
  uint64_t *bin_h;
  uint16_t *bin_v;
  uint32_t *bin_cnt;
  uint32_t bin_cap;
  uint64_t bin_base;
};

// The work of one lookup token in each phase.  pos is the token's index in
// tok_tax / tok_ref.  Returns the taxon in CLASSIFY and FINISH.
__device__ __forceinline__ uint32_t lookup_token(const k2t::Table &table, const StreamArgs &sa,
                                                 int skip_probe, uint64_t key, uint64_t pos) {
  switch (sa.phase) {
    case PROBE_GROUP: {
      uint64_t h = table.G.hash(key);
      uint32_t part = (uint32_t) table.G.part_of(h);
      if (part >= sa.part_lo && part < sa.part_hi)
        sa.tok_tax[pos] = (uint16_t) table.get_blocked_hash(h);
      return 0;
    }
    case BIN: {
      uint64_t h = table.G.hash(key);
      uint32_t part = (uint32_t) table.G.part_of(h);
      uint32_t at = atomicAdd(&sa.bin_cnt[part], 1u);
      if (at < sa.bin_cap) {
        uint64_t e = sa.bin_base + (uint64_t) part * sa.bin_cap + at;
        sa.bin_h[e] = h;
        sa.tok_ref[pos] = (uint32_t) e;
      }
      return 0;
    }
    case FINISH:
      return sa.tok_ref ? (uint32_t) sa.bin_v[sa.tok_ref[pos]] : (uint32_t) sa.tok_tax[pos];
    default:
      // skip_probe exists for timing: it removes the table lookup and nothing
      // else, so the difference in kernel time is the lookup's cost.
      return skip_probe ? 0u : table.get(key);
  }
}

// One block per unit, where a unit is a read or, with mates == 2, a pair of
// consecutive records.  A mate with no ambiguous base is scanned in parallel:
// every k-mer is a token.  A mate with an ambiguous base is scanned by thread
// 0 with Kraken 2's own scanner logic (k2c::Scanner), which reproduces the
// ambiguous tokens it emits while refilling after such a base.  Tokens of both
// mates and the mate border go into stax, then one thread resolves the call
// and formats the line.
__global__ void fused(const char *__restrict__ bases,
                      const uint32_t *__restrict__ offsets,
                      const uint32_t *__restrict__ lengths,
                      uint32_t n_units, int mates, int k, int l,
                      uint64_t spaced_seed_mask, uint64_t toggle_mask,
                      uint64_t min_acceptable,
                      const k2t::Table table, int skip_probe, const StreamArgs sa,
                      const uint64_t *__restrict__ parent,
                      const uint64_t *__restrict__ extid,
                      uint32_t *__restrict__ out_call,
                      const uint32_t *__restrict__ hdr_off,
                      const uint32_t *__restrict__ hdr_len,
                      char *__restrict__ out_text, uint32_t text_stride,
                      uint32_t *__restrict__ out_linelen,
                      uint32_t *__restrict__ out_overflow) {
  __shared__ uint64_t cand[MAX_LEN];
  __shared__ uint8_t  code[MAX_LEN];
  __shared__ uint64_t val[MAX_LEN];
  __shared__ int      look[MAX_LEN];
  __shared__ uint32_t ptax[MAX_LEN];
  __shared__ uint32_t stax[2 * MAX_LEN + 1];
  __shared__ int amb_mate, n_tok, mate_len[2];
  const bool tokens_wanted = sa.phase == CLASSIFY || sa.phase == FINISH;

  for (uint32_t r = blockIdx.x; r < n_units; r += gridDim.x) {
    if (threadIdx.x == 0) n_tok = 0;
    for (int m = 0; m < mates; m++) {
      uint32_t rec = r * mates + m;
      const char *seq = bases + offsets[rec];
      int len = (int) lengths[rec];
      if (threadIdx.x == 0) {
        amb_mate = 0; mate_len[m] = len;
        // Shared memory is sized for MAX_LEN; a longer read would need a
        // different arrangement, so it is reported rather than truncated.
        if (len > MAX_LEN && out_overflow) atomicAdd(out_overflow, 1u);
      }
      if (len > MAX_LEN) len = MAX_LEN;
      __syncthreads();
      uint64_t tbase = sa.tok_base + (uint64_t) r * sa.tok_stride + (uint64_t) m * sa.mate_stride;

      for (int i = threadIdx.x; i < len; i += blockDim.x) {
        int v = k2c::base_code(seq[i]);
        code[i] = (uint8_t) (v < 0 ? 4 : v);
        if (v < 0) amb_mate = 1;
      }
      __syncthreads();

      if (amb_mate) {
        // Sequential: as ScanFragmentTokens and ResolveFragmentTokens do,
        // token by token.
        if (threadIdx.x == 0) {
          k2c::Scanner sc;
          sc.init(k, l, spaced_seed_mask, toggle_mask);
          sc.load(seq, len);
          uint64_t mz, last_min = ~0ULL; bool amb;
          uint32_t last_taxon = 0;
          int t = 0;
          while (sc.next(&mz, &amb)) {
            uint32_t taxon;
            if (amb) taxon = k2c::AMBIG;
            else if (mz != last_min) {
              last_min = mz;
              if (min_acceptable && murmur(mz) < min_acceptable) taxon = 0;
              else taxon = lookup_token(table, sa, skip_probe, mz, tbase + t);
              last_taxon = taxon;
            } else taxon = last_taxon;
            if (tokens_wanted) stax[n_tok + t] = taxon;
            t++;
          }
          n_tok += t;
        }
      } else {
        int n_lmer = len - l + 1; if (n_lmer < 1) n_lmer = 0;
        const uint64_t lmask = (l * 2 >= 64) ? ~0ULL : ((1ULL << (l * 2)) - 1);
        for (int i = threadIdx.x; i < n_lmer; i += blockDim.x) {
          uint64_t lmer = 0;
          for (int j = 0; j < l; j++) lmer = (lmer << 2) | code[i + j];
          lmer &= lmask;
          uint64_t rc = k2c::revcomp(lmer, l);
          uint64_t canon = lmer < rc ? lmer : rc;
          if (spaced_seed_mask) canon &= spaced_seed_mask;
          cand[i] = canon ^ toggle_mask;
        }
        __syncthreads();

        // Every k-mer is a token, as MinimizerScanner::NextMinimizer returns
        // once per k-mer when the read has no ambiguous base.
        int n_kmer = len - k + 1; if (n_kmer < 1) n_kmer = 0;
        int win = k - l + 1;
        for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) {
          uint64_t mn = ~0ULL;
          for (int j = 0; j < win; j++) { uint64_t c = cand[p + j]; if (c < mn) mn = c; }
          val[p] = mn ^ toggle_mask;
        }
        __syncthreads();

        // One lookup per run of equal consecutive minimizers, at the run's
        // first k-mer, unless the subsampling filter drops it: the CPU looks
        // a minimizer up only when it differs from the previous one.
        for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) {
          int keep = (p == 0 || val[p - 1] != val[p]);
          if (keep && min_acceptable && murmur(val[p]) < min_acceptable) keep = 0;
          look[p] = keep;
          if (keep) ptax[p] = lookup_token(table, sa, skip_probe, val[p], tbase + p);
        }
        __syncthreads();

        // A repeat inherits the taxon found at the start of its run, and a
        // run whose start the filter dropped contributes zero, which is what
        // last_taxon does sequentially on the CPU.
        if (tokens_wanted) {
          for (int p = threadIdx.x; p < n_kmer; p += blockDim.x) {
            int s = p;
            while (s > 0 && val[s - 1] == val[p]) s--;
            stax[n_tok + p] = look[s] ? ptax[s] : 0u;
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) n_tok += n_kmer;
      }
      __syncthreads();
      if (threadIdx.x == 0 && mates == 2 && m == 0) {
        if (tokens_wanted) stax[n_tok] = k2c::MATE;
        n_tok++;
      }
      __syncthreads();
    }
    if (!tokens_wanted) continue;

    // Counting distinct taxa and resolving the tree is a few dozen operations
    // on a handful of values, so one thread does it rather than paying for a
    // reduction across the block.
    if (threadIdx.x == 0) {
      uint32_t call = k2c::call_of(parent, stax, n_tok);
      out_call[r] = (uint32_t) (call ? extid[call] : 0);
      if (out_text) {
        // A hit list has one run per change of taxon, so a long read can
        // produce a long line.  Rather than size the slot for the worst case,
        // the formatter stops short of the end and reports it, so a
        // too-narrow slot shows up as a counted failure instead of a
        // neighbouring line quietly overwritten.
        char *w0 = out_text + (size_t) r * text_stride;
        uint32_t rec0 = r * mates;
        bool over = false;
        int n = k2c::format_line(w0, w0 + text_stride - 64, call, extid,
                                 bases + hdr_off[rec0], hdr_len[rec0], mates == 2,
                                 (uint32_t) mate_len[0], mates == 2 ? (uint32_t) mate_len[1] : 0u,
                                 stax, n_tok, &over);
        if (over) atomicAdd(out_overflow, 1u);
        out_linelen[r] = (uint32_t) n;
      }
    }
    __syncthreads();
  }
}

// Looks up the binned minimizers of partitions [part_lo, part_hi), one thread
// per minimizer, and stores each result next to its minimizer.  Bins are
// visited in partition order, so the threads running at any moment work
// within a few partitions, which can sit in L2, and both the reads and the
// stores are sequential.
__global__ void probe_bins(const k2t::Table table,
                           const uint64_t *__restrict__ bin_h,
                           uint16_t *__restrict__ bin_v,
                           const uint32_t *__restrict__ bin_cnt, uint32_t bin_cap,
                           uint64_t bin_base, uint32_t part_lo, uint32_t part_hi,
                           int no_write) {
  uint64_t n = (uint64_t) (part_hi - part_lo) * bin_cap;
  for (uint64_t j = blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; j < n;
       j += (uint64_t) gridDim.x * blockDim.x) {
    uint32_t b = part_lo + (uint32_t) (j / bin_cap), e = (uint32_t) (j % bin_cap);
    if (e >= bin_cnt[b]) continue;
    uint64_t at = bin_base + (uint64_t) b * bin_cap + e;
    uint32_t v = table.get_blocked_hash(bin_h[at]);
    // A taxon never equals 0xFFFF0000, so with no_write nothing is stored,
    // but the lookup cannot be optimized away.
    if (!no_write || v == 0xFFFF0000u) bin_v[at] = (uint16_t) v;
  }
}

// Longest read of a chunk, which sets the per-mate stride of the token state.
__global__ void max_len(const uint32_t *__restrict__ len, uint32_t n,
                        uint32_t *__restrict__ out) {
  uint32_t m = 0;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    m = len[i] > m ? len[i] : m;
  atomicMax(out, m);
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
