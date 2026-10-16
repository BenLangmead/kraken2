// The minimizer table as the device kernels and the host emulation see it:
// either Kraken 2's compact hash table (CHT) or the blocked table of
// src/blocked_table.h, chosen by the file's magic number as classify does.
//
// Everything here compiles both as CUDA and as plain C++, so the kernel's
// lookup and the host emulation's lookup are the same code.  The blocked
// index math is the division-free version in src/blocked_table.h, restated
// without unsigned __int128 on the device side.

#ifndef KRAKEN2_GPU_TABLE_H_
#define KRAKEN2_GPU_TABLE_H_

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define K2HD __host__ __device__ __forceinline__
#else
#define K2HD inline
#endif

namespace k2t {

K2HD uint64_t murmur(uint64_t k) {
  k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33; return k;
}

K2HD uint64_t mulhi64(uint64_t a, uint64_t b) {
#ifdef __CUDA_ARCH__
  return __umul64hi(a, b);
#else
  return (uint64_t) (((unsigned __int128) a * b) >> 64);
#endif
}

// a mod d by Lemire's method with a 128-bit magic number.
struct FastMod {
  uint64_t mlo, mhi, d;
  void init(uint64_t div) {
#ifndef __CUDA_ARCH__
    unsigned __int128 M = (~(unsigned __int128) 0) / div + 1;
    mlo = (uint64_t) M; mhi = (uint64_t) (M >> 64); d = div;
#endif
  }
  K2HD uint64_t mod(uint64_t a) const {
    uint64_t lo = mlo * a;
    uint64_t hi = mhi * a + mulhi64(mlo, a);
    uint64_t bottom = mulhi64(lo, d);
    uint64_t tlo = hi * d, thi = mulhi64(hi, d);
    uint64_t s = tlo + bottom;
    return thi + (s < tlo);
  }
};

// floor(n / d) for n < 2^nbits, as a multiply by ceil(2^k / d).
struct FastDiv {
  uint64_t m; int k;
  void init(uint64_t div, int nbits) {
#ifndef __CUDA_ARCH__
    int lg = 0;
    while (lg < 64 && (1ULL << lg) < div) lg++;
    k = nbits + lg;
    unsigned __int128 one = 1;
    m = (uint64_t) (((one << k) + div - 1) / div);
#endif
  }
  K2HD uint64_t div(uint64_t n) const {
    uint64_t lo = n * m, hi = mulhi64(n, m);
    return k >= 64 ? hi >> (k - 64) : (hi << (64 - k)) | (lo >> k);
  }
};

// pext for a fixed mask, one run of set bits at a time.
struct Pext {
  int nruns, width;
  uint8_t src[32], dst[32];
  uint64_t mask[32];
  void init(uint64_t m) {
    nruns = 0; width = 0;
    for (int b = 0; b < 64; ) {
      if (! ((m >> b) & 1)) { b++; continue; }
      int e = b;
      while (e < 64 && ((m >> e) & 1)) e++;
      src[nruns] = (uint8_t) b; dst[nruns] = (uint8_t) width;
      mask[nruns] = (e - b == 64) ? ~0ULL : ((1ULL << (e - b)) - 1);
      nruns++; width += e - b; b = e;
    }
  }
  K2HD uint64_t operator()(uint64_t x) const {
    uint64_t y = 0;
    for (int i = 0; i < nruns; i++) y |= ((x >> src[i]) & mask[i]) << dst[i];
    return y;
  }
};

// Bijection on B bits, as in BlockedTable.
struct Mix {
  int B; uint64_t M;
  void init(int b) { B = b; M = b == 64 ? ~0ULL : (1ULL << b) - 1; }
  K2HD uint64_t operator()(uint64_t x) const {
    x &= M;
    x ^= x >> (B / 2 - 1); x = (x * 0xff51afd7ed558ccdULL) & M;
    x ^= x >> (B / 2 + 1); x = (x * 0xc4ceb9fe1a85ec53ULL) & M;
    x ^= x >> (B / 2);
    return x;
  }
};

static const uint64_t ALT_SALT = 0x5bd1e9955bd1e995ULL;
static const uint64_t SPILL_SALT = 0x2545f4914f6cdd1dULL;
static const int COUNT_BITS = 14;
static const int BUCKET_BITS = 512;

// The fields of BlockedTableHeader that a lookup needs; the file starts with
// this layout and is padded to 4,096 bytes.
struct BlockedHeader {
  uint64_t magic;
  uint32_t version, bucket_bytes, partition_bits, key_bits, remainder_bits,
           value_bits, slots, spill_bits;
  uint64_t buckets_per_partition, entries, stash_entries, pext_mask,
           guests, spilled_buckets;
};
static const uint64_t BLOCKED_MAGIC = 0x3130544B4C42324BULL;   // "K2BLKT01"
static const size_t BLOCKED_HEADER_BYTES = 4096;

struct Geometry {
  int p, q, B, r, v, w, S, sf, H;
  uint64_t P, nb, qmask, rmask, vmask;
  Pext pext; Mix mix; FastDiv fd_nb; FastMod fm_alt, fm_sf;

  void init(const BlockedHeader &h) {
    pext.init(h.pext_mask);
    B = pext.width; mix.init(B);
    p = (int) h.partition_bits; q = B - p; P = 1ULL << p;
    nb = h.buckets_per_partition;
    r = (int) h.remainder_bits; v = (int) h.value_bits; w = r + v;
    S = (BUCKET_BITS - COUNT_BITS - 1) / w;
    sf = BUCKET_BITS - COUNT_BITS - S * w; if (sf > 64) sf = 64;
    H = COUNT_BITS + sf;
    qmask = (1ULL << q) - 1; rmask = (1ULL << r) - 1; vmask = (1ULL << v) - 1;
    fd_nb.init(nb, q); fm_alt.init(nb - 1);
    if (sf > 1) fm_sf.init((uint64_t) sf);
  }
  K2HD uint64_t hash(uint64_t minimizer) const { return mix(pext(minimizer)); }
  K2HD uint64_t part_of(uint64_t h) const { return h >> q; }
  // Home bucket and remainder from the low q bits of a hash.
  K2HD void locate(uint64_t h, uint64_t &home, uint64_t &rem) const {
    uint64_t low = h & qmask;
    // low < 2^q and nb < 2^(64 - q) for every geometry the builder makes, so
    // the product fits 64 bits; mulhi keeps it exact regardless.
    uint64_t tlo = low * nb, thi = mulhi64(low, nb);
    home = (q == 64) ? thi : ((thi << (64 - q)) | (tlo >> q));
    rem = fd_nb.div(tlo & qmask);
  }
  K2HD uint64_t alt(uint64_t home, uint64_t rem) const {
    uint64_t a = home + 1 + fm_alt.mod(murmur(rem ^ ALT_SALT));
    return a >= nb ? a - nb : a;
  }
  K2HD unsigned spill_bit(uint64_t rem) const {
    return sf == 1 ? 0 : (unsigned) fm_sf.mod(murmur(rem ^ SPILL_SALT));
  }
};

// Bytes p[0..7] of a bucket.  The device reads aligned 32-bit words through
// the read-only cache; the host uses memcpy.
K2HD uint64_t ld64(const uint8_t *p) {
#ifdef __CUDA_ARCH__
  uintptr_t a = (uintptr_t) p;
  const uint32_t *w = (const uint32_t *) (a & ~(uintptr_t) 3);
  unsigned s = (unsigned) (a & 3) * 8;
  uint64_t lo = (uint64_t) __ldg(w) | ((uint64_t) __ldg(w + 1) << 32);
  if (! s) return lo;
  // Callers clamp the byte offset to 56, so w + 2 stays inside the bucket.
  return (lo >> s) | ((uint64_t) __ldg(w + 2) << (64 - s));
#else
  uint64_t v; memcpy(&v, p, 8); return v;
#endif
}

struct Table {
  enum Kind { CHT = 0, BLOCKED = 1 };
  int kind;

  // Compact hash table: 4-byte cells, key in the high bits, value in the low.
  const uint32_t *cells;
  uint64_t capacity;
  int key_bits, value_bits;

  // Blocked table.  buckets points at partition 0 of whatever is resident;
  // part_base is the first partition held there, so a run that holds only a
  // group of partitions addresses them from the start of its slot.
  const uint8_t *buckets;
  uint64_t part_base;
  const uint64_t *stash_off, *stash_key;
  const uint32_t *stash_val;
  Geometry G;
  // How the device scans a bucket: REGS_FULL and REGS_EARLY read it once into
  // registers (see get_blocked_regs) when its geometry has an instantiation,
  // and RELOAD reads each slot with ld64.  The host always uses scan().
  enum ScanMode { REGS_FULL = 0, REGS_EARLY = 1, RELOAD = 2 };
  int scan_mode;

  // Taxon in the slot of [lo, hi) whose remainder is rem, or 0.  Each slot is
  // read with an 8-byte load that stays inside its 64-byte bucket.
  K2HD uint32_t scan(const uint8_t *b, unsigned lo, unsigned hi, uint64_t rem) const {
    uint32_t val = 0;
    for (unsigned i = lo; i < hi; i++) {
      unsigned off = (unsigned) G.H + i * (unsigned) G.w;
      unsigned byte = (off >> 3) < 56 ? (off >> 3) : 56;
      uint64_t s = ld64(b + byte) >> (off - byte * 8);
      if ((s & G.rmask) == rem) val = (uint32_t) ((s >> G.r) & G.vmask);
    }
    return val;
  }

  K2HD uint32_t stash_find(uint64_t part, uint64_t home, uint64_t rem) const {
    uint64_t lo = stash_off[part], hi = stash_off[part + 1];
    uint64_t key = (home << G.r) | rem;
    while (lo < hi) {
      uint64_t mid = lo + (hi - lo) / 2;
      if (stash_key[mid] < key) lo = mid + 1; else hi = mid;
    }
    return (lo < stash_off[part + 1] && stash_key[lo] == key) ? stash_val[lo] : 0u;
  }

  // The blocked lookup for a hash: the natives of home, stopping on a match;
  // then, only if home's spill-filter bit for rem is set, the guests of alt;
  // then the stash only if alt is full.
  K2HD uint32_t get_blocked_hash(uint64_t h) const {
#ifdef __CUDA_ARCH__
    // Every geometry the builder makes from the standard value widths; the
    // test is uniform across a launch, so it costs no divergence.
#define K2T_GEO(gh, gw, gs) \
    if (G.H == gh && G.w == gw && G.S == gs && scan_mode != RELOAD) \
      return scan_mode == REGS_EARLY ? get_blocked_regs<gh, gw, gs, true>(h) \
                                     : get_blocked_regs<gh, gw, gs, false>(h);
    K2T_GEO(36, 34, 14) K2T_GEO(29, 23, 21) K2T_GEO(26, 27, 18)
#undef K2T_GEO
#endif
    uint64_t part = G.part_of(h), home, rem;
    G.locate(h, home, rem);
    const uint8_t *base = buckets + (((part - part_base) * G.nb) << 6);
    const uint8_t *hb = base + (home << 6);
    uint64_t h64 = ld64(hb);
    uint32_t v = scan(hb, 0, (unsigned) (h64 & 127), rem);
    if (v) return v;
    unsigned bit = COUNT_BITS + G.spill_bit(rem);
    if (! ((h64 >> bit) & 1)) return 0;
    const uint8_t *ab = base + (G.alt(home, rem) << 6);
    uint64_t a64 = ld64(ab);
    unsigned lo = (unsigned) (a64 & 127), hi = (unsigned) ((a64 >> 7) & 127);
    v = scan(ab, lo, hi, rem);
    if (! v && hi == (unsigned) G.S && stash_off) v = stash_find(part, home, rem);
    return v;
  }

#ifdef __CUDACC__
  // Bits [H + i * W, H + (i + 1) * W) of a bucket held in 16 registers.  The
  // word indices and shifts are constants once the loop over slots unrolls.
  template<int H, int W>
  __device__ __forceinline__ static uint64_t reg_slot(const uint32_t (&w)[16], int i) {
    const int off = H + i * W, j = off >> 5, s = off & 31;
    uint64_t x = (uint64_t) w[j] | ((uint64_t) w[j + 1 < 16 ? j + 1 : 15] << 32);
    x >>= s;
    if (s && s + W > 64 && j + 2 < 16) x |= (uint64_t) w[j + 2] << (64 - s);
    return x;
  }
  __device__ __forceinline__ static void load_bucket(const uint8_t *b, uint32_t (&w)[16]) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      uint4 u = __ldg((const uint4 *) b + k);
      w[4 * k] = u.x; w[4 * k + 1] = u.y; w[4 * k + 2] = u.z; w[4 * k + 3] = u.w;
    }
  }
  // Taxon in the slot of [lo, hi) whose remainder is rem, or 0.  Without
  // EARLY every slot is compared without branches; with EARLY the scan stops
  // at the first remainder >= rem, since each run of slots is sorted.
  template<int H, int W, int S, bool EARLY>
  __device__ __forceinline__ uint32_t scan_regs(const uint32_t (&w)[16], unsigned lo, unsigned hi,
                                                uint64_t rem) const {
    uint32_t val = 0;
    #pragma unroll
    for (int i = 0; i < S; i++) {
      uint64_t x = reg_slot<H, W>(w, i), xr = x & G.rmask;
      if (EARLY) {
        if ((unsigned) i >= hi) break;
        if ((unsigned) i < lo) continue;
        if (xr >= rem) { if (xr == rem) val = (uint32_t) ((x >> G.r) & G.vmask); break; }
      } else {
        bool m = (unsigned) i >= lo && (unsigned) i < hi && xr == rem;
        val = m ? (uint32_t) ((x >> G.r) & G.vmask) : val;
      }
    }
    return val;
  }
  // get_blocked_hash for a geometry fixed at compile time, with each bucket
  // read once as four 16-byte loads instead of 2 to 3 words per slot.
  template<int H, int W, int S, bool EARLY>
  __device__ __forceinline__ uint32_t get_blocked_regs(uint64_t h) const {
    uint64_t part = G.part_of(h), home, rem;
    G.locate(h, home, rem);
    const uint8_t *base = buckets + (((part - part_base) * G.nb) << 6);
    uint32_t w[16];
    load_bucket(base + (home << 6), w);
    uint32_t v = scan_regs<H, W, S, EARLY>(w, 0, w[0] & 127, rem);
    if (v) return v;
    unsigned bit = COUNT_BITS + G.spill_bit(rem);
    unsigned sp = bit < 32 ? (w[0] >> bit) & 1 : (w[1] >> (bit - 32)) & 1;
    if (! sp) return 0;
    load_bucket(base + (G.alt(home, rem) << 6), w);
    unsigned lo = w[0] & 127, hi = (w[0] >> 7) & 127;
    v = scan_regs<H, W, S, EARLY>(w, lo, hi, rem);
    if (! v && hi == (unsigned) S && stash_off) v = stash_find(part, home, rem);
    return v;
  }
#endif

  K2HD uint32_t get_cht(uint64_t key) const {
    uint64_t hc = murmur(key);
    uint32_t compacted = (uint32_t) (hc >> (64 - key_bits));
    uint64_t idx = hc % capacity, first = idx;
    const uint32_t vmask = (1u << value_bits) - 1;
    while (true) {
      uint32_t cell = cells[idx];
      uint32_t val = cell & vmask;
      if (! val) return 0;
      if ((cell >> value_bits) == compacted) return val;
      if (++idx == capacity) idx = 0;
      if (idx == first) return 0;
    }
  }

  K2HD uint32_t get(uint64_t key) const {
    return kind == BLOCKED ? get_blocked_hash(G.hash(key)) : get_cht(key);
  }
};

}  // namespace k2t

#endif  // KRAKEN2_GPU_TABLE_H_
