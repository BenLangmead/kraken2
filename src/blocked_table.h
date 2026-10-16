/*
 * Copyright 2013-2023, Derrick Wood <dwood@cs.jhu.edu>
 *
 * This file is part of the Kraken 2 taxonomic sequence classification system.
 */

// Static, hash-partitioned minimizer -> taxon table with 64-byte buckets
// (one cache line) and quotiented remainders.  It stores enough of every key
// to tell keys apart ("exact mode"), so a lookup never returns a false
// positive, and its layout depends only on the set of (minimizer, LCA) pairs,
// not on input order or thread count.
//
// Key path, for a minimizer m from MinimizerScanner:
//   x    = pext(m, pext_mask)          B bits; pext_mask is the spaced seed
//                                      mask restricted to the l-mer
//   h    = mix(x)                      bijection on B bits
//   part = h >> q, low = h & (2^q - 1) q = B - p, P = 2^p partitions
//   home = floor(low * nb / 2^q)       nb buckets per partition
//   rem  = low - ceil(home * 2^q / nb) remainder within home's range; r bits
//                                      are enough to store it whole
//   alt  = (home + 1 + g(rem) mod (nb - 1)) mod nb
// home can be recovered from (alt, rem), so a key stored in its alt bucket
// is still unambiguous.
//
// Fingerprint mode stores only the low r bits of the remainder, for an r
// below the exact width, and rem above means those r bits.  Keys with the
// same (home, rem) become one entry whose taxon is their LCA, so a lookup
// matches at most one slot, and that slot's taxon is the LCA of every key
// that maps there.  A minimizer that is not in the table but shares
// (home, rem) with an entry returns that entry's taxon: a false positive, at
// about (slots compared) * 2^-r per absent lookup.
//
// Bucket (512 bits, little-endian bit order):
//   bits 0-6    natives: keys whose home is this bucket, stored first
//   bits 7-13   total: natives plus guests (keys whose alt is this bucket)
//   bits 14..   spill filter, sf bits: a key stored outside its home bucket
//               sets bit hash(rem) mod sf in its home bucket's filter
//   then S slots of w = r + v bits: remainder in the low r bits, taxon
//   code above
// A lookup scans the natives of home and stops on a match.  Otherwise it
// reads alt only if its own filter bit is set, and scans alt's guests.  Keys
// that fit in neither bucket go to a sorted per-partition stash, which a
// lookup searches only when alt is full.  Augmenting placement at build time
// normally leaves the stash empty.
//
// Taxon codes: slots and the stash store a dense code, not the internal
// taxon ID.  Codes 1..U number the U distinct taxa stored in the table, most
// frequent first (by entry count, ties by internal taxon ID), so v is the
// width of U rather than of the whole taxonomy.  Code 0 means "absent".  A
// lookup decodes the code it finds through the code-to-taxon array.
//
// File layout: a BlockedTableHeader padded to 4096 bytes, P * nb buckets,
// then the stash: P + 1 uint64 offsets, the stash keys (home << r | rem,
// uint64) and their taxon codes (uint32), then the code-to-taxon array, U + 1
// uint32 with entry 0 = 0.  Version 1 files have no array and store internal
// taxon IDs directly.

#ifndef KRAKEN2_BLOCKED_TABLE_H_
#define KRAKEN2_BLOCKED_TABLE_H_

#include "kv_store.h"
#include "mmap_file.h"
#include "kraken2_headers.h"
#include "taxonomy.h"

#include <mutex>
#include <string>
#include <vector>

namespace kraken2 {

// "K2BLKT01" read as a little-endian uint64.
const uint64_t BLOCKED_TABLE_MAGIC = 0x3130544B4C42324BULL;
const size_t BLOCKED_TABLE_HEADER_BYTES = 4096;
const int BLOCKED_BUCKET_BITS = 512;
const int BLOCKED_COUNT_BITS = 14;
// The 8-byte slot read in the lookup needs w + 7 <= 64.
const int BLOCKED_MIN_SLOT_BITS = 16;
const int BLOCKED_MAX_SLOT_BITS = 57;

struct BlockedTableHeader {
  uint64_t magic;
  uint32_t version;
  uint32_t bucket_bytes;
  uint32_t partition_bits;      // p
  uint32_t key_bits;            // B
  uint32_t remainder_bits;      // r; below the exact width in fingerprint mode
  uint32_t value_bits;          // v, the width of a taxon code
  uint32_t slots;               // S
  uint32_t spill_bits;          // sf
  uint64_t buckets_per_partition;  // nb
  uint64_t entries;             // stored (home, rem) entries
  uint64_t stash_entries;
  uint64_t pext_mask;
  uint64_t guests;
  uint64_t spilled_buckets;
  uint64_t taxon_codes;         // U; version 2 and later
};

namespace blocked {

inline uint64_t murmur(uint64_t k) {
  k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33;
  return k;
}

inline uint64_t mulhi64(uint64_t a, uint64_t b) {
  return (uint64_t) (((unsigned __int128) a * b) >> 64);
}

// a mod d for 64-bit a, by Lemire's method with a 128-bit magic number.
struct FastMod {
  uint64_t mlo, mhi, d;
  void init(uint64_t div) {
    unsigned __int128 M = (~(unsigned __int128) 0) / div + 1;
    mlo = (uint64_t) M; mhi = (uint64_t) (M >> 64); d = div;
  }
  inline uint64_t mod(uint64_t a) const {
    uint64_t lo = mlo * a;
    uint64_t hi = mhi * a + mulhi64(mlo, a);
    uint64_t bottom = mulhi64(lo, d);
    uint64_t tlo = hi * d, thi = mulhi64(hi, d);
    uint64_t s = tlo + bottom;
    return thi + (s < tlo);
  }
};

// floor(n / d) for n < 2^N, as a multiply by m = ceil(2^k / d) with
// k = N + ceil(log2 d).
struct FastDiv {
  uint64_t m; int k;
  void init(uint64_t div, int nbits) {
    int lg = 0;
    while (lg < 64 && (1ULL << lg) < div) lg++;
    k = nbits + lg;
    unsigned __int128 one = 1;
    m = (uint64_t) (((one << k) + div - 1) / div);
  }
  inline uint64_t div(uint64_t n) const {
    uint64_t lo = n * m, hi = mulhi64(n, m);
    return k >= 64 ? hi >> (k - 64) : (hi << (64 - k)) | (lo >> k);
  }
};

// pext for a fixed mask, one run of set bits at a time.
struct Pext {
  int nruns;
  uint8_t src[32], dst[32];
  uint64_t mask[32];
  int width;
  void init(uint64_t m) {
    nruns = 0; width = 0;
    for (int b = 0; b < 64; ) {
      if (! ((m >> b) & 1)) { b++; continue; }
      int e = b;
      while (e < 64 && ((m >> e) & 1)) e++;
      src[nruns] = b; dst[nruns] = width;
      mask[nruns] = (e - b == 64) ? ~0ULL : ((1ULL << (e - b)) - 1);
      nruns++;
      width += e - b;
      b = e;
    }
  }
  inline uint64_t operator()(uint64_t m) const {
    uint64_t x = 0;
    for (int i = 0; i < nruns; i++)
      x |= ((m >> src[i]) & mask[i]) << dst[i];
    return x;
  }
};

// Bijection on B bits: xorshifts and odd multiplies mod 2^B.
struct Mix {
  int B; uint64_t M;
  void init(int b) { B = b; M = b == 64 ? ~0ULL : (1ULL << b) - 1; }
  inline uint64_t operator()(uint64_t x) const {
    x &= M;
    x ^= x >> (B / 2 - 1); x = (x * 0xff51afd7ed558ccdULL) & M;
    x ^= x >> (B / 2 + 1); x = (x * 0xc4ceb9fe1a85ec53ULL) & M;
    x ^= x >> (B / 2);
    return x;
  }
};

const uint64_t ALT_SALT = 0x5bd1e9955bd1e995ULL;
const uint64_t SPILL_SALT = 0x2545f4914f6cdd1dULL;

// Index math shared by the build and the lookup.
struct Geometry {
  int p, q, B, r, v, w, S, sf, H;
  uint64_t P, nb, qmask, rmask, vmask;
  Pext pext;
  Mix mix;
  FastDiv fd_nb;
  FastMod fm_alt, fm_sf;

  void init(uint64_t pext_mask, int partition_bits, uint64_t buckets,
            int remainder_bits, int value_bits);
  inline uint64_t hash(uint64_t minimizer) const { return mix(pext(minimizer)); }
  // home and rem of the low q bits of a hash.
  inline void locate(uint64_t h, uint64_t &part, uint64_t &home, uint64_t &rem) const {
    part = h >> q;
    unsigned __int128 t = (unsigned __int128) (h & qmask) * nb;
    home = (uint64_t) (t >> q);
    rem = fd_nb.div((uint64_t) t & qmask) & rmask;
  }
  inline uint64_t alt(uint64_t home, uint64_t rem) const {
    uint64_t a = home + 1 + fm_alt.mod(murmur(rem ^ ALT_SALT));
    return a >= nb ? a - nb : a;
  }
  inline unsigned spill_bit(uint64_t rem) const {
    return sf == 1 ? 0 : (unsigned) fm_sf.mod(murmur(rem ^ SPILL_SALT));
  }
};

}  // namespace blocked

class BlockedTable : public KeyValueStore {
  public:
  BlockedTable(const std::string &filename, bool memory_mapping = false);
  ~BlockedTable();

  hvalue_t Get(hkey_t key) const;
  void GetBatch(const hkey_t *keys, hvalue_t *out, size_t n) const;

  // Lookup by hash rather than by minimizer (used by the build's self-check).
  hvalue_t GetByHash(uint64_t h) const;

  const BlockedTableHeader &header() const { return header_; }
  const blocked::Geometry &geometry() const { return geo_; }

  private:
  friend class BlockedTableBuilder;
  BlockedTable();
  BlockedTable(const BlockedTable &);
  BlockedTable &operator=(const BlockedTable &);

  void LoadTable(const char *filename, bool memory_mapping);
  void Init();

  BlockedTableHeader header_;
  blocked::Geometry geo_;
  const uint8_t *buckets_;
  const uint64_t *stash_offsets_;
  const uint64_t *stash_keys_;
  const uint32_t *stash_vals_;
  const uint32_t *code_taxa_;  // code -> internal taxon ID; nullptr in version 1
  char *owned_;            // malloc'ed table bytes, or nullptr when mapped
  size_t owned_bytes_;
  MMapFile backing_file_;
  bool file_backed_;

  typedef hvalue_t (*GetFn)(const BlockedTable &, uint64_t);
  typedef void (*BatchFn)(const BlockedTable &, const hkey_t *, hvalue_t *, size_t, int);
  GetFn get_fn_;
  BatchFn batch_fn_;

  template<int W> friend struct BlockedProbe;
};

// Collects (minimizer, taxon) pairs from many threads, reduces duplicates by
// LCA, and writes the table.  Pairs are grouped by partition as they arrive,
// so each partition is placed independently.  remainder_bits of 0 builds an
// exact table; a positive value is used as r, which gives a fingerprint table
// when it is below the exact width.
class BlockedTableBuilder {
  public:
#pragma pack(push, 1)
  struct Rec { uint64_t h; uint32_t taxid; };
#pragma pack(pop)

  BlockedTableBuilder(uint64_t pext_mask, int partition_bits, double load,
                      const Taxonomy &tax, int remainder_bits = 0);

  // Per-thread buffer; Add() is not thread-safe, Flush() is.
  class Buffer {
    public:
    explicit Buffer(BlockedTableBuilder &b);
    ~Buffer();
    void Add(uint64_t minimizer, uint32_t taxid);
    void Flush();
    private:
    BlockedTableBuilder &builder_;
    std::vector<std::vector<Rec> > parts_;
  };

  // Numbers the stored taxa, places every partition and writes the file.
  // taxonomy_bits is the width of every internal taxon ID; it is checked and
  // reported, but slots use the narrower code width.  Returns the number of
  // distinct minimizers.
  uint64_t Write(const char *filename, int taxonomy_bits, int threads);

  uint64_t pext_mask() const { return pext_mask_; }

  private:
  void Append(size_t part, std::vector<Rec> &recs);
  void Reduce(std::vector<Rec> &v);

  uint64_t pext_mask_;
  int p_, B_;
  double load_;
  int remainder_bits_;
  const Taxonomy &tax_;
  blocked::Pext pext_;
  blocked::Mix mix_;
  std::vector<std::vector<Rec> > parts_;
  std::vector<size_t> reduced_size_;
  std::vector<std::mutex> locks_;
};

}  // end namespace

#endif
