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
// Subsampling keeps only partitions 0 .. K-1, a fraction K / P of hash space,
// and stores each one exactly as a full build would: nb comes from the
// full key count, so a kept partition is byte-identical to the same
// partition of the full table.  A lookup whose partition is K or above is a
// miss and reads a shared all-empty bucket instead of the table.
//
// File layout: a BlockedTableHeader padded to 4096 bytes, K * nb buckets,
// then the stash: K + 1 uint64 offsets, the stash keys (home << r | rem,
// uint64) and their taxon codes (uint32), then the code-to-taxon array, U + 1
// uint32 with entry 0 = 0.  K is kept_partitions, or P when that field is 0.
// Version 1 files have no array and store internal taxon IDs directly, and
// hold kept_partitions where version 2 holds taxon_codes.

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
// Automatic partition count.  Partitions of fewer keys than this vary
// enough in size to spill keys into the stash at the default load, and 4,096
// partitions is what the GPU's partition streaming and the build's parallel
// placement were measured with.
const int BLOCKED_AUTO_MAX_BITS = 12;
const uint64_t BLOCKED_AUTO_MIN_KEYS = 1ULL << 17;

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
  uint64_t kept_partitions;     // K; 0 means all P partitions
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
  uint64_t P, K, nb, qmask, rmask, vmask;
  Pext pext;
  Mix mix;
  FastDiv fd_nb;
  FastMod fm_alt, fm_sf;

  void init(uint64_t pext_mask, int partition_bits, uint64_t buckets,
            int remainder_bits, int value_bits, uint64_t kept_partitions = 0);
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
  uint8_t empty_bucket_[64] __attribute__((aligned(64)));
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
// when it is below the exact width.  kept_partitions of 0 keeps every
// partition; K keeps partitions 0 .. K-1 and drops the rest.
class BlockedTableBuilder {
  public:
#pragma pack(push, 1)
  struct Rec { uint64_t h; uint32_t taxid; };
#pragma pack(pop)

  // A negative partition_bits chooses the partition count from the number
  // of distinct keys: the largest power of two up to 2^BLOCKED_AUTO_MAX_BITS
  // that leaves each partition at least BLOCKED_AUTO_MIN_KEYS keys.
  BlockedTableBuilder(uint64_t pext_mask, int partition_bits, double load,
                      const Taxonomy &tax, int remainder_bits = 0,
                      uint64_t kept_partitions = 0);

  // Per-thread buffer; Add() is not thread-safe, Flush() is.  Flush() adds
  // the buffered records without reducing partitions, which Write() does.
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
  // A partition's reduced records, sorted by hash and distinct, packed as
  // the Rice-coded gap between consecutive low q bits of the hash (k bits
  // of remainder) followed by the internal taxon in a fixed width.  Records
  // cost about the information in the key set, not 12 bytes each.
  struct PackedRun {
    std::vector<uint64_t> words;
    uint64_t n = 0;
    int k = 0;
  };

  void Append(size_t part, std::vector<Rec> &recs, bool may_reduce = true);
  void Reduce(std::vector<Rec> &v);
  void ReduceTail(std::vector<Rec> &v, size_t from);
  void FinishPartition(size_t part, int taxonomy_bits);
  void Gather(std::vector<Rec> &dst, std::vector<std::vector<Rec> > &chunks);
  // Merges reduced records into the partition's packed run by LCA.
  void MergeIntoRun(size_t part, const std::vector<Rec> &recs, int taxonomy_bits);
  // Appends collected partition part's records to out, in hash order.
  void DecodeRun(size_t part, std::vector<Rec> &out) const;
  // Appends the records of placement partition part, which spans 2^d
  // collected partitions after an automatic partition merge.
  void DecodePartition(uint64_t part, std::vector<Rec> &out) const;
  void FreePartition(uint64_t part);
  size_t TailLimit(size_t part) const;

  uint64_t pext_mask_;
  int p_, B_;
  bool auto_p_;
  double load_;
  int remainder_bits_;
  uint64_t kept_;
  const Taxonomy &tax_;
  blocked::Pext pext_;
  blocked::Mix mix_;
  // Each collected partition is a packed run of reduced records (runs_)
  // plus a tail of raw records appended since its last merge (parts_).
  // When the tail outgrows a sixteenth of the run, a thread reduces it and
  // merges it into the run outside the lock; buffers for that partition
  // wait on pending_ meanwhile.  Collection uses p0_ partition bits; an
  // automatic partition count may place at fewer (p_ <= p0_).
  int p0_;
  int tax_bits_;
  int tail_shift_;
  size_t buf_recs_;
  std::vector<PackedRun> runs_;
  std::vector<std::vector<Rec> > parts_;
  std::vector<std::vector<std::vector<Rec> > > pending_;
  std::vector<char> busy_;
  std::vector<std::mutex> locks_;
};

}  // end namespace

#endif
