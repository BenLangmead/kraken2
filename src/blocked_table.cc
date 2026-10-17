/*
 * Copyright 2013-2023, Derrick Wood <dwood@cs.jhu.edu>
 *
 * This file is part of the Kraken 2 taxonomic sequence classification system.
 */

#include "blocked_table.h"
#include "compact_hash.h"

#include <numeric>

using std::string;
using std::vector;
using std::pair;

namespace kraken2 {

namespace blocked {

void Geometry::init(uint64_t pext_mask, int partition_bits, uint64_t buckets,
                    int remainder_bits, int value_bits, uint64_t kept_partitions) {
  pext.init(pext_mask);
  B = pext.width;
  mix.init(B);
  p = partition_bits;
  q = B - p;
  P = 1ULL << p;
  K = kept_partitions ? kept_partitions : P;
  nb = buckets;
  r = remainder_bits;
  v = value_bits;
  w = r + v;
  S = (BLOCKED_BUCKET_BITS - BLOCKED_COUNT_BITS - 1) / w;
  sf = std::min(64, BLOCKED_BUCKET_BITS - BLOCKED_COUNT_BITS - S * w);
  H = BLOCKED_COUNT_BITS + sf;
  qmask = (1ULL << q) - 1;
  rmask = (1ULL << r) - 1;
  vmask = (1ULL << v) - 1;
  fd_nb.init(nb, q);
  fm_alt.init(nb - 1);
  if (sf > 1)
    fm_sf.init(sf);
}

}  // namespace blocked

// Lookup code specialized on the slot width W, so the slot offsets, the slot
// count and the header size are compile-time constants.
template<int W>
struct BlockedProbe {
  static const int S = (BLOCKED_BUCKET_BITS - BLOCKED_COUNT_BITS - 1) / W;
  static const int SPARE = BLOCKED_BUCKET_BITS - BLOCKED_COUNT_BITS - S * W;
  static const int SF = SPARE < 64 ? SPARE : 64;
  static const int H = BLOCKED_COUNT_BITS + SF;

  static inline uint64_t load64(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
  }

  // Taxon in the slot of [lo, hi) whose remainder is rem, or 0.  Each slot is
  // read with an 8-byte load that stays inside the 64-byte bucket, so a probe
  // never touches the next cache line.
  static inline hvalue_t scan(const uint8_t *b, unsigned lo, unsigned hi,
                              uint64_t rem, const blocked::Geometry &G) {
    hvalue_t val = 0;
#if defined(__GNUC__) && ! defined(__clang__)
#pragma GCC unroll 32
#endif
    for (int i = 0; i < S; i++) {
      const int off = H + i * W;
      const int byte = (off >> 3) < 56 ? (off >> 3) : 56;
      uint64_t s = load64(b + byte) >> (off - byte * 8);
      bool m = ((unsigned) i >= lo) & ((unsigned) i < hi) & ((s & G.rmask) == rem);
      val = m ? (hvalue_t) ((s >> G.r) & G.vmask) : val;
    }
    return val;
  }

  static inline bool spilled(const uint8_t *hb, uint64_t rem, const blocked::Geometry &G) {
    unsigned bit = BLOCKED_COUNT_BITS + G.spill_bit(rem);
    return (hb[bit >> 3] >> (bit & 7)) & 1;
  }

  static inline hvalue_t stash_find(const BlockedTable &T, uint64_t part,
                                    uint64_t home, uint64_t rem) {
    uint64_t lo = T.stash_offsets_[part], hi = T.stash_offsets_[part + 1];
    if (lo == hi)
      return 0;
    uint64_t key = (home << T.geo_.r) | rem;
    const uint64_t *it = std::lower_bound(T.stash_keys_ + lo, T.stash_keys_ + hi, key);
    if (it != T.stash_keys_ + hi && *it == key)
      return T.stash_vals_[it - T.stash_keys_];
    return 0;
  }

  // Guests of the alt bucket, then the stash if alt is full.
  static inline hvalue_t alt_part(const BlockedTable &T, const uint8_t *ab,
                                  uint64_t part, uint64_t home, uint64_t rem) {
    uint64_t ah = load64(ab);
    unsigned lo = ah & 127, hi = (ah >> 7) & 127;
    hvalue_t v = scan(ab, lo, hi, rem, T.geo_);
    if (! v && hi == (unsigned) S)
      v = stash_find(T, part, home, rem);
    return v;
  }

  // A dropped partition's keys all read the shared empty bucket, which has
  // no slots and no spill bits, so they miss without touching the table.
  static inline const uint8_t *bucket(const BlockedTable &T, uint64_t part, uint64_t b) {
    return part < T.geo_.K ? T.buckets_ + ((part * T.geo_.nb + b) << 6)
                           : T.empty_bucket_;
  }

  static hvalue_t get(const BlockedTable &T, uint64_t h) {
    const blocked::Geometry &G = T.geo_;
    uint64_t part, home, rem;
    G.locate(h, part, home, rem);
    const uint8_t *hb = bucket(T, part, home);
    hvalue_t v = scan(hb, 0, load64(hb) & 127, rem, G);
    if (v || ! spilled(hb, rem, G))
      return v;
    return alt_part(T, bucket(T, part, G.alt(home, rem)), part, home, rem);
  }

  // Two-stage pipeline: home buckets are prefetched D queries ahead and a
  // query stops at a home match.  A query that needs its alt bucket gets that
  // bucket prefetched and is finished D deferrals later.
  static void batch(const BlockedTable &T, const hkey_t *keys, hvalue_t *out,
                    size_t n, int D) {
    const blocked::Geometry &G = T.geo_;
    struct Q { const uint8_t *hb; uint64_t part, home, rem; };
    struct Def { Q x; const uint8_t *ab; size_t o; };
    Q ring[64];
    Def dq[64];
    size_t dh = 0, dt = 0;
    size_t pre = std::min((size_t) D, n);
    for (size_t i = 0; i < pre; i++) {
      Q &y = ring[i & 63];
      G.locate(G.hash(keys[i]), y.part, y.home, y.rem);
      y.hb = bucket(T, y.part, y.home);
      __builtin_prefetch(y.hb, 0, 0);
    }
    for (size_t i = 0; i < n; i++) {
      if (i + D < n) {
        Q &y = ring[(i + D) & 63];
        G.locate(G.hash(keys[i + D]), y.part, y.home, y.rem);
        y.hb = bucket(T, y.part, y.home);
        __builtin_prefetch(y.hb, 0, 0);
      }
      const Q &x = ring[i & 63];
      hvalue_t v = scan(x.hb, 0, load64(x.hb) & 127, x.rem, G);
      if (v || ! spilled(x.hb, x.rem, G)) {
        out[i] = v;
      }
      else {
        Def &d = dq[dt++ & 63];
        d.x = x;
        d.o = i;
        d.ab = bucket(T, x.part, G.alt(x.home, x.rem));
        __builtin_prefetch(d.ab, 0, 0);
        if (dt - dh > (size_t) D) {
          Def &e = dq[dh++ & 63];
          out[e.o] = alt_part(T, e.ab, e.x.part, e.x.home, e.x.rem);
        }
      }
    }
    while (dh < dt) {
      Def &e = dq[dh++ & 63];
      out[e.o] = alt_part(T, e.ab, e.x.part, e.x.home, e.x.rem);
    }
  }
};

template<int W>
struct BlockedDispatch {
  static void pick(int w, hvalue_t (**get)(const BlockedTable &, uint64_t),
                   void (**batch)(const BlockedTable &, const hkey_t *, hvalue_t *, size_t, int)) {
    if (w == W) {
      *get = &BlockedProbe<W>::get;
      *batch = &BlockedProbe<W>::batch;
    }
    else {
      BlockedDispatch<W + 1>::pick(w, get, batch);
    }
  }
};

template<>
struct BlockedDispatch<BLOCKED_MAX_SLOT_BITS + 1> {
  static void pick(int w, hvalue_t (**)(const BlockedTable &, uint64_t),
                   void (**)(const BlockedTable &, const hkey_t *, hvalue_t *, size_t, int)) {
    errx(EX_DATAERR, "blocked table slot width %d is not supported", w);
  }
};

static int blocked_prefetch_dist() {
  const char *e = getenv("K2_PREFETCH_DIST");
  int d = e ? atoi(e) : 8;
  if (d < 1) d = 1;
  if (d > 63) d = 63;
  return d;
}

BlockedTable::BlockedTable()
    : buckets_(nullptr), stash_offsets_(nullptr), stash_keys_(nullptr),
      stash_vals_(nullptr), code_taxa_(nullptr), owned_(nullptr), owned_bytes_(0),
      file_backed_(false), get_fn_(nullptr), batch_fn_(nullptr) {
  memset(&header_, 0, sizeof(header_));
  memset(empty_bucket_, 0, sizeof(empty_bucket_));
}

BlockedTable::BlockedTable(const string &filename, bool memory_mapping)
    : BlockedTable() {
  LoadTable(filename.c_str(), memory_mapping);
}

BlockedTable::~BlockedTable() {
  if (owned_)
    free(owned_);
}

// Sizes of the four parts of the file after the header.
static void blocked_sizes(const BlockedTableHeader &h, size_t &bucket_bytes,
                          size_t &offset_bytes, size_t &stash_bytes,
                          size_t &code_bytes) {
  uint64_t K = h.kept_partitions ? h.kept_partitions : 1ULL << h.partition_bits;
  bucket_bytes = K * h.buckets_per_partition * 64;
  offset_bytes = (K + 1) * sizeof(uint64_t);
  stash_bytes = h.stash_entries * (sizeof(uint64_t) + sizeof(uint32_t));
  code_bytes = h.version >= 2 ? (h.taxon_codes + 1) * sizeof(uint32_t) : 0;
}

void BlockedTable::LoadTable(const char *filename, bool memory_mapping) {
  int fd = open(filename, O_RDONLY);
  if (fd < 0)
    errx(EX_OSERR, "unable to open %s", filename);
  read_fully(fd, &header_, sizeof(header_), filename);
  if (header_.magic != BLOCKED_TABLE_MAGIC || header_.version < 1 ||
      header_.version > 2 || header_.bucket_bytes != 64)
    errx(EX_DATAERR, "%s is not a version 1 or 2 blocked table", filename);
  if (header_.version < 2) {
    header_.kept_partitions = header_.taxon_codes;
    header_.taxon_codes = 0;
  }
  size_t bucket_bytes, offset_bytes, stash_bytes, code_bytes;
  blocked_sizes(header_, bucket_bytes, offset_bytes, stash_bytes, code_bytes);
  size_t total = BLOCKED_TABLE_HEADER_BYTES + bucket_bytes + offset_bytes +
                 stash_bytes + code_bytes;
  struct stat sb;
  if (fstat(fd, &sb) < 0 || (size_t) sb.st_size != total)
    errx(EX_DATAERR, "size mismatch in %s, aborting", filename);
  const char *base;
  if (memory_mapping) {
    close(fd);
    backing_file_.OpenFile(filename);
    base = backing_file_.fptr();
    file_backed_ = true;
  }
  else {
    // 2 MB-aligned so MADV_HUGEPAGE can back the table with huge pages; the
    // lookup touches one random bucket per query, so TLB reach matters.
    if (posix_memalign((void **) &owned_, (size_t) 1 << 21, total) != 0)
      errx(EX_OSERR, "unable to allocate %zu bytes for the blocked table", total);
    owned_bytes_ = total;
#ifdef MADV_HUGEPAGE
    madvise(owned_, total, MADV_HUGEPAGE);
#endif
    pread_parallel(fd, owned_, total, 0, db_read_threads(), filename);
    close(fd);
    base = owned_;
  }
  const char *p = base + BLOCKED_TABLE_HEADER_BYTES;
  buckets_ = (const uint8_t *) p;
  p += bucket_bytes;
  stash_offsets_ = (const uint64_t *) p;
  p += offset_bytes;
  stash_keys_ = (const uint64_t *) p;
  p += header_.stash_entries * sizeof(uint64_t);
  stash_vals_ = (const uint32_t *) p;
  p += header_.stash_entries * sizeof(uint32_t);
  if (header_.version >= 2) {
    code_taxa_ = (const uint32_t *) p;
    if (code_taxa_[0] != 0)
      errx(EX_DATAERR, "%s has a nonzero taxon for code 0", filename);
  }
  Init();
}

void BlockedTable::Init() {
  if (header_.slots != (uint32_t) ((BLOCKED_BUCKET_BITS - BLOCKED_COUNT_BITS - 1) /
                                   (header_.remainder_bits + header_.value_bits)))
    errx(EX_DATAERR, "blocked table header has an inconsistent slot count");
  if (header_.version >= 2 && header_.taxon_codes >> header_.value_bits)
    errx(EX_DATAERR, "blocked table has more taxon codes than its value width holds");
  geo_.init(header_.pext_mask, header_.partition_bits,
            header_.buckets_per_partition, header_.remainder_bits,
            header_.value_bits, header_.kept_partitions);
  if (geo_.K > geo_.P)
    errx(EX_DATAERR, "blocked table keeps more partitions than it has");
  if ((uint32_t) geo_.sf != header_.spill_bits || (uint32_t) geo_.B != header_.key_bits)
    errx(EX_DATAERR, "blocked table header is inconsistent");
  BlockedDispatch<BLOCKED_MIN_SLOT_BITS>::pick(geo_.w, &get_fn_, &batch_fn_);
}

hvalue_t BlockedTable::Get(hkey_t key) const {
  hvalue_t c = get_fn_(*this, geo_.hash(key));
  return code_taxa_ ? code_taxa_[c] : c;
}

hvalue_t BlockedTable::GetByHash(uint64_t h) const {
  hvalue_t c = get_fn_(*this, h);
  return code_taxa_ ? code_taxa_[c] : c;
}

void BlockedTable::GetBatch(const hkey_t *keys, hvalue_t *out, size_t n) const {
  static const int D = blocked_prefetch_dist();
  batch_fn_(*this, keys, out, n, D);
  if (code_taxa_)
    for (size_t i = 0; i < n; i++)
      out[i] = code_taxa_[out[i]];
}

// ------------------------------------------------------------------ builder

BlockedTableBuilder::BlockedTableBuilder(uint64_t pext_mask, int partition_bits,
                                         double load, const Taxonomy &tax,
                                         int remainder_bits, uint64_t kept_partitions)
    : pext_mask_(pext_mask),
      p_(partition_bits < 0 ? BLOCKED_AUTO_MAX_BITS : partition_bits),
      auto_p_(partition_bits < 0), load_(load),
      remainder_bits_(remainder_bits), kept_(kept_partitions), tax_(tax),
      parts_(1ULL << p_), reduced_size_(1ULL << p_, 0),
      locks_(1ULL << p_) {
  pext_.init(pext_mask);
  B_ = pext_.width;
  if (B_ > 62 || B_ < p_ + 8)
    errx(EX_SOFTWARE, "blocked table needs between %d and 62 key bits, not %d",
         p_ + 8, B_);
  mix_.init(B_);
  if (load_ <= 0 || load_ >= 1)
    errx(EX_USAGE, "blocked table load must be between 0 and 1");
  if (auto_p_ && kept_)
    errx(EX_USAGE, "keeping some partitions needs an explicit partition count");
  if (kept_ > (1ULL << p_))
    errx(EX_USAGE, "blocked table has only %llu partitions to keep",
         (unsigned long long) (1ULL << p_));
}

BlockedTableBuilder::Buffer::Buffer(BlockedTableBuilder &b)
    : builder_(b), parts_(b.parts_.size()) {}

BlockedTableBuilder::Buffer::~Buffer() {
  Flush();
}

void BlockedTableBuilder::Buffer::Add(uint64_t minimizer, uint32_t taxid) {
  if (minimizer & ~builder_.pext_mask_)
    errx(EX_SOFTWARE, "minimizer %016llx has bits outside the spaced seed mask",
         (unsigned long long) minimizer);
  uint64_t h = builder_.mix_(builder_.pext_(minimizer));
  size_t part = h >> (builder_.B_ - builder_.p_);
  Rec rec;
  rec.h = h;
  rec.taxid = taxid;
  parts_[part].push_back(rec);
  if (parts_[part].size() >= 4096)
    builder_.Append(part, parts_[part]);
}

void BlockedTableBuilder::Buffer::Flush() {
  for (size_t i = 0; i < parts_.size(); i++)
    if (! parts_[i].empty())
      builder_.Append(i, parts_[i]);
}

// Sorts by hash and merges duplicates by LCA.  The result depends only on
// the multiset of records.
void BlockedTableBuilder::Reduce(vector<Rec> &v) {
  std::sort(v.begin(), v.end(), [](const Rec &a, const Rec &b) { return a.h < b.h; });
  size_t o = 0;
  for (size_t i = 0; i < v.size(); ) {
    uint64_t h = v[i].h;
    uint64_t t = 0;
    while (i < v.size() && v[i].h == h)
      t = tax_.LowestCommonAncestor(t, v[i++].taxid);
    v[o].h = h;
    v[o].taxid = (uint32_t) t;
    o++;
  }
  v.resize(o);
}

void BlockedTableBuilder::Append(size_t part, vector<Rec> &recs) {
  std::lock_guard<std::mutex> g(locks_[part]);
  vector<Rec> &dst = parts_[part];
  dst.insert(dst.end(), recs.begin(), recs.end());
  recs.clear();
  // Keep memory near the distinct-key count by reducing whenever the
  // partition has doubled since its last reduction.
  if (dst.size() > 2 * std::max(reduced_size_[part], (size_t) 1 << 20)) {
    Reduce(dst);
    reduced_size_[part] = dst.size();
  }
}

static inline void set_bits(uint64_t *b, int off, int len, uint64_t val) {
  int wi = off >> 6, bi = off & 63;
  b[wi] |= val << bi;
  if (bi + len > 64)
    b[wi + 1] |= val >> (64 - bi);
}

uint64_t BlockedTableBuilder::Write(const char *filename, int taxonomy_bits, int threads) {
  const int AUGMENT_LIMIT = 2000;  // buckets visited per augmenting search

  #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (uint64_t part = 0; part < (uint64_t) parts_.size(); part++) {
    Reduce(parts_[part]);
    parts_[part].shrink_to_fit();
  }
  uint64_t N = 0;
  for (auto &v : parts_)
    N += v.size();

  // With an automatic partition count, pairs were collected at the largest
  // count; now that N is known, merge runs of 2^d neighbouring partitions.
  // A partition is the top bits of the hash and each is sorted by hash, so
  // concatenating neighbours gives exactly the sorted partition the smaller
  // count would have collected.
  if (auto_p_) {
    int p = 0;
    while (p < p_ && (N >> (p + 1)) >= BLOCKED_AUTO_MIN_KEYS)
      p++;
    int d = p_ - p;
    if (d > 0) {
      vector<vector<Rec> > merged(1ULL << p);
      for (uint64_t j = 0; j < merged.size(); j++) {
        size_t n = 0;
        for (uint64_t i = j << d; i < (j + 1) << d; i++)
          n += parts_[i].size();
        merged[j].reserve(n);
        for (uint64_t i = j << d; i < (j + 1) << d; i++) {
          merged[j].insert(merged[j].end(), parts_[i].begin(), parts_[i].end());
          vector<Rec>().swap(parts_[i]);
        }
      }
      parts_.swap(merged);
      p_ = p;
    }
  }
  const uint64_t P = parts_.size();
  const uint64_t K = kept_ ? kept_ : P;
  const int q = B_ - p_;
  for (uint64_t part = 0; part < P; part++)
    for (auto &rec : parts_[part])
      if (rec.taxid >> taxonomy_bits || rec.taxid >= tax_.node_count())
        errx(EX_SOFTWARE, "taxon %u does not fit in %d bits", rec.taxid, taxonomy_bits);

  // Entries of one partition as (home << r | rem, internal taxon), sorted
  // and distinct.  In exact mode recs are sorted by hash, and home and rem
  // are nondecreasing in the low q bits of the hash, so the entries are
  // already sorted and distinct.  In fingerprint mode rem is the low r bits
  // of the offset, so the entries are sorted and those with the same key
  // merged by LCA; ex then gets the taxon each record's lookup must return.
  auto make_entries = [&](uint64_t part, int r, uint64_t nb, bool exact,
                          vector<pair<uint64_t, uint32_t> > &e,
                          vector<uint32_t> *ex) {
    const vector<Rec> &recs = parts_[part];
    const uint64_t qmask = (1ULL << q) - 1, rmask = (1ULL << r) - 1;
    e.resize(recs.size());
    for (size_t i = 0; i < recs.size(); i++) {
      uint64_t low = recs[i].h & qmask;
      uint64_t home = (uint64_t) (((unsigned __int128) low * nb) >> q);
      uint64_t lo = (uint64_t) ((((unsigned __int128) home << q) + nb - 1) / nb);
      uint64_t rem = low - lo;
      if (exact && rem > rmask)
        errx(EX_SOFTWARE, "remainder exceeds %d bits", r);
      e[i] = std::make_pair((home << r) | (rem & rmask), recs[i].taxid);
    }
    if (exact)
      return;
    vector<uint32_t> order(e.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return e[a].first != e[b].first ? e[a].first < e[b].first : a < b;
    });
    if (ex)
      ex->resize(e.size());
    vector<pair<uint64_t, uint32_t> > m;
    for (size_t i = 0; i < order.size(); ) {
      uint64_t key = e[order[i]].first;
      size_t j = i;
      uint64_t t = 0;
      while (j < order.size() && e[order[j]].first == key)
        t = tax_.LowestCommonAncestor(t, e[order[j++]].second);
      if (ex)
        for (size_t a = i; a < j; a++)
          (*ex)[order[a]] = (uint32_t) t;
      m.push_back(std::make_pair(key, (uint32_t) t));
      i = j;
    }
    e.swap(m);
  };

  // Entry count of every internal taxon over the kept partitions, so codes
  // number only the taxa the table stores.  In exact mode entries are keys.
  auto count_taxa = [&](int r, uint64_t nb, bool exact) {
    vector<uint64_t> count(tax_.node_count(), 0);
    std::mutex count_lock;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (uint64_t part = 0; part < K; part++) {
      vector<uint32_t> t;
      if (exact) {
        t.resize(parts_[part].size());
        for (size_t i = 0; i < t.size(); i++)
          t[i] = parts_[part][i].taxid;
      }
      else {
        vector<pair<uint64_t, uint32_t> > e;
        make_entries(part, r, nb, false, e, nullptr);
        t.resize(e.size());
        for (size_t i = 0; i < t.size(); i++)
          t[i] = e[i].second;
      }
      std::sort(t.begin(), t.end());
      std::lock_guard<std::mutex> g(count_lock);
      for (size_t i = 0; i < t.size(); ) {
        size_t j = i;
        while (j < t.size() && t[j] == t[i])
          j++;
        count[t[i]] += j - i;
        i = j;
      }
    }
    return count;
  };
  auto width = [](uint64_t x) { return x ? 64 - __builtin_clzll(x) : 1; };
  auto distinct = [](const vector<uint64_t> &c) {
    uint64_t u = 0;
    for (size_t t = 1; t < c.size(); t++) u += c[t] != 0;
    return u;
  };

  // nb depends on S, which depends on r and v.  Raising r only shrinks S,
  // grows nb and lowers the exact width, so the first r that reaches the
  // exact width is the smallest one that keeps every key distinct.  r starts
  // high enough for the minimum slot width the lookup supports.  A requested
  // r is used as given.
  auto size_table = [&](int value_bits, int &r, int &S, int &r_exact, uint64_t &nb) {
    r = remainder_bits_ > 0 ? remainder_bits_
                            : std::max(1, BLOCKED_MIN_SLOT_BITS - value_bits);
    if (r + value_bits < BLOCKED_MIN_SLOT_BITS)
      errx(EX_USAGE, "blocked table slot needs at least %d bits; raise -R to %d",
           BLOCKED_MIN_SLOT_BITS, BLOCKED_MIN_SLOT_BITS - value_bits);
    while (true) {
      int w = r + value_bits;
      if (w > BLOCKED_MAX_SLOT_BITS)
        errx(EX_SOFTWARE, "blocked table slot would need %d bits", w);
      S = (BLOCKED_BUCKET_BITS - BLOCKED_COUNT_BITS - 1) / w;
      nb = (uint64_t) ceil((double) N / P / (S * load_));
      if (nb < 2)
        nb = 2;
      uint64_t range = (uint64_t) ((((unsigned __int128) 1 << q) + nb - 1) / nb);
      r_exact = range <= 1 ? 1 : 64 - __builtin_clzll(range - 1);
      if (r >= r_exact || remainder_bits_ > 0)
        break;
      r++;
    }
  };

  // The value width v is the width of U, the number of distinct stored taxa.
  // In fingerprint mode the merge by (home, rem) can add ancestors that no
  // key carries, and the merge depends on nb, which depends on v; v only
  // grows, so this settles in a step or two.
  vector<uint64_t> taxon_count = count_taxa(0, 0, true);
  const uint64_t key_taxa = distinct(taxon_count);
  int value_bits = width(key_taxa);
  int r = 0, S = 0, r_exact = 0;
  uint64_t nb = 0;
  while (true) {
    size_table(value_bits, r, S, r_exact, nb);
    if (r >= r_exact)
      break;
    taxon_count = count_taxa(r, nb, false);
    int vb = width(distinct(taxon_count));
    if (vb <= value_bits)
      break;
    value_bits = vb;
  }
  const bool exact = r >= r_exact;
  if (S > 127)
    errx(EX_SOFTWARE, "blocked table bucket would hold %d slots", S);

  // Codes 1..U in order of decreasing entry count, ties by internal taxon.
  // Taxon 0 keeps code 0, which a lookup reports as absent either way.
  vector<uint32_t> code_taxa(1, 0);
  for (size_t t = 1; t < taxon_count.size(); t++)
    if (taxon_count[t])
      code_taxa.push_back((uint32_t) t);
  std::sort(code_taxa.begin() + 1, code_taxa.end(), [&](uint32_t a, uint32_t b) {
    return taxon_count[a] != taxon_count[b] ? taxon_count[a] > taxon_count[b] : a < b;
  });
  const uint64_t U = code_taxa.size() - 1;
  vector<uint32_t> taxon_code(tax_.node_count(), 0);
  for (uint64_t c = 1; c <= U; c++)
    taxon_code[code_taxa[c]] = (uint32_t) c;

  blocked::Geometry G;
  G.init(pext_mask_, p_, nb, r, value_bits, K);
  const int w = G.w, H = G.H, sf = G.sf;
  const uint64_t rmask = G.rmask;

  size_t bucket_bytes = K * nb * 64;
  uint64_t *table;
  if (posix_memalign((void **) &table, (size_t) 1 << 21, bucket_bytes) != 0)
    errx(EX_OSERR, "unable to allocate %zu bytes for the blocked table", bucket_bytes);
  memset(table, 0, bucket_bytes);

  vector<vector<uint64_t> > stash_keys(K);
  vector<vector<uint32_t> > stash_vals(K);
  vector<uint64_t> guests(K, 0), spilled(K, 0), stored(K, 0);
  // Fingerprint mode: the taxon each record's lookup must return, which is
  // the LCA of every record sharing its (home, rem).
  vector<vector<uint32_t> > expect(exact ? 0 : K);

  #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (uint64_t part = 0; part < K; part++) {
    // Entries as (home << r | rem, taxon code), sorted.
    vector<pair<uint64_t, uint32_t> > e;
    make_entries(part, r, nb, exact, e, exact ? nullptr : &expect[part]);
    for (auto &x : e)
      x.second = taxon_code[x.second];
    stored[part] = e.size();
    uint64_t base = part * nb;
    // Placement on a flat slot array: cnt[b] entries of bucket b live in
    // slot[b*S .. b*S+cnt[b]), as indices into e; at_alt[i] says whether
    // entry i sits in its alt bucket.
    vector<uint8_t> cnt(nb, 0);
    vector<uint64_t> spill(nb, 0);
    vector<uint32_t> slot(nb * S);
    vector<uint8_t> at_alt(e.size(), 0);
    auto home_of = [&](uint32_t i) { return e[i].first >> r; };
    auto alt_of = [&](uint32_t i) {
      uint64_t home = e[i].first >> r, rem = e[i].first & rmask;
      return (home + 1 + blocked::murmur(rem ^ blocked::ALT_SALT) % (nb - 1)) % nb;
    };
    auto sbit = [&](uint32_t i) -> uint64_t {
      uint64_t rem = e[i].first & rmask;
      return sf == 1 ? 0 : blocked::murmur(rem ^ blocked::SPILL_SALT) % sf;
    };
    // Natives: the first S entries of each home bucket, by remainder.
    vector<uint32_t> over;
    for (size_t i = 0; i < e.size(); ) {
      uint64_t home = home_of(i);
      size_t j = i;
      while (j < e.size() && home_of(j) == home)
        j++;
      for (size_t a = i; a < j; a++) {
        if (a < i + S)
          slot[home * S + cnt[home]++] = a;
        else
          over.push_back(a);
      }
      i = j;
    }
    // Overflow goes to alt buckets in (alt, rem, index) order while they
    // have room.
    std::sort(over.begin(), over.end(), [&](uint32_t a, uint32_t b) {
      uint64_t aa = alt_of(a), ab = alt_of(b);
      if (aa != ab) return aa < ab;
      uint64_t ra = e[a].first & rmask, rb = e[b].first & rmask;
      return ra != rb ? ra < rb : a < b;
    });
    vector<uint32_t> left;
    for (auto i : over) {
      uint64_t a = alt_of(i);
      if (cnt[a] < S) {
        slot[a * S + cnt[a]++] = i;
        at_alt[i] = 1;
      }
      else {
        left.push_back(i);
      }
    }
    // Bounded breadth-first search for augmenting paths: move placed
    // entries between their two buckets to free a slot.  Entries are handled
    // in a fixed order by one thread per partition, so the result depends
    // only on the key set.
    vector<uint32_t> stash_idx;
    if (! left.empty()) {
      std::sort(left.begin(), left.end());
      vector<int64_t> from_bucket(nb, -2);
      vector<uint32_t> via_entry(nb);
      vector<uint64_t> touched, queue;
      for (auto k : left) {
        queue.clear();
        touched.clear();
        uint64_t h0 = home_of(k), a0 = alt_of(k);
        uint64_t starts[2] = {h0, a0};
        for (int si = 0; si < 2; si++) {
          uint64_t s0 = starts[si];
          if (from_bucket[s0] != -2) continue;
          from_bucket[s0] = -1;
          touched.push_back(s0);
          queue.push_back(s0);
        }
        int64_t found = -1;
        for (size_t qi = 0; qi < queue.size() && found < 0; qi++) {
          uint64_t X = queue[qi];
          if (cnt[X] < S) { found = X; break; }
          if (touched.size() >= (size_t) AUGMENT_LIMIT) continue;
          for (int s = 0; s < cnt[X]; s++) {
            uint32_t j = slot[X * S + s];
            uint64_t Y = at_alt[j] ? home_of(j) : alt_of(j);
            if (from_bucket[Y] != -2) continue;
            from_bucket[Y] = X;
            via_entry[Y] = j;
            touched.push_back(Y);
            queue.push_back(Y);
          }
        }
        if (found >= 0) {
          // Walk back: move via_entry[Y] from from_bucket[Y] into Y.
          uint64_t Y = found;
          while (from_bucket[Y] >= 0) {
            uint64_t X = from_bucket[Y];
            uint32_t j = via_entry[Y];
            int s = 0;
            while (slot[X * S + s] != j) s++;
            slot[X * S + s] = slot[X * S + --cnt[X]];
            slot[Y * S + cnt[Y]++] = j;
            at_alt[j] ^= 1;
            Y = X;
          }
          slot[Y * S + cnt[Y]++] = k;
          at_alt[k] = (Y != h0);
        }
        else {
          stash_idx.push_back(k);
        }
        for (auto b : touched)
          from_bucket[b] = -2;
      }
    }
    // Stash, sorted by (home, rem).
    std::sort(stash_idx.begin(), stash_idx.end());
    for (auto i : stash_idx) {
      stash_keys[part].push_back(e[i].first);
      stash_vals[part].push_back(e[i].second);
      spill[home_of(i)] |= 1ULL << sbit(i);
    }
    // Pack each bucket: natives by remainder, then guests by (rem, index).
    uint64_t g_ct = 0, sp = 0;
    vector<uint32_t> nat_l, gst_l;
    vector<uint8_t> nat_ct(nb, 0);
    for (uint64_t b = 0; b < nb; b++) {
      nat_l.clear();
      gst_l.clear();
      for (int s = 0; s < cnt[b]; s++) {
        uint32_t i = slot[b * S + s];
        if (at_alt[i]) {
          gst_l.push_back(i);
          spill[home_of(i)] |= 1ULL << sbit(i);
        }
        else {
          nat_l.push_back(i);
        }
      }
      std::sort(nat_l.begin(), nat_l.end());
      std::sort(gst_l.begin(), gst_l.end(), [&](uint32_t a, uint32_t c) {
        uint64_t ra = e[a].first & rmask, rc = e[c].first & rmask;
        return ra != rc ? ra < rc : a < c;
      });
      uint64_t *bw = table + (base + b) * 8;
      int f = 0;
      for (auto i : nat_l)
        set_bits(bw, H + f++ * w, w, (e[i].first & rmask) | ((uint64_t) e[i].second << r));
      for (auto i : gst_l)
        set_bits(bw, H + f++ * w, w, (e[i].first & rmask) | ((uint64_t) e[i].second << r));
      g_ct += gst_l.size();
      nat_ct[b] = nat_l.size();
    }
    for (uint64_t b = 0; b < nb; b++) {
      uint64_t *bw = table + (base + b) * 8;
      set_bits(bw, 0, BLOCKED_COUNT_BITS, nat_ct[b] | ((uint64_t) cnt[b] << 7));
      set_bits(bw, BLOCKED_COUNT_BITS, sf, spill[b]);
      sp += spill[b] != 0;
    }
    guests[part] = g_ct;
    spilled[part] = sp;
  }

  // Flatten the stash.
  vector<uint64_t> offsets(K + 1, 0), skeys;
  vector<uint32_t> svals;
  for (uint64_t part = 0; part < K; part++) {
    offsets[part + 1] = offsets[part] + stash_keys[part].size();
    skeys.insert(skeys.end(), stash_keys[part].begin(), stash_keys[part].end());
    svals.insert(svals.end(), stash_vals[part].begin(), stash_vals[part].end());
  }

  BlockedTableHeader hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.magic = BLOCKED_TABLE_MAGIC;
  hdr.version = 2;
  hdr.bucket_bytes = 64;
  hdr.partition_bits = p_;
  hdr.key_bits = B_;
  hdr.remainder_bits = r;
  hdr.value_bits = value_bits;
  hdr.slots = S;
  hdr.spill_bits = sf;
  hdr.buckets_per_partition = nb;
  hdr.kept_partitions = kept_ ? K : 0;
  uint64_t entries = 0;
  for (uint64_t part = 0; part < K; part++)
    entries += stored[part];
  hdr.entries = entries;
  hdr.stash_entries = skeys.size();
  hdr.pext_mask = pext_mask_;
  for (uint64_t part = 0; part < K; part++) {
    hdr.guests += guests[part];
    hdr.spilled_buckets += spilled[part];
  }
  hdr.taxon_codes = U;

  // Self-check: every key must come back with its own taxon (in fingerprint
  // mode, the LCA of its entry) through the same lookup code classify uses,
  // and every key of a dropped partition must miss.
  {
    BlockedTable T;
    T.header_ = hdr;
    T.buckets_ = (const uint8_t *) table;
    T.stash_offsets_ = offsets.data();
    T.stash_keys_ = skeys.data();
    T.stash_vals_ = svals.data();
    T.code_taxa_ = code_taxa.data();
    T.Init();
    uint64_t bad = 0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:bad) num_threads(threads)
    for (uint64_t part = 0; part < P; part++) {
      const vector<Rec> &recs = parts_[part];
      for (size_t i = 0; i < recs.size(); i++)
        bad += T.GetByHash(recs[i].h) !=
               (part >= K ? 0 : exact ? recs[i].taxid : expect[part][i]);
    }
    if (bad)
      errx(EX_SOFTWARE, "blocked table self-check failed for %llu keys",
           (unsigned long long) bad);
  }

  FILE *f = fopen(filename, "wb");
  if (! f)
    err(EX_CANTCREAT, "unable to create %s", filename);
  vector<char> pad(BLOCKED_TABLE_HEADER_BYTES, 0);
  memcpy(pad.data(), &hdr, sizeof(hdr));
  bool ok = fwrite(pad.data(), 1, pad.size(), f) == pad.size();
  ok = ok && fwrite(table, 1, bucket_bytes, f) == bucket_bytes;
  ok = ok && fwrite(offsets.data(), sizeof(uint64_t), offsets.size(), f) == offsets.size();
  if (! skeys.empty()) {
    ok = ok && fwrite(skeys.data(), sizeof(uint64_t), skeys.size(), f) == skeys.size();
    ok = ok && fwrite(svals.data(), sizeof(uint32_t), svals.size(), f) == svals.size();
  }
  ok = ok && fwrite(code_taxa.data(), sizeof(uint32_t), code_taxa.size(), f) == code_taxa.size();
  ok = (fclose(f) == 0) && ok;
  if (! ok)
    err(EX_IOERR, "error writing %s", filename);
  free(table);

  // Records whose own taxon differs from the LCA they now share.
  uint64_t merged_changed = 0;
  if (! exact)
    for (uint64_t part = 0; part < K; part++)
      for (size_t i = 0; i < parts_[part].size(); i++)
        merged_changed += parts_[part][i].taxid != expect[part][i];
  std::cerr << "Blocked table: " << N << " keys";
  if (! exact)
    std::cerr << " in " << entries << " entries (" << merged_changed
              << " keys with a merged taxon)";
  uint64_t kept_keys = 0;
  for (uint64_t part = 0; part < K; part++)
    kept_keys += parts_[part].size();
  if (K < P)
    std::cerr << ", " << kept_keys << " kept in " << K << " of " << P
              << " partitions";
  std::cerr << ", " << P << " partitions of "
            << nb << " buckets, " << (exact ? "exact" : "fingerprint")
            << " r = " << r << " (exact " << r_exact << "), v = " << value_bits
            << " for " << U << " taxon codes (" << key_taxa
            << " taxa on keys, taxonomy " << tax_.node_count() << " nodes, "
            << taxonomy_bits << " bits)"
            << ", " << S << " slots of " << w << " bits, spill filter "
            << sf << " bits, " << (double) bucket_bytes * 8 / N
            << " bits/key, load " << (double) entries / (K * nb * S)
            << ", guests " << hdr.guests << ", stash " << hdr.stash_entries
            << std::endl;
  return N;
}

}  // end namespace
