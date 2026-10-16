// Memory sectors per lookup, counted from addresses rather than measured.
//
// Replays the lookups the fused kernel makes for the reads of a FASTQ file
// (one per run of equal consecutive minimizers, mates without ambiguous bases
// only) against both tables, and for each lookup records which 32-byte
// sectors and which 64-byte segments of the table the device code reads: the
// probed cells of the compact hash table (CHT), or the header word and slot
// windows of the blocked table's home and, when read, alt bucket, as the
// aligned 32-bit loads of k2t::ld64 cover them.  A 32-byte sector is the unit
// of an L2 request on NVIDIA GPUs; 64 bytes is the smallest DRAM transfer.
//
// usage: sectors reads.fq cht/hash.k2d blk/hash.k2d opts.k2d [reads]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "table.h"
#include "classify_core.h"

static std::vector<char> slurp(const char *path) {
  int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st) != 0) { perror(path); exit(1); }
  std::vector<char> b(st.st_size);
  for (size_t got = 0; got < b.size(); ) {
    ssize_t n = read(fd, b.data() + got, b.size() - got);
    if (n <= 0) { perror(path); exit(1); }
    got += n;
  }
  close(fd);
  return b;
}

// Distinct units of 2^shift bytes among the byte ranges a lookup reads.
struct Touch {
  uint64_t lo[16], hi[16]; int n = 0;
  void add(uint64_t a, uint64_t len) { if (n < 16) { lo[n] = a; hi[n] = a + len - 1; n++; } }
  int count(int shift) const {
    uint64_t u[64]; int m = 0;
    for (int i = 0; i < n; i++)
      for (uint64_t x = lo[i] >> shift; x <= hi[i] >> shift; x++) {
        int j = 0; while (j < m && u[j] != x) j++;
        if (j == m && m < 64) u[m++] = x;
      }
    return m;
  }
};

// The bytes k2t::ld64 reads on the device for an 8-byte window at p.
static void touch_ld64(Touch &t, uint64_t p) {
  uint64_t w = p & ~3ULL;
  t.add(w, (p & 3) ? 12 : 8);
}

struct Stats {
  uint64_t n = 0, s32 = 0, s64 = 0, found = 0, s32_found = 0, s32_absent = 0, alt = 0;
  uint64_t hist[8] = {0};
  void add(const Touch &t, bool hit, bool used_alt) {
    int a = t.count(5), b = t.count(6);
    n++; s32 += a; s64 += b; alt += used_alt;
    if (hit) { found++; s32_found += a; } else s32_absent += a;
    hist[a < 7 ? a : 7]++;
  }
  void print(const char *name) const {
    printf("%-8s lookups %" PRIu64 "  32-byte sectors per lookup %.3f (present %.3f, absent %.3f)"
           "  64-byte segments per lookup %.3f  alt bucket read %.2f%%\n",
           name, n, (double) s32 / n, found ? (double) s32_found / found : 0.0,
           n > found ? (double) s32_absent / (n - found) : 0.0, (double) s64 / n, 100.0 * alt / n);
    printf("         lookups by 32-byte sectors read:");
    for (int i = 1; i < 8; i++) printf("  %d%s: %.2f%%", i, i == 7 ? "+" : "", 100.0 * hist[i] / n);
    printf("\n");
  }
};

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: sectors reads.fq cht/hash.k2d blk/hash.k2d opts.k2d [reads]\n");
    return 1;
  }
  size_t limit = argc > 5 ? (size_t) atoll(argv[5]) : 1000000;
  uint64_t o[8];
  { std::vector<char> ob = slurp(argv[4]); memcpy(o, ob.data(), sizeof o); }
  int k = (int) o[0], l = (int) o[1];
  uint64_t ssm = o[2], tog = o[3], mahv = o[5];

  std::vector<char> cf = slurp(argv[2]), bf = slurp(argv[3]);
  k2t::Table cht{}, blk{};
  { uint64_t hdr[4]; memcpy(hdr, cf.data(), sizeof hdr);
    cht.kind = k2t::Table::CHT; cht.capacity = hdr[0];
    cht.key_bits = (int) hdr[2]; cht.value_bits = (int) hdr[3];
    cht.cells = (const uint32_t *) (cf.data() + 32); }
  { k2t::BlockedHeader bh; memcpy(&bh, bf.data(), sizeof bh);
    if (bh.magic != k2t::BLOCKED_MAGIC) { fprintf(stderr, "%s is not a blocked table\n", argv[3]); return 1; }
    blk.kind = k2t::Table::BLOCKED; blk.G.init(bh);
    size_t nbk = (size_t) (blk.G.P * blk.G.nb) * 64;
    size_t off_o = k2t::BLOCKED_HEADER_BYTES + nbk, off_k = off_o + (blk.G.P + 1) * 8;
    blk.buckets = (const uint8_t *) (bf.data() + k2t::BLOCKED_HEADER_BYTES);
    blk.stash_off = (const uint64_t *) (bf.data() + off_o);
    blk.stash_key = (const uint64_t *) (bf.data() + off_k);
    blk.stash_val = (const uint32_t *) (bf.data() + off_k + bh.stash_entries * 8); }
  const k2t::Geometry &G = blk.G;

  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 1; }
  Stats sc, sb;
  uint64_t mismatch = 0;
  char line[8192];
  std::vector<uint64_t> cand(400), val(400);
  size_t nr = 0;
  while (nr < limit && fgets(line, sizeof line, f)) {
    if (!fgets(line, sizeof line, f)) break;
    size_t L = strlen(line); while (L && (line[L-1] == '\n' || line[L-1] == '\r')) L--;
    char skip[8192];
    if (!fgets(skip, sizeof skip, f) || !fgets(skip, sizeof skip, f)) break;
    nr++;
    bool amb = false;
    for (size_t i = 0; i < L; i++) if (k2c::base_code(line[i]) < 0) amb = true;
    if (amb || L > 320) continue;
    int n_lmer = (int) L - l + 1, n_kmer = (int) L - k + 1, win = k - l + 1;
    if (n_kmer < 1) continue;
    const uint64_t lmask = (l * 2 >= 64) ? ~0ULL : ((1ULL << (l * 2)) - 1);
    for (int i = 0; i < n_lmer; i++) {
      uint64_t lmer = 0;
      for (int j = 0; j < l; j++) lmer = (lmer << 2) | (uint64_t) k2c::base_code(line[i + j]);
      lmer &= lmask;
      uint64_t rc = k2c::revcomp(lmer, l), canon = lmer < rc ? lmer : rc;
      if (ssm) canon &= ssm;
      cand[i] = canon ^ tog;
    }
    for (int p = 0; p < n_kmer; p++) {
      uint64_t mn = ~0ULL;
      for (int j = 0; j < win; j++) if (cand[p + j] < mn) mn = cand[p + j];
      val[p] = mn ^ tog;
    }
    for (int p = 0; p < n_kmer; p++) {
      if (p > 0 && val[p - 1] == val[p]) continue;
      uint64_t key = val[p];
      if (mahv && k2t::murmur(key) < mahv) continue;

      // CHT: the cells probed, 4 bytes each, from the home index on.
      Touch tc;
      {
        uint64_t hc = k2t::murmur(key);
        uint32_t compacted = (uint32_t) (hc >> (64 - cht.key_bits));
        uint64_t idx = hc % cht.capacity, first = idx;
        const uint32_t vmask = (1u << cht.value_bits) - 1;
        while (true) {
          tc.add(32 + idx * 4, 4);
          uint32_t cell = cht.cells[idx], v = cell & vmask;
          if (!v || (cell >> cht.value_bits) == compacted) break;
          if (++idx == cht.capacity) idx = 0;
          if (idx == first) break;
        }
      }
      uint32_t vc = cht.get(key);
      sc.add(tc, vc != 0, false);

      // Blocked: home bucket header and native slots; alt bucket header and
      // guest slots when the spill bit is set and home had no match.
      Touch tb;
      uint64_t h = G.hash(key), part = G.part_of(h), home, rem;
      G.locate(h, home, rem);
      uint64_t hb = k2t::BLOCKED_HEADER_BYTES + ((part * G.nb + home) << 6);
      const uint8_t *hp = blk.buckets + ((part * G.nb + home) << 6);
      touch_ld64(tb, hb);
      uint64_t h64; memcpy(&h64, hp, 8);
      auto touch_slots = [&](uint64_t base, unsigned lo, unsigned hi) {
        for (unsigned i = lo; i < hi; i++) {
          unsigned off = (unsigned) G.H + i * (unsigned) G.w;
          unsigned byte = (off >> 3) < 56 ? (off >> 3) : 56;
          touch_ld64(tb, base + byte);
        }
      };
      touch_slots(hb, 0, (unsigned) (h64 & 127));
      uint32_t vb = blk.scan(hp, 0, (unsigned) (h64 & 127), rem);
      bool used_alt = false;
      if (!vb && ((h64 >> (k2t::COUNT_BITS + G.spill_bit(rem))) & 1)) {
        used_alt = true;
        uint64_t a = G.alt(home, rem);
        uint64_t ab = k2t::BLOCKED_HEADER_BYTES + ((part * G.nb + a) << 6);
        const uint8_t *ap = blk.buckets + ((part * G.nb + a) << 6);
        touch_ld64(tb, ab);
        uint64_t a64; memcpy(&a64, ap, 8);
        touch_slots(ab, (unsigned) (a64 & 127), (unsigned) ((a64 >> 7) & 127));
      }
      uint32_t vbf = blk.get(key);
      sb.add(tb, vbf != 0, used_alt);
      if ((vc != 0) != (vbf != 0)) mismatch++;
    }
  }
  printf("reads %zu (mates with ambiguous bases skipped)\n", nr);
  sc.print("CHT");
  sb.print("blocked");
  printf("lookups present in one table and absent from the other: %" PRIu64
         " (CHT false positives)\n", mismatch);
  return 0;
}
