// Host emulation of the fused device kernel, checked against k2mindump.
//
// GPU time is the scarce resource, so everything in the kernel that is not
// CUDA-specific gets verified here first: the l-mer construction, the
// argmin-based reporting rule, consecutive-duplicate removal by walking back to
// the previous reporting position, the subsampling filter, the stable
// compaction that fixes output order, and an independent probe of the compact
// table that does not go through CompactHashTable.  What is left untested by
// this is only the parts that need a device: shared memory, the parallel prefix
// sum, and the launch itself.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static const int MAX_LEN = 320;

static inline uint64_t murmur(uint64_t k) {
  k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33; return k;
}

static inline uint64_t revcomp(uint64_t kmer, int n) {
  kmer = ((kmer & 0xCCCCCCCCCCCCCCCCULL) >> 2) | ((kmer & 0x3333333333333333ULL) << 2);
  kmer = ((kmer & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | ((kmer & 0x0F0F0F0F0F0F0F0FULL) << 4);
  kmer = ((kmer & 0xFF00FF00FF00FF00ULL) >> 8) | ((kmer & 0x00FF00FF00FF00FFULL) << 8);
  kmer = ((kmer & 0xFFFF0000FFFF0000ULL) >> 16) | ((kmer & 0x0000FFFF0000FFFFULL) << 16);
  kmer = (kmer >> 32) | (kmer << 32);
  return ((~kmer) >> (64 - n * 2)) & ((1ULL << (n * 2)) - 1);
}

// Kraken 2's tree resolution, over the handful of distinct taxa one read hits.
// A taxon is its own ancestor, so every taxon scores at least its own count and
// the winning score is never zero.  Ties fold with the lowest common ancestor,
// which is commutative and associative, so the answer does not depend on the
// order the taxa are visited -- which matters, because the CPU walks them in
// the arbitrary order of an unordered_map.
static inline bool is_ancestor(const uint64_t *parent, uint64_t a, uint64_t b) {
  if (!a || !b) return false;
  while (b > a) b = parent[b];
  return b == a;
}

static inline uint64_t lca(const uint64_t *parent, uint64_t a, uint64_t b) {
  if (!a || !b) return a ? a : b;
  while (a != b) {
    if (a > b) a = parent[a]; else b = parent[b];
  }
  return a;
}

static uint32_t resolve(const uint64_t *parent, const uint32_t *tax,
                        const uint32_t *cnt, int n) {
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

static inline uint32_t probe(const uint32_t *t, uint64_t cap, uint64_t key,
                             int key_bits, int value_bits) {
  uint64_t hc = murmur(key);
  uint32_t compacted = (uint32_t) (hc >> (64 - key_bits));
  uint64_t idx = hc % cap, first = idx;
  const uint32_t vmask = (1u << value_bits) - 1;
  while (true) {
    uint32_t cell = t[idx];
    uint32_t val = cell & vmask;
    if (!val) return 0;
    if ((cell >> value_bits) == compacted) return val;
    if (++idx == cap) idx = 0;
    if (idx == first) return 0;
  }
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: emul reads.fq hash.k2d opts.k2d taxo.k2d oracle [reads]\n"
                    "  oracle is either a k2mindump file (checks the taxa array)\n"
                    "  or classify's own output (checks the final call)\n");
    return 1;
  }
  size_t limit = argc > 6 ? (size_t) atoll(argv[6]) : 50000;

  uint64_t o[8];
  { int fd = open(argv[3], O_RDONLY);
    if (fd < 0 || read(fd, o, sizeof o) < (ssize_t) sizeof o) { perror("opts"); return 1; }
    close(fd); }
  int k = (int) o[0], l = (int) o[1];
  uint64_t ssm = o[2], tog = o[3], mahv = o[5];

  int hf = open(argv[2], O_RDONLY);
  struct stat st; fstat(hf, &st);
  char *map = (char *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, hf, 0);
  if (map == MAP_FAILED) { perror("mmap"); return 1; }
  uint64_t cap, sz, key_bits, value_bits;
  memcpy(&cap, map, 8); memcpy(&sz, map+8, 8);
  memcpy(&key_bits, map+16, 8); memcpy(&value_bits, map+24, 8);
  const uint32_t *table = (const uint32_t *) (map + 32);
  printf("k=%d l=%d capacity=%" PRIu64 " key_bits=%" PRIu64 " value_bits=%" PRIu64 "\n",
         k, l, cap, key_bits, value_bits);

  // Taxonomy: only the parent and external-id columns are needed.
  int tf = open(argv[4], O_RDONLY);
  if (tf < 0) { fprintf(stderr, "cannot open taxonomy %s\n", argv[4]); return 1; }
  struct stat tst; fstat(tf, &tst);
  char *tmap = (char *) mmap(nullptr, tst.st_size, PROT_READ, MAP_PRIVATE, tf, 0);
  if (tmap == MAP_FAILED) { perror("mmap taxo"); return 1; }
  if (memcmp(tmap, "K2TAXDAT", 8) != 0) { fprintf(stderr, "%s is not a taxonomy\n", argv[4]); return 1; }
  uint64_t node_count; memcpy(&node_count, tmap + 8, 8);
  const uint64_t *nodes = (const uint64_t *) (tmap + 32);   // 7 words per node
  std::vector<uint64_t> parent(node_count), extid(node_count);
  for (uint64_t i = 0; i < node_count; i++) {
    parent[i] = nodes[i * 7 + 0];
    extid[i]  = nodes[i * 7 + 5];
  }
  printf("taxonomy: %" PRIu64 " nodes\n", node_count);

  FILE *f = fopen(argv[1], "rb");
  if (! f) { fprintf(stderr, "cannot open reads %s\n", argv[1]); return 1; }
  FILE *rf = fopen(argv[5], "rb");
  if (! rf) { fprintf(stderr, "cannot open oracle %s\n", argv[5]); return 1; }

  // The oracle is either k2mindump's binary dump or classify's own output.
  // The second is the one that matters: it checks the call, not an
  // intermediate.
  char magic[8]; size_t z = fread(magic,1,8,rf); (void) z;
  bool call_mode = (memcmp(magic, "K2MDUMP", 7) != 0);
  bool taxa_mode = false;
  uint64_t rn = (uint64_t) -1;
  if (! call_mode) {
    char flags; z = fread(&flags,1,1,rf);
    z = fread(&rn,8,1,rf);
    if (flags != 2 && flags != 3) { fprintf(stderr, "oracle needs -H or -H -T\n"); return 1; }
    taxa_mode = (flags == 3);
  } else {
    rewind(rf);
    taxa_mode = true;   // the call is computed from the taxa array
  }
  printf("mode: %s\n", call_mode ? "final call vs classify" : "intermediate vs k2mindump");

  char line[8192];
  size_t nr = 0, bad = 0, pairs = 0;
  std::vector<uint64_t> cand(MAX_LEN), val(MAX_LEN);
  std::vector<int> flag(MAX_LEN), look(MAX_LEN), scan(MAX_LEN), tmp(MAX_LEN);
  std::vector<uint8_t> code(MAX_LEN);

  while (nr < limit && nr < rn && fgets(line, sizeof line, f)) {
    if (!fgets(line, sizeof line, f)) break;
    size_t L = strlen(line); while (L && (line[L-1]=='\n'||line[L-1]=='\r')) L--;
    if ((int) L > MAX_LEN) {
      // Truncating would silently misclassify the read.  Shared memory is sized
      // for MAX_LEN, so a longer read needs a different arrangement, not a
      // quiet shortening.
      fprintf(stderr, "read %zu is %zu bases, longer than the %d this build "
              "handles\n", nr, L, MAX_LEN);
      return 1;
    }
    const char *seq = line;

    for (size_t i = 0; i < L; i++) {
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
    int n_lmer = (int) L - l + 1; if (n_lmer < 1) n_lmer = 0;
    const uint64_t lmask = (l*2 >= 64) ? ~0ULL : ((1ULL << (l*2)) - 1);
    for (int i = 0; i < n_lmer; i++) {
      uint64_t lmer = 0; bool amb = false;
      for (int j = 0; j < l; j++) {
        uint8_t v = code[i+j];
        if (v == 4) { amb = true; break; }
        lmer = (lmer << 2) | v;
      }
      if (amb) { cand[i] = ~0ULL; continue; }
      lmer &= lmask;
      uint64_t rc = revcomp(lmer, l);
      uint64_t canon = lmer < rc ? lmer : rc;
      if (ssm) canon &= ssm;
      cand[i] = canon ^ tog;
    }

    int n_kmer = (int) L - k + 1; if (n_kmer < 1) n_kmer = 0;
    int win = k - l + 1;
    for (int p = 0; p < n_kmer; p++) {
      int arg = -1; uint64_t m = ~0ULL; bool ok = true;
      for (int j = 0; j < win; j++) {
        uint64_t c = cand[p+j];
        if (c == ~0ULL) { ok = false; break; }
        if (arg < 0 || c < m) { m = c; arg = p+j; }
      }
      int parg = -1; bool pok = false;
      if (p > 0) {
        uint64_t pm = ~0ULL; pok = true;
        for (int j = 0; j < win; j++) {
          uint64_t c = cand[p-1+j];
          if (c == ~0ULL) { pok = false; break; }
          if (parg < 0 || c < pm) { pm = c; parg = p-1+j; }
        }
      }
      val[p] = ok ? (m ^ tog) : 0;
      flag[p] = (ok && (!pok || arg != parg)) ? 1 : 0;
    }
    // The two rejections are independent and must not be folded into one
    // array.  A minimizer dropped by the subsampling filter still counts as
    // the previous reported minimizer for duplicate detection, because
    // ClassifySequence updates last_minimizer before it decides to skip.
    for (int p = 0; p < n_kmer; p++) {
      int keep = flag[p];
      if (keep) {
        int q = p - 1;
        while (q >= 0 && !flag[q]) q--;        // flag[] stays unmodified here
        if (q >= 0 && val[q] == val[p]) keep = 0;
        else if (mahv && murmur(val[p]) < mahv) keep = 0;
      }
      look[p] = keep;
    }
    for (int p = 0; p < n_kmer; p++) scan[p] = look[p];
    for (int off = 1; off < n_kmer; off <<= 1) {
      for (int p = 0; p < n_kmer; p++) tmp[p] = (p >= off) ? scan[p] + scan[p-off] : scan[p];
      for (int p = 0; p < n_kmer; p++) scan[p] = tmp[p];
    }
    int count = n_kmer ? scan[n_kmer-1] : 0;

    if (taxa_mode) {
      // One entry per reported minimizer.  A repeat inherits the taxon set at
      // the start of its equal-value run, and a run whose start was dropped by
      // the subsampling filter contributes zero, which is exactly what
      // last_taxon does sequentially on the CPU.
      std::vector<uint32_t> gt;
      for (int p = 0; p < n_kmer; p++) {
        if (!flag[p]) continue;
        int start = p, q = p - 1;
        while (true) {
          while (q >= 0 && !flag[q]) q--;
          if (q < 0 || val[q] != val[p]) break;
          start = q; q--;
        }
        bool filtered = mahv && murmur(val[start]) < mahv;
        gt.push_back(filtered ? 0u
                              : probe(table, cap, val[p], (int) key_bits, (int) value_bits));
      }
      if (call_mode) {
        // Distinct taxa and their counts, then Kraken 2's tree resolution.
        uint32_t tx[MAX_LEN]; uint32_t ct[MAX_LEN]; int nd = 0;
        for (size_t j = 0; j < gt.size(); j++) {
          if (!gt[j]) continue;
          int q = 0; while (q < nd && tx[q] != gt[j]) q++;
          if (q == nd) { tx[nd] = gt[j]; ct[nd] = 1; nd++; } else ct[q]++;
        }
        uint32_t call = resolve(&parent[0], tx, ct, nd);
        uint32_t ext = call ? (uint32_t) extid[call] : 0u;
        pairs++;
        char oline[8192];
        if (!fgets(oline, sizeof oline, rf)) break;
        unsigned long want = 0;
        { char *p1 = strchr(oline, '\t');
          char *p2 = p1 ? strchr(p1 + 1, '\t') : nullptr;
          if (p2) want = strtoul(p2 + 1, nullptr, 10); }
        if ((unsigned long) ext != want) {
          if (bad < 5) fprintf(stderr, "read %zu call differs: classify %lu, emul %u\n", nr, want, ext);
          bad++;
        }
      } else {
        uint32_t c; if (fread(&c,4,1,rf)!=1) break;
        bool ok = (c == gt.size());
        for (uint32_t j = 0; j < c; j++) {
          uint32_t tax; if (fread(&tax,4,1,rf)!=1) { ok=false; break; }
          pairs++;
          if (ok && gt[j] != tax) ok = false;
        }
        if (!ok) { if (bad<5) fprintf(stderr,"read %zu taxa differ (cpu %u, emul %zu)\n", nr, c, gt.size()); bad++; }
      }
      if (!fgets(line, sizeof line, f)) break;
      if (!fgets(line, sizeof line, f)) break;
      nr++;
      continue;
    }

    std::vector<uint64_t> gk(count); std::vector<uint32_t> gt(count);
    for (int p = 0; p < n_kmer; p++) {
      if (!look[p]) continue;
      int slot = scan[p] - 1;
      gk[slot] = val[p];
      gt[slot] = probe(table, cap, val[p], (int) key_bits, (int) value_bits);
    }

    uint32_t c; if (fread(&c,4,1,rf)!=1) break;
    bool ok = ((int) c == count);
    for (uint32_t j = 0; j < c; j++) {
      uint64_t key; uint32_t tax;
      if (fread(&key,8,1,rf)!=1 || fread(&tax,4,1,rf)!=1) { ok = false; break; }
      pairs++;
      if (ok && (gk[j] != key || gt[j] != tax)) ok = false;
    }
    if (!ok) { if (bad < 5) fprintf(stderr,"read %zu differs (cpu %u, emul %d)\n", nr, c, count); bad++; }

    if (!fgets(line, sizeof line, f)) break;
    if (!fgets(line, sizeof line, f)) break;
    nr++;
  }
  printf("emulated %zu reads, %zu %s: %zu mismatched\n", nr, pairs,
         call_mode ? "calls" : "entries", bad);
  munmap(map, st.st_size); close(hf);
  return bad ? 2 : 0;
}
