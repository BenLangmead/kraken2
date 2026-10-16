// Host emulation of the fused device kernel, checked line by line against
// classify's own output.
//
// GPU time is the scarce resource, so everything in the kernel that is not
// CUDA-specific gets verified here first: the l-mer construction, one token
// per k-mer for a mate without ambiguous bases, one lookup per run of equal
// minimizers, the sequential scanner for a mate with an ambiguous base, the
// mate border, tree resolution and the formatting of the line.  The scanner,
// resolution, formatting and table lookup are the same code the kernel runs
// (classify_core.h, table.h); the per-k-mer logic is restated here in the
// kernel's own terms.  What is left untested is only what needs a device:
// shared memory, the barriers, and the launch itself.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "table.h"
#include "classify_core.h"

static const int MAX_LEN = 320;

// One FASTQ record: the identifier (past '@', up to whitespace) and bases.
static bool read_record(FILE *f, std::string &id, std::string &seq) {
  char line[8192];
  if (!fgets(line, sizeof line, f)) return false;
  const char *h = line[0] == '@' ? line + 1 : line;
  size_t e = 0;
  while (h[e] && h[e] != ' ' && h[e] != '\t' && h[e] != '\n' && h[e] != '\r') e++;
  id.assign(h, e);
  if (!fgets(line, sizeof line, f)) return false;
  size_t L = strlen(line);
  while (L && (line[L-1] == '\n' || line[L-1] == '\r')) L--;
  seq.assign(line, L);
  if (!fgets(line, sizeof line, f) || !fgets(line, sizeof line, f)) return false;
  return true;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: emul reads.fq hash.k2d opts.k2d taxo.k2d classify.out [units] [mate2.fq]\n"
                    "  compares every output line with classify's; with mate2.fq the\n"
                    "  reads are classified as pairs\n");
    return 1;
  }
  size_t limit = argc > 6 ? (size_t) atoll(argv[6]) : 50000;
  bool paired = argc > 7;

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

  // Either table, chosen by magic number, through the same k2t::Table code
  // the kernel uses.
  k2t::Table table{};
  uint64_t tmagic; memcpy(&tmagic, map, 8);
  if (tmagic == k2t::BLOCKED_MAGIC) {
    k2t::BlockedHeader bh;
    if (const char *e = k2t::map_blocked(map, st.st_size, table, bh)) {
      fprintf(stderr, "%s: %s\n", argv[2], e); return 1;
    }
    printf("k=%d l=%d blocked: version %u, %" PRIu64 " partitions of %" PRIu64
           " buckets, r=%d v=%d\n", k, l, bh.version, table.G.P, table.G.nb, table.G.r, table.G.v);
  } else {
    uint64_t hdr[4]; memcpy(hdr, map, sizeof hdr);
    table.kind = k2t::Table::CHT;
    table.capacity = hdr[0]; table.key_bits = (int) hdr[2]; table.value_bits = (int) hdr[3];
    table.cells = (const uint32_t *) (map + 32);
    printf("k=%d l=%d compact hash: capacity=%" PRIu64 " key_bits=%d value_bits=%d\n",
           k, l, table.capacity, table.key_bits, table.value_bits);
  }

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

  FILE *f[2] = {fopen(argv[1], "rb"), paired ? fopen(argv[7], "rb") : nullptr};
  if (!f[0] || (paired && !f[1])) { fprintf(stderr, "cannot open reads\n"); return 1; }
  FILE *rf = fopen(argv[5], "rb");
  if (!rf) { fprintf(stderr, "cannot open oracle %s\n", argv[5]); return 1; }

  const int mates = paired ? 2 : 1;
  std::vector<uint64_t> cand(MAX_LEN), val(MAX_LEN);
  std::vector<int> look(MAX_LEN);
  std::vector<uint32_t> ptax(MAX_LEN), stax(2 * MAX_LEN + 1);
  size_t nu = 0, bad = 0, n_amb = 0;
  std::string id[2], seq[2];
  std::vector<char> buf(16384);
  char oline[16384];

  while (nu < limit) {
    bool ok = true;
    for (int m = 0; m < mates; m++) ok = ok && read_record(f[m], id[m], seq[m]);
    if (!ok) break;
    int n_tok = 0;
    uint32_t mlen[2] = {0, 0};
    for (int m = 0; m < mates; m++) {
      int len = (int) seq[m].size();
      mlen[m] = (uint32_t) len;
      if (len > MAX_LEN) { fprintf(stderr, "read %zu is longer than %d\n", nu, MAX_LEN); return 1; }
      const char *s = seq[m].data();
      bool amb = false;
      for (int i = 0; i < len; i++) if (k2c::base_code(s[i]) < 0) amb = true;
      if (amb) {
        n_amb++;
        k2c::Scanner sc;
        sc.init(k, l, ssm, tog);
        sc.load(s, len);
        uint64_t mz, last_min = ~0ULL; bool a;
        uint32_t last_taxon = 0;
        while (sc.next(&mz, &a)) {
          uint32_t taxon;
          if (a) taxon = k2c::AMBIG;
          else if (mz != last_min) {
            last_min = mz;
            taxon = (mahv && k2t::murmur(mz) < mahv) ? 0u : table.get(mz);
            last_taxon = taxon;
          } else taxon = last_taxon;
          stax[n_tok++] = taxon;
        }
      } else {
        int n_lmer = len - l + 1; if (n_lmer < 1) n_lmer = 0;
        const uint64_t lmask = (l * 2 >= 64) ? ~0ULL : ((1ULL << (l * 2)) - 1);
        for (int i = 0; i < n_lmer; i++) {
          uint64_t lmer = 0;
          for (int j = 0; j < l; j++) lmer = (lmer << 2) | (uint64_t) k2c::base_code(s[i + j]);
          lmer &= lmask;
          uint64_t rc = k2c::revcomp(lmer, l);
          uint64_t canon = lmer < rc ? lmer : rc;
          if (ssm) canon &= ssm;
          cand[i] = canon ^ tog;
        }
        int n_kmer = len - k + 1; if (n_kmer < 1) n_kmer = 0;
        int win = k - l + 1;
        for (int p = 0; p < n_kmer; p++) {
          uint64_t mn = ~0ULL;
          for (int j = 0; j < win; j++) if (cand[p + j] < mn) mn = cand[p + j];
          val[p] = mn ^ tog;
        }
        for (int p = 0; p < n_kmer; p++) {
          int keep = (p == 0 || val[p - 1] != val[p]);
          if (keep && mahv && k2t::murmur(val[p]) < mahv) keep = 0;
          look[p] = keep;
          if (keep) ptax[p] = table.get(val[p]);
        }
        for (int p = 0; p < n_kmer; p++) {
          int sx = p;
          while (sx > 0 && val[sx - 1] == val[p]) sx--;
          stax[n_tok + p] = look[sx] ? ptax[sx] : 0u;
        }
        n_tok += n_kmer;
      }
      if (paired && m == 0) stax[n_tok++] = k2c::MATE;
    }
    uint32_t call = k2c::call_of(parent.data(), stax.data(), n_tok);
    bool over = false;
    int n = k2c::format_line(buf.data(), buf.data() + buf.size() - 64, call, extid.data(),
                             id[0].data(), (uint32_t) id[0].size(), paired, mlen[0], mlen[1],
                             stax.data(), n_tok, &over);
    if (!fgets(oline, sizeof oline, rf)) { fprintf(stderr, "oracle ended early\n"); return 1; }
    if (over || strlen(oline) != (size_t) n || memcmp(oline, buf.data(), n) != 0) {
      if (bad < 3) fprintf(stderr, "unit %zu differs\n  classify: %s  emul:     %.*s", nu, oline, n, buf.data());
      bad++;
    }
    nu++;
  }
  printf("emulated %zu %s (%zu mates with an ambiguous base): %zu lines differ\n",
         nu, paired ? "pairs" : "reads", n_amb, bad);
  return bad ? 2 : 0;
}
