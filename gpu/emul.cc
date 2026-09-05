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
  if (argc < 5) {
    fprintf(stderr, "usage: emul reads.fq hash.k2d opts.k2d ref.k2mm [reads]\n");
    return 1;
  }
  size_t limit = argc > 5 ? (size_t) atoll(argv[5]) : 50000;

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

  FILE *f = fopen(argv[1], "rb");
  FILE *rf = fopen(argv[4], "rb");
  char magic[8]; size_t z = fread(magic,1,8,rf); (void) z;
  char flags; z = fread(&flags,1,1,rf);
  uint64_t rn; z = fread(&rn,8,1,rf);
  if (flags != 2) { fprintf(stderr, "oracle not built with -H\n"); return 1; }

  char line[8192];
  size_t nr = 0, bad = 0, pairs = 0;
  std::vector<uint64_t> cand(MAX_LEN), val(MAX_LEN);
  std::vector<int> flag(MAX_LEN), look(MAX_LEN), scan(MAX_LEN), tmp(MAX_LEN);
  std::vector<uint8_t> code(MAX_LEN);

  while (nr < limit && nr < rn && fgets(line, sizeof line, f)) {
    if (!fgets(line, sizeof line, f)) break;
    size_t L = strlen(line); while (L && (line[L-1]=='\n'||line[L-1]=='\r')) L--;
    if ((int) L > MAX_LEN) L = MAX_LEN;
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
  printf("emulated %zu reads, %zu lookups: %zu mismatched\n", nr, pairs, bad);
  munmap(map, st.st_size); close(hf);
  return bad ? 2 : 0;
}
