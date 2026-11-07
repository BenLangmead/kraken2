/*
 * Exactness test for MinimizerScanner.  It compares src/mmscanner.{h,cc}
 * against mmscanner_orig.{h,cc}, a copy of the queue-based scanner in
 * namespace kraken2_orig, output by output.
 *
 *   mmscanner_test random [TRIALS [SEED]]
 *   mmscanner_test fasta FILE...        whole sequences and subblocks
 *
 * For every output the two scanners must agree on the value (including the
 * stale values returned while an l-mer is incomplete), is_ambiguous() and
 * the sequence position reached, and they must produce the same number of
 * outputs.
 */

#include "kraken2_headers.h"
#define private public
#include "mmscanner.h"
#include "mmscanner_orig.h"
#undef private
#include <random>

using namespace std;

typedef kraken2::MinimizerScanner NewScanner;
typedef kraken2_orig::MinimizerScanner OrigScanner;

struct Params {
  ssize_t k, l;
  uint64_t seed, toggle;
  bool dna;
  int rv;
};

static string Describe(const Params &p) {
  char b[200];
  snprintf(b, sizeof b, "k=%zd l=%zd seed=%016llx toggle=%016llx %s rv=%d",
           p.k, p.l, (unsigned long long) p.seed, (unsigned long long) p.toggle,
           p.dna ? "dna" : "protein", p.rv);
  return b;
}

static uint64_t n_compared = 0;

// Scans [start, finish) with both scanners in lockstep.  Returns false at
// the first difference and prints the last few outputs before it.
static bool Compare(const Params &p, const char *seq, size_t len, size_t start,
                    size_t finish, OrigScanner &orig, NewScanner &stream,
                    const char *what) {
  orig.LoadSequence(seq, len, start, finish);
  stream.LoadSequence(seq, len, start, finish);
  struct Row { size_t idx, p1; uint64_t v1; int a1; size_t p2; uint64_t v2; int a2; };
  Row recent[4] = {};
  size_t idx = 0;
  bool ok = true;
  for (;; idx++) {
    uint64_t *m1 = orig.NextMinimizer();
    uint64_t *m2 = stream.NextMinimizer();
    if (! m1 && ! m2)
      break;
    recent[idx % 4] = { idx, m1 ? orig.str_pos_ : 0, m1 ? *m1 : 0,
                        m1 ? orig.is_ambiguous() : -1,
                        m2 ? stream.str_pos_ : 0, m2 ? *m2 : 0,
                        m2 ? stream.is_ambiguous() : -1 };
    if (! m1 || ! m2 || *m1 != *m2 || orig.str_pos_ != stream.str_pos_ ||
        orig.is_ambiguous() != stream.is_ambiguous()) {
      ok = false;
      break;
    }
  }
  n_compared += idx;
  if (! ok) {
    fprintf(stderr, "Mismatch (%s): %s len=%zu start=%zu finish=%zu\n",
            what, Describe(p).c_str(), len, start, finish);
    for (size_t i = idx >= 3 ? idx - 3 : 0; i <= idx; i++) {
      const Row &r = recent[i % 4];
      fprintf(stderr, "  [%zu] orig pos=%zu %016llx %d | stream pos=%zu %016llx %d\n",
              r.idx, r.p1, (unsigned long long) r.v1, r.a1,
              r.p2, (unsigned long long) r.v2, r.a2);
    }
    if (len < 400)
      fprintf(stderr, "  seq=%.*s\n", (int) len, seq);
  }
  return ok;
}

static string RandomSeq(mt19937_64 &rng, size_t len, bool dna) {
  static const char dna_chars[] = "ACGTacgt";
  static const char pro_chars[] = "ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy*UO";
  static const char odd[] = "NnXRYK-.\n@";
  string s(len, 'A');
  // A mix of per-sequence ambiguity rates, plus occasional runs of N.
  double amb_rate = (rng() % 4 == 0) ? 0 : pow(10, -(double) (rng() % 4) - 0.5);
  for (size_t i = 0; i < len; i++) {
    double u = (rng() >> 11) * (1.0 / 9007199254740992.0);
    if (u < amb_rate)
      s[i] = odd[rng() % (sizeof odd - 1)];
    else if (dna)
      s[i] = dna_chars[rng() % (sizeof dna_chars - 1)];
    else
      s[i] = pro_chars[rng() % (sizeof pro_chars - 1)];
  }
  int runs = rng() % 3;
  for (int r = 0; r < runs && len; r++) {
    size_t a = rng() % len, b = std::min<size_t>(len, a + 1 + rng() % 80);
    for (size_t i = a; i < b; i++)
      s[i] = 'N';
  }
  // Low-complexity stretches give ties in the window.
  if (rng() % 4 == 0 && len) {
    size_t a = rng() % len, b = std::min<size_t>(len, a + 1 + rng() % 200);
    char c = dna ? "ACGT"[rng() % 4] : 'A';
    for (size_t i = a; i < b; i++)
      s[i] = c;
  }
  return s;
}

static int Random(uint64_t trials, uint64_t seed) {
  mt19937_64 rng(seed);
  uint64_t fails = 0;
  for (uint64_t t = 0; t < trials; t++) {
    Params p;
    p.dna = rng() % 8 != 0;
    int maxl = p.dna ? 31 : 15;
    switch (rng() % 4) {
      case 0: p.l = p.dna ? 31 : 12; p.k = p.dna ? 35 : 15; break;
      case 1: p.l = 1 + rng() % maxl; p.k = p.l; break;
      case 2: p.l = 1 + rng() % maxl; p.k = p.l + rng() % 8; break;
      default: p.l = 1 + rng() % maxl; p.k = p.l + rng() % 48; break;
    }
    int bpc = p.dna ? 2 : 4;
    uint64_t lmask = (((uint64_t) 1) << (p.l * bpc)) - 1;
    switch (rng() % 3) {
      case 0: p.seed = 0; break;
      case 1: {
        // Kraken 2's spaced seed: every other position from the second
        // masked out, up to s positions.
        p.seed = lmask;
        int s = rng() % 8;
        for (int i = 0; i < s && 2 * i + 1 < p.l; i++)
          p.seed &= ~((uint64_t) ((1 << bpc) - 1) << ((2 * i + 1) * bpc));
        break;
      }
      default: p.seed = rng(); break;
    }
    p.toggle = rng() % 3 == 0 ? 0 : (rng() % 2 ? kraken2::DEFAULT_TOGGLE_MASK : rng());
    p.rv = rng() % 4 == 0 ? 0 : 1;
    OrigScanner orig(p.k, p.l, p.seed, p.dna, p.toggle, p.rv);
    NewScanner stream(p.k, p.l, p.seed, p.dna, p.toggle, p.rv);
    size_t len;
    switch (rng() % 4) {
      case 0: len = rng() % (p.k + 3); break;
      case 1: len = rng() % 200; break;
      case 2: len = rng() % 3000; break;
      default: len = rng() % 20000; break;
    }
    string s = RandomSeq(rng, len, p.dna);
    // Several intervals per sequence: whole, random, near the ends, and a
    // tiling of subblocks that overlap by k - 1 as in build_db.
    for (int rep = 0; rep < 4; rep++) {
      size_t start = 0, finish = SIZE_MAX;
      if (rep == 1 && len) {
        start = rng() % len;
        finish = start + rng() % (len - start + 5);
      }
      else if (rep == 2 && len) {
        start = rng() % (p.k + 2);
        finish = len - rng() % (std::min(len, (size_t) 3) + 1);
      }
      else if (rep == 3) {
        size_t sb = 1 + rng() % 600;
        for (size_t i = 0; i < len; i += sb)
          if (! Compare(p, s.data(), len, i, i + sb + p.k - 1, orig, stream,
                        "subblock"))
            fails++;
        continue;
      }
      if (! Compare(p, s.data(), len, start, finish, orig, stream, "random"))
        fails++;
    }
    if (fails > 10)
      break;
  }
  printf("random: %llu trials, %llu outputs compared, %llu failures\n",
         (unsigned long long) trials, (unsigned long long) n_compared,
         (unsigned long long) fails);
  return fails ? 1 : 0;
}

static vector<string> ReadFasta(const char *path, size_t &bases) {
  vector<string> seqs;
  ifstream in(path);
  if (! in)
    err(EX_NOINPUT, "%s", path);
  string line;
  while (getline(in, line)) {
    if (line.empty())
      continue;
    if (line[0] == '>') {
      seqs.emplace_back();
      continue;
    }
    if (seqs.empty())
      seqs.emplace_back();
    seqs.back() += line;
    bases += line.size();
  }
  return seqs;
}

// Kraken 2's default DNA settings, without and with the default spaced seed,
// and with the revcom_version 0 canonical form.
static int FastaTest(int nfiles, char **files) {
  const uint64_t spaced = 0x3FFFFFFFFFFFFFFFULL & ~0x0CCCCCCCCCCCCCCCULL;
  const vector<Params> ps = {
    {35, 31, 0, kraken2::DEFAULT_TOGGLE_MASK, true, 1},
    {35, 31, spaced, kraken2::DEFAULT_TOGGLE_MASK, true, 1},
    {35, 31, spaced, kraken2::DEFAULT_TOGGLE_MASK, true, 0},
  };
  uint64_t fails = 0, nseqs = 0;
  size_t bases = 0;
  for (int fi = 0; fi < nfiles; fi++) {
    vector<string> seqs = ReadFasta(files[fi], bases);
    nseqs += seqs.size();
    for (auto &p : ps) {
      OrigScanner orig(p.k, p.l, p.seed, p.dna, p.toggle, p.rv);
      NewScanner stream(p.k, p.l, p.seed, p.dna, p.toggle, p.rv);
      for (auto &s : seqs) {
        if (! Compare(p, s.data(), s.size(), 0, SIZE_MAX, orig, stream, "whole"))
          fails++;
        // build_db's subblocks: 342 bases, overlapping by k - 1.
        for (size_t i = 0; i < s.size(); i += 342)
          if (! Compare(p, s.data(), s.size(), i, i + 342 + p.k - 1, orig, stream,
                        "subblock"))
            fails++;
      }
    }
  }
  printf("fasta: %d files, %llu sequences, %zu bases, %zu settings, "
         "%llu outputs compared, %llu failures\n",
         nfiles, (unsigned long long) nseqs, bases, ps.size(),
         (unsigned long long) n_compared, (unsigned long long) fails);
  return fails ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc < 2)
    errx(EX_USAGE, "usage: %s random [TRIALS [SEED]] | fasta FILE...", argv[0]);
  string mode = argv[1];
  if (mode == "random")
    return Random(argc > 2 ? strtoull(argv[2], nullptr, 10) : 2000,
                  argc > 3 ? strtoull(argv[3], nullptr, 10) : 1);
  if (mode == "fasta" && argc > 2)
    return FastaTest(argc - 2, argv + 2);
  errx(EX_USAGE, "bad arguments");
}
