/*
 * Copyright 2026, Ben Langmead <ben.langmead@gmail.com>
 *
 * This file is part of the Kraken 2 taxonomic sequence classification system.
 */

// Computes Bracken's databaseNmers.kmer_distrib files directly from a built
// database, for several read lengths in one pass over the library.
//
// Bracken's own build (bracken-build) classifies every library sequence with
// kraken2, then kmer2read_distr slides a window of N - k + 1 k-mer positions
// along each sequence's hit list and calls each window the way Kraken 2 would
// at confidence 0, and generate_kmer_distribution.py pivots the per-sequence
// tallies into the kmer_distrib file.  This program does the same three steps
// in memory and writes byte-identical files.
//
// Taxa of k-mer positions.  Each library sequence is scanned with the
// database's own MinimizerScanner settings (k, l, spaced seed, toggle mask,
// revcom version, all from opts.k2d) and each minimizer is looked up with the
// table's own GetBatch, with classify's rules: an ambiguous k-mer has no
// taxon, a minimizer below minimum_acceptable_hash_value has none, and a
// minimizer equal to the one before it reuses that one's taxon.
//
// Window calls.  A window's call is the taxon x present in the window that
// maximizes the number of the window's positions whose taxon is x or an
// ancestor of x, with ties going to the LCA of the tied taxa.  Positions at
// the root add the same amount to every taxon's score and so never change the
// call, except that a window whose only taxon is the root is called at the
// root.  A window with no taxa at all is called 0.  This is what Bracken's
// KmerClassifier computes.
//
// Runs.  The positions' taxa are kept as runs.  As the window slides by one,
// one position leaves and one enters, and between two run boundaries the
// leaving taxon a and the entering taxon b stay the same.  Over such a stretch
// every taxon's score changes linearly, by -1, 0 or +1 per step, so the call
// is constant except at the few steps where two of those three lines cross.
// Each stretch therefore costs a constant number of call evaluations, each
// quadratic in the number of distinct taxa in the window, instead of one
// evaluation per window.
//
// Parallelism.  Sequences are cut into chunks of window start positions.  A
// chunk scans its own k-mers plus the N - 1 positions that its last windows
// reach past its end, so long chromosomes are split across threads.
//
// Output order.  generate_kmer_distribution.py writes taxa and genomes in the
// order its dictionaries first saw them, which follows the line order of
// kmer2read_distr's output.  With one kmer2read_distr thread that is the
// order of the library sequences, and this program writes that order.  With
// several threads Bracken's order depends on scheduling; the content is the
// same.

#include "kraken2_headers.h"
#include "kv_store.h"
#include "omp_hack.h"
#include "taxonomy.h"
#include "fast_reader.h"
#include "mmscanner.h"
#include "compact_hash.h"
#include "blocked_table.h"
#include "kraken2_data.h"
#include "utilities.h"

#include <chrono>
#include <fcntl.h>

using std::string;
using std::vector;
using std::pair;
using namespace kraken2;

namespace {

typedef uint32_t tax_t;
const tax_t ROOT = 1;

struct Options {
  string index_filename, taxonomy_filename, options_filename;
  string seqid_map_filename, output_prefix;
  vector<int> read_lengths;
  vector<string> inputs;
  int threads = 1;
  size_t chunk = 1 << 20;            // window start positions per chunk
  size_t block_bytes = 64 << 20;     // input bytes per block
  bool naive = false;                // score every window from scratch
  bool memory_mapping = false;
};

double Now() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Preorder intervals: y is x or an ancestor of x iff
// pre[y] <= pre[x] < end[y].
struct Tree {
  const Taxonomy &tax;
  vector<uint32_t> pre, end;

  explicit Tree(const Taxonomy &t) : tax(t) {
    size_t n = t.node_count();
    pre.assign(n, 0);
    end.assign(n, 0);
    if (n < 2)
      return;
    const TaxonomyNode *nodes = t.nodes();
    uint32_t clock = 0;
    vector<pair<uint64_t, uint64_t> > stack;  // node, next child offset
    pre[ROOT] = clock++;
    stack.push_back({ROOT, 0});
    while (! stack.empty()) {
      auto &top = stack.back();
      const TaxonomyNode &nd = nodes[top.first];
      if (top.second < nd.child_count) {
        uint64_t c = nd.first_child + top.second++;
        pre[c] = clock++;
        stack.push_back({c, 0});
      }
      else {
        end[top.first] = clock;
        stack.pop_back();
      }
    }
  }
  inline bool AncOrSelf(tax_t y, tax_t x) const {
    return pre[y] <= pre[x] && pre[x] < end[y];
  }
  inline tax_t Lca(tax_t a, tax_t b) const {
    return (tax_t) tax.LowestCommonAncestor(a, b);
  }
};

// Window calls tallied by internal taxon.  A chunk sees few distinct calls, so
// a short vector searched from the last hit is enough.
struct Tally {
  vector<pair<tax_t, uint64_t> > v;
  size_t last = 0;
  inline void Add(tax_t t, uint64_t n) {
    if (! n)
      return;
    if (last < v.size() && v[last].first == t) {
      v[last].second += n;
      return;
    }
    for (size_t i = 0; i < v.size(); i++)
      if (v[i].first == t) {
        v[i].second += n;
        last = i;
        return;
      }
    last = v.size();
    v.push_back({t, n});
  }
};

// Taxa present in the window with their position counts.  Root and 0 are not
// kept as entries: root's count is tracked apart, and 0 is ignored.
struct Window {
  const Tree &tree;
  vector<pair<tax_t, int64_t> > act;
  int64_t rootc = 0;
  // Scratch for the stretch evaluation.
  vector<int64_t> score;
  vector<int> slope;

  explicit Window(const Tree &t) : tree(t) { }

  void Clear() { act.clear(); rootc = 0; }

  inline void Add(tax_t t, int64_t n) {
    if (t == 0)
      return;
    if (t == ROOT) {
      rootc += n;
      return;
    }
    for (auto &e : act)
      if (e.first == t) {
        e.second += n;
        return;
      }
    act.push_back({t, n});
  }
  inline void Sub(tax_t t, int64_t n) {
    if (t == 0)
      return;
    if (t == ROOT) {
      rootc -= n;
      return;
    }
    for (size_t i = 0; i < act.size(); i++)
      if (act[i].first == t) {
        act[i].second -= n;
        if (act[i].second == 0) {
          act[i] = act.back();
          act.pop_back();
        }
        return;
      }
  }

  void Scores() {
    size_t n = act.size();
    score.assign(n, 0);
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < n; j++)
        if (tree.AncOrSelf(act[j].first, act[i].first))
          score[i] += act[j].second;
  }

  tax_t Call() {
    if (act.empty())
      return rootc > 0 ? ROOT : 0;
    Scores();
    int64_t best = 0;
    tax_t call = 0;
    for (size_t i = 0; i < act.size(); i++) {
      if (score[i] > best) {
        best = score[i];
        call = act[i].first;
      }
      else if (score[i] == best) {
        call = tree.Lca(call, act[i].first);
      }
    }
    return call;
  }

  // Tallies the calls of the next J windows, the j-th of which (j = 1..J)
  // has a's count lowered by j and b's raised by j from the current state.
  // The caller guarantees that b is already present and that a stays present
  // throughout, so the set of taxa does not change.  Leaves the state as is.
  void Stretch(uint64_t J, tax_t a, tax_t b, Tally &tally) {
    bool a_act = a > ROOT, b_act = b > ROOT;
    if (act.empty()) {
      // Neither a nor b is a kept taxon; only the root count can move.
      int rs = (b == ROOT) - (a == ROOT);
      if (rs >= 0) {
        tally.Add(rootc + rs > 0 ? ROOT : 0, J);
      }
      else {
        uint64_t n_root = rootc - 1 > 0 ? std::min<uint64_t>(J, rootc - 1) : 0;
        tally.Add(ROOT, n_root);
        tally.Add(0, J - n_root);
      }
      return;
    }
    if (! a_act && ! b_act) {
      tally.Add(Call(), J);
      return;
    }
    Scores();
    size_t n = act.size();
    // Per slope class -1, 0, +1: the best score and the LCA of its holders.
    int64_t M[3];
    tax_t L[3];
    bool has[3] = {false, false, false};
    for (size_t i = 0; i < n; i++) {
      tax_t x = act[i].first;
      int s = (b_act && tree.AncOrSelf(b, x)) - (a_act && tree.AncOrSelf(a, x));
      int c = s + 1;
      if (! has[c] || score[i] > M[c]) {
        has[c] = true;
        M[c] = score[i];
        L[c] = x;
      }
      else if (score[i] == M[c]) {
        L[c] = tree.Lca(L[c], x);
      }
    }
    // Steps where the order of two class lines can change.
    int64_t pts[8];
    int np = 0;
    pts[np++] = 1;
    for (int c1 = 2; c1 >= 0; c1--)
      for (int c2 = c1 - 1; c2 >= 0; c2--) {
        if (! has[c1] || ! has[c2])
          continue;
        // M[c1] + (c1-1) j == M[c2] + (c2-1) j  at  j = (M[c2]-M[c1]) / (c1-c2)
        int64_t num = M[c2] - M[c1], den = c1 - c2;
        int64_t q = num >= 0 ? num / den : -((-num + den - 1) / den);  // floor
        if (q * den == num) {
          pts[np++] = q;
          pts[np++] = q + 1;
        }
        else {
          pts[np++] = q + 1;  // ceil
        }
      }
    int64_t jmax = (int64_t) J;
    int m = 0;
    for (int i = 0; i < np; i++)
      if (pts[i] >= 1 && pts[i] <= jmax)
        pts[m++] = pts[i];
    std::sort(pts, pts + m);
    m = std::unique(pts, pts + m) - pts;
    for (int i = 0; i < m; i++) {
      int64_t j = pts[i];
      int64_t next = i + 1 < m ? pts[i + 1] : jmax + 1;
      int64_t best = INT64_MIN;
      tax_t call = 0;
      for (int c = 0; c < 3; c++) {
        if (! has[c])
          continue;
        int64_t v = M[c] + (int64_t) (c - 1) * j;
        if (v > best) {
          best = v;
          call = L[c];
        }
        else if (v == best) {
          call = tree.Lca(call, L[c]);
        }
      }
      tally.Add(call, next - j);
    }
  }
};

// The positions' taxa as runs: run i has taxon taxon[i] and ends just before
// position end[i].
struct Runs {
  vector<tax_t> taxon;
  vector<size_t> end;
  void Clear() { taxon.clear(); end.clear(); }
  inline void Push(tax_t t, size_t p) {  // position p has taxon t
    if (! taxon.empty() && taxon.back() == t)
      end.back() = p + 1;
    else {
      taxon.push_back(t);
      end.push_back(p + 1);
    }
  }
};

// Tallies the calls of windows starting at 0 .. nwin-1, with W positions per
// window.  The runs cover at least nwin + W - 1 positions.
void ScoreWindows(const Runs &runs, size_t nwin, size_t W, Window &win,
                  Tally &tally, bool naive) {
  win.Clear();
  size_t ri = 0;
  for (size_t p = 0; p < W; ri++) {
    size_t e = std::min(runs.end[ri], W);
    win.Add(runs.taxon[ri], e - p);
    p = e;
  }
  if (naive) {
    vector<tax_t> t;
    for (size_t i = 0, p = 0; p < nwin + W - 1; i++)
      for (; p < runs.end[i]; p++)
        t.push_back(runs.taxon[i]);
    tally.Add(win.Call(), 1);
    for (size_t s = 1; s < nwin; s++) {
      win.Sub(t[s - 1], 1);
      win.Add(t[s + W - 1], 1);
      tally.Add(win.Call(), 1);
    }
    return;
  }
  // li is the run holding the next position to leave, s; ei the run holding
  // the next to enter, s + W.
  size_t li = 0, ei = 0;
  tax_t cached = win.Call();
  tally.Add(cached, 1);
  size_t done = 1, s = 0;
  while (done < nwin) {
    while (runs.end[li] <= s)
      li++;
    while (runs.end[ei] <= s + W)
      ei++;
    tax_t a = runs.taxon[li], b = runs.taxon[ei];
    size_t rl = runs.end[li] - s;
    size_t re = runs.end[ei] - (s + W);
    size_t m = std::min(std::min(rl, re), nwin - done);
    if (a == b) {
      tally.Add(cached, m);
    }
    else {
      win.Sub(a, 1);
      win.Add(b, 1);
      cached = win.Call();
      tally.Add(cached, 1);
      if (m >= 3) {
        win.Stretch(m - 2, a, b, tally);
        win.Sub(a, m - 2);
        win.Add(b, m - 2);
      }
      if (m >= 2) {
        win.Sub(a, 1);
        win.Add(b, 1);
        cached = win.Call();
        tally.Add(cached, 1);
      }
    }
    s += m;
    done += m;
  }
}

void Usage(int code) {
  std::cerr <<
    "Usage: bracken_distrib [options] FASTA...\n"
    "Writes PREFIX<N>mers.kmer_distrib for each read length N.\n"
    "  -H FILE   hash.k2d\n"
    "  -t FILE   taxo.k2d\n"
    "  -o FILE   opts.k2d\n"
    "  -m FILE   seqid2taxid.map\n"
    "  -O PREFIX output prefix, for example DB/database\n"
    "  -r LIST   read lengths, comma-separated (default 100)\n"
    "  -p N      threads (default 1)\n"
    "  -c N      window start positions per chunk (default 1048576)\n"
    "  -B N      input block bytes (default 67108864)\n"
    "  -M        memory-map the table\n"
    "  -X        score every window from scratch (testing)\n";
  exit(code);
}

void ParseCommandLine(int argc, char **argv, Options &o) {
  int c;
  string lengths = "100";
  while ((c = getopt(argc, argv, "H:t:o:m:O:r:p:c:B:MXh")) != -1) {
    switch (c) {
      case 'H': o.index_filename = optarg; break;
      case 't': o.taxonomy_filename = optarg; break;
      case 'o': o.options_filename = optarg; break;
      case 'm': o.seqid_map_filename = optarg; break;
      case 'O': o.output_prefix = optarg; break;
      case 'r': lengths = optarg; break;
      case 'p': o.threads = atoi(optarg); break;
      case 'c': o.chunk = strtoull(optarg, nullptr, 10); break;
      case 'B': o.block_bytes = strtoull(optarg, nullptr, 10); break;
      case 'M': o.memory_mapping = true; break;
      case 'X': o.naive = true; break;
      case 'h': Usage(0); break;
      default: Usage(EX_USAGE);
    }
  }
  for (int i = optind; i < argc; i++)
    o.inputs.push_back(argv[i]);
  std::stringstream ss(lengths);
  string tok;
  while (std::getline(ss, tok, ','))
    o.read_lengths.push_back(atoi(tok.c_str()));
  if (o.index_filename.empty() || o.taxonomy_filename.empty() ||
      o.options_filename.empty() || o.seqid_map_filename.empty() ||
      o.output_prefix.empty() || o.inputs.empty() || o.read_lengths.empty())
    Usage(EX_USAGE);
  if (o.threads < 1 || o.chunk < 1)
    Usage(EX_USAGE);
}

KeyValueStore *LoadTable(string &filename, const IndexOptions &idx, bool mm) {
  switch (GetKVStoreCellType(filename)) {
    case CompactHash32:
      return new CompactHashTable<CompactHashCell>(filename, mm);
    case CompactHash40:
      return new CompactHashTable<CompactHashCell40>(filename, mm);
    case BlockedExact:
      return new BlockedTable(filename, mm);
    default:
      errx(EX_DATAERR, "unable to determine the table type of %s",
           filename.c_str());
  }
}

// Bracken's seqid2taxid reader: tab-separated, first entry for a seqid wins,
// taxid parsed with atoi.
void ReadSeqidMap(const string &filename,
                  std::unordered_map<string, int64_t> &m) {
  std::ifstream in(filename);
  if (! in)
    err(EX_NOINPUT, "unable to open %s", filename.c_str());
  string line;
  while (std::getline(in, line)) {
    size_t tab = line.find('\t');
    string id = line.substr(0, tab);
    int64_t taxid = tab == string::npos ? 0 : atoi(line.c_str() + tab + 1);
    m.emplace(id, taxid);
  }
}

// One genome's tallies by mapped taxon, in the order generate_kmer_distribution
// first saw each mapped taxon.
struct GenomeDist {
  vector<pair<uint64_t, uint64_t> > mapped;  // external mapped taxid, windows
  std::unordered_map<uint64_t, size_t> where;
  uint64_t total = 0;
};

struct LengthDist {
  vector<int64_t> genome_order;
  std::unordered_map<int64_t, GenomeDist> genomes;

  // Folds one sequence's tallies, sorted by external mapped taxid, the order
  // kmer2read_distr prints them.
  void Fold(int64_t genome, const vector<pair<uint64_t, uint64_t> > &seq) {
    if (seq.empty())
      return;
    auto it = genomes.find(genome);
    if (it == genomes.end()) {
      genome_order.push_back(genome);
      it = genomes.emplace(genome, GenomeDist()).first;
    }
    GenomeDist &g = it->second;
    for (auto &e : seq) {
      g.total += e.second;
      auto w = g.where.find(e.first);
      if (w == g.where.end()) {
        g.where.emplace(e.first, g.mapped.size());
        g.mapped.push_back(e);
      }
      else {
        g.mapped[w->second].second += e.second;
      }
    }
  }

  void Write(const string &filename) const {
    FILE *out = fopen(filename.c_str(), "w");
    if (! out)
      err(EX_CANTCREAT, "unable to open %s", filename.c_str());
    vector<uint64_t> order;
    std::unordered_map<uint64_t, size_t> index;
    vector<string> lines;
    char buf[96];
    for (int64_t gid : genome_order) {
      const GenomeDist &g = genomes.at(gid);
      for (auto &e : g.mapped) {
        auto w = index.find(e.first);
        if (w == index.end()) {
          w = index.emplace(e.first, order.size()).first;
          order.push_back(e.first);
          lines.emplace_back();
        }
        int n = snprintf(buf, sizeof(buf), "%lld:%llu:%llu ", (long long) gid,
                         (unsigned long long) e.second,
                         (unsigned long long) g.total);
        lines[w->second].append(buf, n);
      }
    }
    fputs("mapped_taxid\tgenome_taxids:kmers_mapped:total_genome_kmers\n", out);
    for (size_t i = 0; i < order.size(); i++) {
      fprintf(out, "%llu\t", (unsigned long long) order[i]);
      fwrite(lines[i].data(), 1, lines[i].size(), out);
      fputc('\n', out);
    }
    if (fclose(out) != 0)
      err(EX_IOERR, "error writing %s", filename.c_str());
  }
};

struct Unit {
  size_t seq;       // index into the block's sequences
  size_t c0, c1;    // window start positions [c0, c1)
};

struct ThreadTimes {
  double scan = 0, score = 0;
  uint64_t positions = 0, lookups = 0, runs = 0;
  char pad[64];
};

}  // namespace

int main(int argc, char **argv) {
  Options opts;
  ParseCommandLine(argc, argv, opts);
  omp_set_num_threads(opts.threads);
  double t_start = Now();

  IndexOptions idx = {0};
  {
    std::ifstream in(opts.options_filename);
    struct stat sb;
    if (! in || stat(opts.options_filename.c_str(), &sb) < 0)
      err(EX_NOINPUT, "unable to read %s", opts.options_filename.c_str());
    in.read((char *) &idx, std::min<size_t>(sb.st_size, sizeof(idx)));
  }
  if (! idx.dna_db)
    errx(EX_DATAERR, "only nucleotide databases are supported");
  Taxonomy taxonomy(opts.taxonomy_filename, opts.memory_mapping);
  if (taxonomy.node_count() < 2 || taxonomy.nodes()[ROOT].external_id != 1)
    errx(EX_DATAERR, "expected the taxonomy's root to have external ID 1");
  Tree tree(taxonomy);
  KeyValueStore *hash = LoadTable(opts.index_filename, idx, opts.memory_mapping);
  std::unordered_map<string, int64_t> seqid_map;
  ReadSeqidMap(opts.seqid_map_filename, seqid_map);
  double t_loaded = Now();

  const size_t k = idx.k;
  vector<size_t> Ws;
  for (int N : opts.read_lengths) {
    if (N < (int) k)
      errx(EX_USAGE, "read length %d is shorter than k = %zu", N, k);
    Ws.push_back(N - k + 1);
  }
  const size_t nlen = Ws.size();
  const size_t Wmax = *std::max_element(Ws.begin(), Ws.end());
  const size_t Wmin = *std::min_element(Ws.begin(), Ws.end());
  vector<LengthDist> dists(nlen);

  vector<ThreadTimes> times(opts.threads);
  uint64_t n_seqs = 0, n_bases = 0, n_unmapped = 0;
  for (const string &path : opts.inputs) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
      err(EX_NOINPUT, "unable to open %s", path.c_str());
    StreamCursor cursor;
    bool have = PrimeStream(fd, cursor);
    FastReader reader;
    while (have && reader.LoadBlock(fd, cursor, opts.block_bytes)) {
      reader.Parse();
      size_t nseq = reader.size();
      vector<int64_t> genome(nseq, -1);
      vector<Unit> units;
      for (size_t i = 0; i < nseq; i++) {
        const SeqView &sv = reader[i];
        n_seqs++;
        n_bases += sv.seq_len;
        auto it = seqid_map.find(string(sv.header, sv.header_len));
        if (it == seqid_map.end()) {
          n_unmapped++;
          continue;
        }
        genome[i] = it->second;
        if (sv.seq_len < k)
          continue;
        size_t P = sv.seq_len - k + 1;
        if (P < Wmin)
          continue;
        size_t nstart = P - Wmin + 1;
        for (size_t c0 = 0; c0 < nstart; c0 += opts.chunk)
          units.push_back({i, c0, std::min(nstart, c0 + opts.chunk)});
      }
      vector<vector<Tally> > results(units.size(), vector<Tally>(nlen));

      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        ThreadTimes &tt = times[tid];
        MinimizerScanner scanner(idx.k, idx.l, idx.spaced_seed_mask,
                                 idx.dna_db, idx.toggle_mask,
                                 idx.revcom_version);
        Window win(tree);
        Runs runs;
        vector<uint64_t> keys;
        vector<hvalue_t> vals;
        vector<int64_t> slot;
        #pragma omp for schedule(dynamic, 1)
        for (size_t u = 0; u < units.size(); u++) {
          const Unit &un = units[u];
          const SeqView &sv = reader[un.seq];
          size_t P = sv.seq_len - k + 1;
          size_t pos_end = std::min(P, un.c1 + Wmax - 1);
          size_t npos = pos_end - un.c0;
          double t0 = Now();
          // Positions c0 .. pos_end-1, scanned as if the sequence began at
          // c0, which gives each k-mer the same minimizer and ambiguity.
          scanner.LoadSequence(sv.seq + un.c0, npos + k - 1);
          keys.clear();
          slot.clear();
          uint64_t *mp;
          uint64_t last_min = UINT64_MAX;
          int64_t last_slot = -1;
          while ((mp = scanner.NextMinimizer()) != nullptr) {
            if (scanner.is_ambiguous()) {
              slot.push_back(-1);
              continue;
            }
            if (*mp != last_min) {
              last_min = *mp;
              if (idx.minimum_acceptable_hash_value &&
                  MurmurHash3(*mp) < idx.minimum_acceptable_hash_value) {
                last_slot = -1;
              }
              else {
                last_slot = keys.size();
                keys.push_back(*mp);
              }
            }
            slot.push_back(last_slot);
          }
          if (slot.size() != npos)
            errx(EX_SOFTWARE, "scanner gave %zu k-mers for %zu positions",
                 slot.size(), npos);
          vals.resize(keys.size());
          if (! keys.empty())
            hash->GetBatch(keys.data(), vals.data(), keys.size());
          runs.Clear();
          for (size_t p = 0; p < npos; p++)
            runs.Push(slot[p] < 0 ? 0 : vals[slot[p]], p);
          double t1 = Now();
          tt.scan += t1 - t0;
          tt.positions += npos;
          tt.lookups += keys.size();
          tt.runs += runs.taxon.size();
          for (size_t li = 0; li < nlen; li++) {
            size_t W = Ws[li];
            if (P < W)
              continue;
            size_t last_start = P - W + 1;  // exclusive
            if (un.c0 >= last_start)
              continue;
            size_t nwin = std::min(un.c1, last_start) - un.c0;
            ScoreWindows(runs, nwin, W, win, results[u][li], opts.naive);
          }
          tt.score += Now() - t1;
        }
      }

      // Merge each sequence's units, then fold sequences in input order.
      size_t u = 0;
      for (size_t i = 0; i < nseq; i++) {
        vector<std::map<uint64_t, uint64_t> > seq(nlen);
        for (; u < units.size() && units[u].seq == i; u++)
          for (size_t li = 0; li < nlen; li++)
            for (auto &e : results[u][li].v)
              seq[li][e.first ? taxonomy.nodes()[e.first].external_id : 0] += e.second;
        if (genome[i] < 0)
          continue;
        for (size_t li = 0; li < nlen; li++) {
          vector<pair<uint64_t, uint64_t> > v(seq[li].begin(), seq[li].end());
          dists[li].Fold(genome[i], v);
        }
      }
    }
    if (! cursor.error.empty())
      errx(EX_IOERR, "%s (%s)", cursor.error.c_str(), path.c_str());
    close(fd);
  }
  double t_scored = Now();

  for (size_t li = 0; li < nlen; li++)
    dists[li].Write(opts.output_prefix + std::to_string(opts.read_lengths[li]) +
                    "mers.kmer_distrib");
  double t_end = Now();

  ThreadTimes sum;
  for (auto &tt : times) {
    sum.scan += tt.scan;
    sum.score += tt.score;
    sum.positions += tt.positions;
    sum.lookups += tt.lookups;
    sum.runs += tt.runs;
  }
  if (n_unmapped)
    warnx("%llu sequences have no seqid2taxid entry and were skipped",
          (unsigned long long) n_unmapped);
  fprintf(stderr,
          "bracken_distrib: %llu sequences, %llu bp, %llu positions, "
          "%llu lookups, %llu runs, %zu read lengths\n"
          "  load %.3f s, scan+score %.3f s wall (thread sums: scan+lookup "
          "%.3f s, window scoring %.3f s), write %.3f s, total %.3f s\n",
          (unsigned long long) n_seqs, (unsigned long long) n_bases,
          (unsigned long long) sum.positions, (unsigned long long) sum.lookups,
          (unsigned long long) sum.runs, nlen,
          t_loaded - t_start, t_scored - t_loaded, sum.scan, sum.score,
          t_end - t_scored, t_end - t_start);
  delete hash;
  return 0;
}
