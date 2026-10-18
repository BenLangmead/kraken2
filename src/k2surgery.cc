/*
 * Copyright 2013-2023, Derrick Wood <dwood@cs.jhu.edu>
 *
 * This file is part of the Kraken 2 taxonomic sequence classification system.
 */

// Reads (minimizer, taxon) pairs back out of exact blocked tables and builds
// new tables from them, so a built table can be changed without its library.
//
//   k2surgery dump [-c] [-s] -H hash.k2d -t taxo.k2d
//     Prints one line per stored key, the minimizer in hex and its external
//     taxon ID, in table order.  -c looks every key up again and counts the
//     keys whose lookup disagrees; -s prints only the summary (on stderr).
//
//   k2surgery rebuild -H out.k2d -t out_taxo.k2d [-e load] [-P bits]
//                     [-p threads] [-d taxid]... [-m taxid:taxid]...
//                     in.k2d:in_taxo.k2d ...
//     Builds a new exact table from the keys of one or more input tables.
//     Each input's taxa are mapped to out_taxo.k2d by external ID, so the
//     inputs may come from builds with different taxonomies.  A key in more
//     than one input gets the LCA of its taxa, as if its sequences had been
//     in one library.  -d drops the keys whose stored taxon is the given
//     external ID; -m moves them to another taxon.  Both act on each input's
//     stored taxon, before inputs are merged.  -e and -P are build_db's -e
//     and -P, with the same defaults.

#include "kraken2_headers.h"
#include "blocked_table.h"
#include "taxonomy.h"

#include <map>
#include <set>

using std::string;
using std::vector;
using namespace kraken2;

static void usage() {
  fprintf(stderr,
      "Usage: k2surgery dump [-c] [-s] -H hash.k2d -t taxo.k2d\n"
      "       k2surgery rebuild -H out.k2d -t out_taxo.k2d [-e load] [-P bits]\n"
      "                 [-p threads] [-d taxid]... [-m taxid:taxid]...\n"
      "                 in.k2d:in_taxo.k2d ...\n");
  exit(EX_USAGE);
}

static int dump(int argc, char **argv) {
  string hash_file, taxo_file;
  bool check = false, summary_only = false;
  int opt;
  while ((opt = getopt(argc, argv, "csH:t:")) != -1) {
    switch (opt) {
      case 'c': check = true; break;
      case 's': summary_only = true; break;
      case 'H': hash_file = optarg; break;
      case 't': taxo_file = optarg; break;
      default: usage();
    }
  }
  if (hash_file.empty() || taxo_file.empty())
    usage();
  BlockedTable table(hash_file);
  Taxonomy tax(taxo_file);
  const TaxonomyNode *nodes = tax.nodes();
  uint64_t keys = 0, bad = 0;
  table.ForEachKey([&](uint64_t m, hvalue_t t) {
    keys++;
    if (check && table.Get(m) != t)
      bad++;
    if (! summary_only)
      printf("%016llx\t%llu\n", (unsigned long long) m,
             (unsigned long long) nodes[t].external_id);
  });
  const BlockedTableHeader &h = table.header();
  fprintf(stderr, "%llu keys (header: %llu entries, %llu in the stash)\n",
          (unsigned long long) keys, (unsigned long long) h.entries,
          (unsigned long long) h.stash_entries);
  if (check)
    fprintf(stderr, "%llu keys whose lookup disagrees\n", (unsigned long long) bad);
  return (keys != h.entries || bad) ? 1 : 0;
}

static uint64_t parse_taxid(const char *s) {
  char *end;
  uint64_t v = strtoull(s, &end, 10);
  if (*s == '\0' || *end != '\0')
    errx(EX_USAGE, "bad taxon ID %s", s);
  return v;
}

static int rebuild(int argc, char **argv) {
  string out_file, taxo_file;
  double load = 0.99;
  int partition_bits = -1, threads = 1;
  std::set<uint64_t> drop;
  std::map<uint64_t, uint64_t> move;
  int opt;
  while ((opt = getopt(argc, argv, "H:t:e:P:p:d:m:")) != -1) {
    switch (opt) {
      case 'H': out_file = optarg; break;
      case 't': taxo_file = optarg; break;
      case 'e': load = atof(optarg); break;
      case 'P': partition_bits = atoi(optarg); break;
      case 'p': threads = atoi(optarg); break;
      case 'd': drop.insert(parse_taxid(optarg)); break;
      case 'm': {
        string a = optarg;
        size_t c = a.find(':');
        if (c == string::npos)
          usage();
        move[parse_taxid(a.substr(0, c).c_str())] = parse_taxid(a.substr(c + 1).c_str());
        break;
      }
      default: usage();
    }
  }
  if (out_file.empty() || taxo_file.empty() || optind >= argc)
    usage();

  Taxonomy out_tax(taxo_file);
  out_tax.GenerateExternalToInternalIDMap();
  size_t bits_for_taxid = 1;
  while ((1ULL << bits_for_taxid) < out_tax.node_count())
    bits_for_taxid++;
  for (auto &kv : move)
    if (out_tax.GetInternalID(kv.second) == 0)
      errx(EX_DATAERR, "taxon %llu is not in %s",
           (unsigned long long) kv.second, taxo_file.c_str());

  BlockedTableBuilder *builder = nullptr;
  BlockedTableBuilder::Buffer *buffer = nullptr;
  uint64_t pext_mask = 0;
  for (int i = optind; i < argc; i++) {
    string arg = argv[i];
    size_t c = arg.find(':');
    if (c == string::npos)
      usage();
    string in_hash = arg.substr(0, c), in_taxo = arg.substr(c + 1);
    BlockedTable table(in_hash);
    Taxonomy in_tax(in_taxo);
    const BlockedTableHeader &h = table.header();
    if (! builder) {
      pext_mask = h.pext_mask;
      builder = new BlockedTableBuilder(pext_mask, partition_bits, load, out_tax);
      buffer = new BlockedTableBuilder::Buffer(*builder);
    }
    else if (h.pext_mask != pext_mask) {
      errx(EX_DATAERR, "%s was built with a different spaced seed", in_hash.c_str());
    }
    // Internal taxon of the input -> internal taxon of the output, or 0 to
    // drop the key.
    const TaxonomyNode *nodes = in_tax.nodes();
    vector<uint32_t> to_out(in_tax.node_count(), 0);
    for (size_t t = 1; t < in_tax.node_count(); t++) {
      uint64_t ext = nodes[t].external_id;
      if (drop.count(ext)) {
        continue;
      }
      if (move.count(ext))
        ext = move[ext];
      to_out[t] = (uint32_t) out_tax.GetInternalID(ext);
    }
    uint64_t keys = 0, kept = 0, moved = 0;
    table.ForEachKey([&](uint64_t m, hvalue_t t) {
      keys++;
      uint32_t o = to_out[t];
      if (! o) {
        if (! drop.count(nodes[t].external_id))
          errx(EX_DATAERR, "taxon %llu of %s is not in %s",
               (unsigned long long) nodes[t].external_id, in_hash.c_str(),
               taxo_file.c_str());
        return;
      }
      if (move.count(nodes[t].external_id))
        moved++;
      kept++;
      buffer->Add(m, o);
    });
    fprintf(stderr, "%s: %llu keys read, %llu kept (%llu moved), %llu dropped\n",
            in_hash.c_str(), (unsigned long long) keys, (unsigned long long) kept,
            (unsigned long long) moved, (unsigned long long) (keys - kept));
  }
  delete buffer;
  uint64_t n = builder->Write(out_file.c_str(), (int) bits_for_taxid, threads);
  fprintf(stderr, "wrote %llu distinct keys to %s\n", (unsigned long long) n,
          out_file.c_str());
  delete builder;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2)
    usage();
  string cmd = argv[1];
  if (cmd == "dump")
    return dump(argc - 1, argv + 1);
  if (cmd == "rebuild")
    return rebuild(argc - 1, argv + 1);
  usage();
}
