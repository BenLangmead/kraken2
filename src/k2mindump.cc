/*
 * k2mindump -- dump the minimizers a reference CPU scan produces, as an oracle
 * for checking a device implementation against.
 *
 * The bar for the GPU port is not that intermediate values match bit for bit,
 * it is that the same classification decisions come out.  But the minimizer
 * stream is upstream of every decision, so if it agrees exactly then any later
 * divergence is attributable to the hash table rather than the scan, and the
 * two can be debugged separately.  That is worth having as a separate artifact.
 *
 * Emits, per read, the sequence of minimizers MinimizerScanner reports, in
 * order.  With -H it emits instead the (minimizer, taxon) pairs that survive
 * consecutive-duplicate removal and the minimum_acceptable_hash_value filter,
 * and were therefore actually looked up.  That form exercises the whole front
 * of the pipeline at once: the scan, the subsampling filter, and the table
 * probe, which is what a fused device implementation has to reproduce.
 */

#include "kraken2_headers.h"
#include "kraken2_data.h"
#include "mmscanner.h"
#include "fast_reader.h"
#include "compact_hash.h"
#include "kv_store.h"
#include <fcntl.h>
#include <unistd.h>

using namespace kraken2;

// TAXID_MAX - 2, narrowed to the 32-bit field this dump uses.
static const uint32_t AMBIGUOUS_SPAN_TAXON_32 = 0xFFFFFFFDu;

static const char DUMP_MAGIC[8] = {'K','2','M','D','U','M','P','\1'};

int main(int argc, char **argv) {
  const char *opts_file = nullptr, *out_file = nullptr, *hash_file = nullptr;
  bool per_position = false, want_taxa = false;
  size_t limit = SIZE_MAX;
  int c;
  while ((c = getopt(argc, argv, "o:d:n:pH:T")) != -1) {
    switch (c) {
      case 'o': opts_file = optarg; break;
      case 'd': out_file = optarg; break;
      case 'n': limit = (size_t) atoll(optarg); break;
      case 'p': per_position = true; break;
      case 'H': hash_file = optarg; break;
      case 'T': want_taxa = true; break;
      default:
        std::cerr << "usage: k2mindump -o opts.k2d -d out.k2mm [-n reads] [-p] in.fq"
                  << std::endl;
        return 1;
    }
  }
  if (! opts_file || ! out_file || optind >= argc)
    errx(EX_USAGE, "-o, -d and an input file are all required");
  if (want_taxa && ! hash_file)
    errx(EX_USAGE, "-T needs -H, since the taxa come from the table");

  IndexOptions idx;
  memset(&idx, 0, sizeof idx);
  int ofd = open(opts_file, O_RDONLY);
  if (ofd < 0 || read(ofd, &idx, sizeof idx) < (ssize_t) sizeof idx)
    errx(EX_NOINPUT, "cannot read %s", opts_file);
  close(ofd);
  std::cerr << "k=" << idx.k << " l=" << idx.l
            << " spaced_seed_mask=0x" << std::hex << idx.spaced_seed_mask
            << " toggle_mask=0x" << idx.toggle_mask << std::dec
            << " revcom=" << idx.revcom_version << std::endl;

  int fd = open(argv[optind], O_RDONLY);
  if (fd < 0)
    errx(EX_NOINPUT, "cannot open %s", argv[optind]);
  FILE *out = fopen(out_file, "wb");
  if (! out)
    errx(EX_CANTCREAT, "cannot write %s", out_file);

  KeyValueStore *hash = nullptr;
  if (hash_file) {
    hash = new CompactHashTable<CompactHashCell>(hash_file, true);
    std::cerr << "loaded " << hash_file << std::endl;
  }

  uint64_t n_reads = 0, n_min = 0, n_looked_up = 0;
  char flags = want_taxa ? 3 : (hash_file ? 2 : (per_position ? 1 : 0));
  fwrite(DUMP_MAGIC, 1, 8, out);
  fwrite(&flags, 1, 1, out);
  long count_pos = ftell(out);
  fwrite(&n_reads, sizeof n_reads, 1, out);   // patched at the end

  MinimizerScanner scanner(idx.k, idx.l, idx.spaced_seed_mask, idx.dna_db,
                           idx.toggle_mask, idx.revcom_version);
  FastReader reader;
  StreamCursor cur;
  std::vector<uint64_t> mins;

  bool have_input = PrimeStream(fd, cur);
  while (have_input && n_reads < limit && reader.LoadBlock(fd, cur, 8 << 20)) {
    reader.Parse();
    for (size_t i = 0; i < reader.size() && n_reads < limit; i++) {
      const SeqView &s = reader[i];

      if (want_taxa) {
        // Mirrors ScanFragmentTokens and the replay that follows it: one entry
        // per reported minimizer, a repeat inheriting the running taxon, and a
        // skipped minimizer resetting it to zero.
        std::vector<uint32_t> taxa;
        scanner.LoadSequence(s.seq, s.seq_len);
        uint64_t last_minimizer = UINT64_MAX;
        uint32_t last_taxon = 0;
        uint64_t *mp;
        while ((mp = scanner.NextMinimizer()) != nullptr) {
          if (scanner.is_ambiguous()) {
            taxa.push_back(AMBIGUOUS_SPAN_TAXON_32);
            continue;
          }
          uint32_t taxon;
          if (*mp != last_minimizer) {
            last_minimizer = *mp;
            if (idx.minimum_acceptable_hash_value &&
                MurmurHash3(*mp) < idx.minimum_acceptable_hash_value) {
              taxon = 0; last_taxon = 0;
            } else {
              taxon = (uint32_t) hash->Get(*mp);
              last_taxon = taxon;
            }
          } else {
            taxon = last_taxon;
          }
          taxa.push_back(taxon);
        }
        uint32_t cnt = (uint32_t) taxa.size();
        fwrite(&cnt, sizeof cnt, 1, out);
        if (cnt) fwrite(&taxa[0], sizeof(uint32_t), cnt, out);
        for (uint32_t j = 0; j < cnt; j++)
          if (taxa[j] && taxa[j] != AMBIGUOUS_SPAN_TAXON_32) n_looked_up++;
        n_min += cnt;
        n_reads++;
        continue;
      }

      mins.clear();
      scanner.LoadSequence(s.seq, s.seq_len);
      uint64_t *m;
      while ((m = scanner.NextMinimizer()) != nullptr) {
        if (scanner.is_ambiguous())
          continue;
        mins.push_back(*m);
      }
      n_min += mins.size();
      if (! hash) {
        uint32_t cnt = (uint32_t) mins.size();
        fwrite(&cnt, sizeof cnt, 1, out);
        if (cnt)
          fwrite(&mins[0], sizeof(uint64_t), cnt, out);
        n_reads++;
        continue;
      }
      // Drop consecutive duplicates, then the subsampling filter, exactly as
      // ClassifySequence does before it builds its lookup batch.
      std::vector<uint64_t> keys;
      uint64_t last = 0;
      bool have_last = false;
      for (size_t j = 0; j < mins.size(); j++) {
        if (have_last && mins[j] == last)
          continue;
        last = mins[j];
        have_last = true;
        if (idx.minimum_acceptable_hash_value &&
            MurmurHash3(mins[j]) < idx.minimum_acceptable_hash_value)
          continue;
        keys.push_back(mins[j]);
      }
      uint32_t cnt = (uint32_t) keys.size();
      fwrite(&cnt, sizeof cnt, 1, out);
      for (uint32_t j = 0; j < cnt; j++) {
        uint64_t key = keys[j];
        uint32_t taxon = (uint32_t) hash->Get(key);
        fwrite(&key, sizeof key, 1, out);
        fwrite(&taxon, sizeof taxon, 1, out);
      }
      n_looked_up += cnt;
      n_reads++;
    }
  }
  fseek(out, count_pos, SEEK_SET);
  fwrite(&n_reads, sizeof n_reads, 1, out);
  fclose(out);
  close(fd);
  std::cerr << "reads " << n_reads << ", minimizers " << n_min
            << ", mean " << (n_reads ? (double) n_min / n_reads : 0)
            << " per read";
  if (hash)
    std::cerr << "; looked up " << n_looked_up << " ("
              << (n_min ? 100.0 * n_looked_up / n_min : 0) << "% of minimizers)";
  std::cerr << std::endl;
  delete hash;
  return 0;
}
