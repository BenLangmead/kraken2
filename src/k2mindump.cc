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
 * order.  With -p it emits instead one minimizer per k-mer position, before
 * consecutive duplicates are collapsed, which is the form a device kernel
 * naturally produces and from which the collapsed form can be derived.
 */

#include "kraken2_headers.h"
#include "kraken2_data.h"
#include "mmscanner.h"
#include "fast_reader.h"
#include <fcntl.h>
#include <unistd.h>

using namespace kraken2;

static const char DUMP_MAGIC[8] = {'K','2','M','D','U','M','P','\1'};

int main(int argc, char **argv) {
  const char *opts_file = nullptr, *out_file = nullptr;
  bool per_position = false;
  size_t limit = SIZE_MAX;
  int c;
  while ((c = getopt(argc, argv, "o:d:n:p")) != -1) {
    switch (c) {
      case 'o': opts_file = optarg; break;
      case 'd': out_file = optarg; break;
      case 'n': limit = (size_t) atoll(optarg); break;
      case 'p': per_position = true; break;
      default:
        std::cerr << "usage: k2mindump -o opts.k2d -d out.k2mm [-n reads] [-p] in.fq"
                  << std::endl;
        return 1;
    }
  }
  if (! opts_file || ! out_file || optind >= argc)
    errx(EX_USAGE, "-o, -d and an input file are all required");

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

  uint64_t n_reads = 0, n_min = 0;
  char flags = per_position ? 1 : 0;
  fwrite(DUMP_MAGIC, 1, 8, out);
  fwrite(&flags, 1, 1, out);
  long count_pos = ftell(out);
  fwrite(&n_reads, sizeof n_reads, 1, out);   // patched at the end

  MinimizerScanner scanner(idx.k, idx.l, idx.spaced_seed_mask, idx.dna_db,
                           idx.toggle_mask, idx.revcom_version);
  FastReader reader;
  StreamCursor cur;
  std::vector<uint64_t> mins;

  while (n_reads < limit && reader.LoadBlock(fd, cur, 8 << 20)) {
    reader.Parse();
    for (size_t i = 0; i < reader.size() && n_reads < limit; i++) {
      const SeqView &s = reader[i];
      mins.clear();
      scanner.LoadSequence(s.seq, s.seq_len);
      uint64_t *m;
      while ((m = scanner.NextMinimizer()) != nullptr) {
        if (scanner.is_ambiguous())
          continue;
        mins.push_back(*m);
      }
      uint32_t cnt = (uint32_t) mins.size();
      fwrite(&cnt, sizeof cnt, 1, out);
      if (cnt)
        fwrite(&mins[0], sizeof(uint64_t), cnt, out);
      n_reads++;
      n_min += cnt;
    }
  }
  fseek(out, count_pos, SEEK_SET);
  fwrite(&n_reads, sizeof n_reads, 1, out);
  fclose(out);
  close(fd);
  std::cerr << "reads " << n_reads << ", minimizers " << n_min
            << ", mean " << (n_reads ? (double) n_min / n_reads : 0)
            << " per read" << std::endl;
  return 0;
}
