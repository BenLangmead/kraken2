/*
 * An indexed, record-aligned BGZF input file.
 *
 * k2bgzf writes FASTQ as BGZF in which every block begins exactly at a record
 * boundary, plus a companion index giving, per block, its offset in the file,
 * its offset in the uncompressed stream, and the number of records that precede
 * it.  Two properties follow, and together they remove the input critical
 * section entirely.
 *
 * A thread that seeks to a block can parse forward immediately, because the
 * block cannot begin in the middle of a record.  Nothing has to scan for a
 * boundary, and no partial record has to be carried between threads, so there
 * is no shared cursor to guard.
 *
 * A range of record indices maps into each mate file's own blocks
 * independently, so the two files stay in step without either one being read in
 * order.  That is what the uncompressed path cannot do: mate files have no
 * fixed byte correspondence, which is why LoadRecords has to pull mate two in
 * lockstep behind mate one, under the same lock.
 */

#ifndef KRAKEN2_BGZF_INDEX_H_
#define KRAKEN2_BGZF_INDEX_H_

#include "kraken2_headers.h"

namespace kraken2 {

class BgzfIndex {
  public:
  BgzfIndex();
  ~BgzfIndex();

  // Opens bgz_path together with its index, which is bgz_path with a trailing
  // ".k2bgz" replaced by ".k2i", or ".k2i" appended if there is no such suffix.
  // Returns false, without exiting, if either file is missing or unreadable, so
  // the caller can fall back to the uncompressed path.
  bool Open(const std::string &bgz_path);

  uint64_t records() const { return n_records_; }
  int fd() const { return fd_; }

  // Describes the blocks spanning records [r0, r1): where they begin in the
  // file, how many compressed and uncompressed bytes they span, and how many
  // records of the first block precede r0.
  void Range(uint64_t r0, uint64_t r1, uint64_t *coff, size_t *clen,
             size_t *ulen, size_t *skip) const;

  private:
  struct Entry { uint64_t coff, uoff, rec; };

  std::vector<Entry> ents_;
  std::vector<uint64_t> recs_;   // ents_[i].rec, contiguous for binary search
  uint64_t n_records_;
  int fd_;

  BgzfIndex(const BgzfIndex &);
  BgzfIndex &operator=(const BgzfIndex &);
};

}  // end namespace

#endif  // KRAKEN2_BGZF_INDEX_H_
