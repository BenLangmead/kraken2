/*
 * Block reader that keeps parsing out of the input critical section.
 *
 * BatchSequenceReader pulls records with kseq_read and copies each one into a
 * Sequence while the caller holds critical(seqread), so every thread's parsing
 * is serialized against every other thread's.  A sampling profile of classify at
 * 8 threads puts about half of all samples in __psynch_mutexwait.
 *
 * Here the critical section does two things only: read raw bytes, and cut the
 * buffer at an exact record boundary.  Parsing runs afterwards, outside the
 * lock, over the thread's own buffer.  Langmead et al. 2019 (Bioinformatics
 * 35:421) measures this arrangement, which it calls B/L-parsing, as the best of
 * four for thread scaling; kseq inside the lock is their O-parsing.
 *
 * Record boundaries come from counting newlines rather than scanning for a line
 * that begins with '@', which is unsound because Illumina quality strings
 * contain '@' at Phred 31.  Each block is cut on a boundary, so the next begins
 * on one and the count is exact by induction.  Bytes of a trailing partial
 * record are carried forward in a StreamCursor shared by the threads reading
 * that file, so a block costs one read and one small memcpy.
 */

#ifndef KRAKEN2_FAST_READER_H_
#define KRAKEN2_FAST_READER_H_

#include "kraken2_headers.h"
#include "seqreader.h"
#include "bgzf_index.h"

namespace kraken2 {

// A record as pointers into the reader's own buffer, valid until the next load.
// Avoids the four std::string copies a Sequence costs per record; the bases are
// also exactly the flat span the minimizer scanner and a device kernel want.
struct SeqView {
  const char *header;   // identifier, up to the first whitespace
  const char *comment;  // remainder of the header line, may be null
  const char *seq;
  const char *quals;    // null for FASTA
  uint32_t header_len;
  uint32_t comment_len;
  uint32_t seq_len;
  uint32_t quals_len;
  SequenceFormat format;

  // Quality masking rewrites bases in place in the reader's buffer.
  char *seq_mutable() const { return const_cast<char *>(seq); }
};

// Per-stream state: the tail of an incomplete record, carried to the next
// block.  One per input file, shared by all threads, only ever touched with the
// input lock held.
struct StreamCursor {
  std::vector<char> carry;
  SequenceFormat format;
  bool eof;
  StreamCursor() : format(FORMAT_AUTO_DETECT), eof(false) { }
};

class FastReader {
  public:
  FastReader();
  ~FastReader();

  // Both loaders must be called with the input lock held, and both leave the
  // reader holding a whole number of records.

  // Pulls about `target_bytes`, then trims back to the last complete record.
  // `record_multiple` forces the kept record count to be a multiple of that
  // value, so interleaved mate pairs are never split across blocks.
  bool LoadBlock(int fd, StreamCursor &cur, size_t target_bytes,
                 size_t record_multiple = 1);

  // Pulls exactly `records` records, for the second mate of a pair so the two
  // files stay in step.
  bool LoadRecords(int fd, StreamCursor &cur, size_t records);

  // Fills the reader with records [r0, r1) of an indexed, record-aligned BGZF
  // file.  Unlike the two loaders above this needs no lock and no shared
  // cursor: the caller has already claimed the range, the blocks holding it are
  // read with pread, and every block begins on a record boundary.  Whole blocks
  // are inflated, so the reader may hold records either side of the range;
  // Parse trims to it.
  bool LoadBgzfRange(const BgzfIndex &ix, uint64_t r0, uint64_t r1);

  // Number of records held, without parsing them.
  size_t RecordCount() const;

  // Call outside the lock.  Splits the loaded bytes into records in place.
  void Parse();

  size_t size() const { return records_.size(); }
  const SeqView &operator[](size_t i) const { return records_[i]; }
  SeqView &at(size_t i) { return records_[i]; }
  SequenceFormat file_format() const { return format_; }

  private:
  void ScanNewlines();
  void TruncateIndex(size_t keep);
  bool Fill(int fd, StreamCursor &cur, size_t bytes);

  std::vector<char> buf_;
  std::vector<uint32_t> nl_;   // offsets of newlines within buf_
  size_t scanned_;             // bytes of buf_ already covered by nl_
  std::vector<SeqView> records_;
  size_t record_count_;
  SequenceFormat format_;

  std::vector<char> cbuf_;     // compressed bytes of a claimed BGZF range
  void *inflater_;             // decompressor, allocated on first BGZF load
  size_t range_skip_;          // records of the first block preceding the range
  size_t range_take_;          // records in the range, SIZE_MAX if unbounded

  // Holds a decompressor, so copying would double free it.
  FastReader(const FastReader &);
  FastReader &operator=(const FastReader &);
};

}  // end namespace

#endif  // KRAKEN2_FAST_READER_H_
