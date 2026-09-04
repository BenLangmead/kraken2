/*
 * Block reader that keeps parsing out of the input critical section.
 * See fast_reader.h.
 */

#include "fast_reader.h"
#include <cstring>
#include <unistd.h>

#ifdef KRAKEN2_ZLIB_INFLATE
#include <zlib.h>
#else
#include <libdeflate.h>
#endif

namespace kraken2 {

static const size_t REFILL_CHUNK = 1 << 20;

FastReader::FastReader()
    : scanned_(0), record_count_(0), format_(FORMAT_AUTO_DETECT),
      inflater_(NULL), range_skip_(0), range_take_(SIZE_MAX) {
  buf_.reserve(16 << 20);
  nl_.reserve(1 << 17);
  records_.reserve(1 << 15);
}

FastReader::~FastReader() {
  if (! inflater_)
    return;
#ifdef KRAKEN2_ZLIB_INFLATE
  z_stream *zs = (z_stream *) inflater_;
  inflateEnd(zs);
  delete zs;
#else
  libdeflate_free_decompressor((libdeflate_decompressor *) inflater_);
#endif
}

// Inflates one BGZF block, which is a complete gzip member, and reports how
// many bytes it produced.
static bool InflateBlock(void *inflater, const unsigned char *in, size_t in_len,
                         char *out, size_t out_avail, size_t *out_len) {
#ifdef KRAKEN2_ZLIB_INFLATE
  z_stream *zs = (z_stream *) inflater;
  if (inflateReset(zs) != Z_OK)
    return false;
  zs->next_in = (Bytef *) in;
  zs->avail_in = (uInt) in_len;
  zs->next_out = (Bytef *) out;
  zs->avail_out = (uInt) out_avail;
  if (inflate(zs, Z_FINISH) != Z_STREAM_END)
    return false;
  *out_len = out_avail - zs->avail_out;
  return true;
#else
  return libdeflate_gzip_decompress((libdeflate_decompressor *) inflater, in,
                                    in_len, out, out_avail, out_len) ==
         LIBDEFLATE_SUCCESS;
#endif
}

void FastReader::ScanNewlines() {
  const char *base = buf_.data();
  const char *p = base + scanned_;
  const char *end = base + buf_.size();
  while (p < end) {
    const char *nl = (const char *) memchr(p, '\n', end - p);
    if (! nl) break;
    nl_.push_back((uint32_t) (nl - base));
    p = nl + 1;
  }
  scanned_ = buf_.size();
}

void FastReader::TruncateIndex(size_t keep) {
  size_t n = nl_.size();
  while (n > 0 && (size_t) nl_[n - 1] >= keep)
    n--;
  nl_.resize(n);
  scanned_ = keep;
}

// Reads until `bytes` have been appended or the descriptor is exhausted.  A
// single read() on a pipe returns whatever is available, so loop.
bool FastReader::Fill(int fd, StreamCursor &cur, size_t bytes) {
  size_t base = buf_.size();
  buf_.resize(base + bytes);
  size_t got = 0;
  while (got < bytes) {
    ssize_t n = read(fd, buf_.data() + base + got, bytes - got);
    if (n <= 0)
      break;
    got += (size_t) n;
  }
  buf_.resize(base + got);
  if (got == 0) {
    cur.eof = true;
    return false;
  }
  return true;
}

static void DetectFormat(StreamCursor &cur, const std::vector<char> &buf) {
  if (cur.format != FORMAT_AUTO_DETECT || buf.empty())
    return;
  if (buf[0] == '@') cur.format = FORMAT_FASTQ;
  else if (buf[0] == '>') cur.format = FORMAT_FASTA;
  else errx(EX_DATAERR, "sequence reader - unrecognized file format");
}

bool FastReader::LoadBlock(int fd, StreamCursor &cur, size_t target_bytes,
                           size_t record_multiple) {
  range_skip_ = 0;
  range_take_ = SIZE_MAX;
  records_.clear();
  record_count_ = 0;
  buf_.clear();
  nl_.clear();
  scanned_ = 0;
  buf_.insert(buf_.end(), cur.carry.begin(), cur.carry.end());
  cur.carry.clear();

  bool got = Fill(fd, cur, target_bytes);
  if (! got && buf_.empty())
    return false;
  DetectFormat(cur, buf_);
  format_ = cur.format;

  // At end of input a final record may lack its trailing newline.
  if (! got && ! buf_.empty() && buf_.back() != '\n')
    buf_.push_back('\n');

  if (record_multiple < 1)
    record_multiple = 1;
  ScanNewlines();
  while (record_multiple > 1 && got && nl_.size() / 4 < record_multiple) {
    got = Fill(fd, cur, target_bytes);
    if (! got && ! buf_.empty() && buf_.back() != '\n')
      buf_.push_back('\n');
    ScanNewlines();
  }

  size_t keep;
  if (format_ == FORMAT_FASTQ) {
    size_t recs = nl_.size() / 4;
    recs -= recs % record_multiple;
    keep = recs ? (size_t) nl_[recs * 4 - 1] + 1 : 0;
  }
  else {
    // FASTA: cut before the last line-initial '>', whose record may continue
    // into the next block.
    keep = buf_.size();
    if (got) {
      size_t i = buf_.size();
      while (i > 0) {
        i--;
        if (buf_[i] == '>' && (i == 0 || buf_[i - 1] == '\n')) {
          keep = i;
          break;
        }
      }
    }
  }

  if (keep < buf_.size()) {
    cur.carry.assign(buf_.begin() + keep, buf_.end());
    buf_.resize(keep);
    TruncateIndex(keep);
  }
  return ! buf_.empty();
}

bool FastReader::LoadRecords(int fd, StreamCursor &cur, size_t records) {
  range_skip_ = 0;
  range_take_ = SIZE_MAX;
  records_.clear();
  record_count_ = 0;
  buf_.clear();
  nl_.clear();
  scanned_ = 0;
  buf_.insert(buf_.end(), cur.carry.begin(), cur.carry.end());
  cur.carry.clear();
  if (records == 0)
    return false;

  DetectFormat(cur, buf_);
  bool live = true;
  ScanNewlines();
  const size_t want_lines =
      (cur.format == FORMAT_FASTA) ? records * 2 : records * 4;
  while (nl_.size() < want_lines && live) {
    live = Fill(fd, cur, REFILL_CHUNK);
    ScanNewlines();
  }
  if (buf_.empty())
    return false;
  DetectFormat(cur, buf_);
  format_ = cur.format;

  if (! live && ! buf_.empty() && buf_.back() != '\n') {
    buf_.push_back('\n');
    ScanNewlines();
  }

  size_t keep;
  if (format_ == FORMAT_FASTQ) {
    size_t want = records * 4;
    size_t lines = nl_.size() < want ? (nl_.size() / 4) * 4 : want;
    keep = lines ? (size_t) nl_[lines - 1] + 1 : 0;
  }
  else {
    keep = buf_.size();
  }
  if (keep < buf_.size()) {
    cur.carry.assign(buf_.begin() + keep, buf_.end());
    buf_.resize(keep);
    TruncateIndex(keep);
  }
  return ! buf_.empty();
}

bool FastReader::LoadBgzfRange(const BgzfIndex &ix, uint64_t r0, uint64_t r1) {
  records_.clear();
  record_count_ = 0;
  // buf_ is deliberately not cleared: resizing up from zero would value
  // initialize the whole buffer, so every claim would memset several megabytes
  // before inflating over them.  Resizing from the previous length only zeroes
  // any growth, which stops once the buffer reaches the largest range seen.
  nl_.clear();
  scanned_ = 0;
  format_ = FORMAT_FASTQ;
  if (r0 >= r1 || r1 > ix.records())
    return false;

  uint64_t coff;
  size_t clen, ulen;
  ix.Range(r0, r1, &coff, &clen, &ulen, &range_skip_);
  range_take_ = (size_t) (r1 - r0);

  cbuf_.resize(clen);
  size_t got = 0;
  while (got < clen) {
    ssize_t n = pread(ix.fd(), &cbuf_[0] + got, clen - got, (off_t) (coff + got));
    if (n <= 0)
      break;
    got += (size_t) n;
  }
  if (got < clen)
    errx(EX_IOERR, "short read from indexed BGZF input");

  if (! inflater_) {
#ifdef KRAKEN2_ZLIB_INFLATE
    z_stream *zs = new z_stream;
    memset(zs, 0, sizeof *zs);
    if (inflateInit2(zs, 15 + 16) != Z_OK)
      errx(EX_SOFTWARE, "cannot initialize inflate");
    inflater_ = zs;
#else
    inflater_ = libdeflate_alloc_decompressor();
    if (! inflater_)
      errx(EX_SOFTWARE, "cannot allocate decompressor");
#endif
  }

  buf_.resize(ulen);
  size_t opos = 0, cpos = 0;
  while (cpos + 18 <= got) {
    const unsigned char *p = (const unsigned char *) &cbuf_[0] + cpos;
    if (p[0] != 0x1f || p[1] != 0x8b)
      errx(EX_DATAERR, "input is not BGZF at offset %llu",
           (unsigned long long) (coff + cpos));
    uint32_t xlen = p[10] | (p[11] << 8);
    if (xlen != 6 || p[12] != 'B' || p[13] != 'C')
      errx(EX_DATAERR, "BGZF block lacks its BC field");
    size_t bsize = (size_t) (p[16] | (p[17] << 8)) + 1;
    size_t produced = 0;
    if (! InflateBlock(inflater_, p, bsize, &buf_[0] + opos, buf_.size() - opos,
                       &produced))
      errx(EX_DATAERR, "inflate failed on a BGZF block");
    opos += produced;
    cpos += bsize;
  }
  buf_.resize(opos);
  ScanNewlines();
  return ! buf_.empty();
}

size_t FastReader::RecordCount() const {
  if (format_ == FORMAT_FASTQ)
    return nl_.size() / 4;
  size_t n = 0;
  for (size_t i = 0; i < buf_.size(); i++)
    if (buf_[i] == '>' && (i == 0 || buf_[i - 1] == '\n'))
      n++;
  return n;
}

void FastReader::Parse() {
  records_.clear();
  char *base = buf_.data();

  auto trim = [](const char *s, const char *e) -> uint32_t {
    while (e > s && (e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t'))
      e--;
    return (uint32_t) (e - s);
  };
  auto line_start = [&](size_t j) -> char * {
    return base + (j == 0 ? 0 : (size_t) nl_[j - 1] + 1);
  };
  auto line_stop = [&](size_t j) -> char * { return base + nl_[j]; };

  // Splits a header line into identifier and comment the way kseq does.
  auto set_header = [&](SeqView &v, char *hs, char *he) {
    char *p = hs;
    while (p < he && *p != ' ' && *p != '\t')
      p++;
    v.header = hs;
    v.header_len = trim(hs, p);
    while (p < he && (*p == ' ' || *p == '\t'))
      p++;
    v.comment = p;
    v.comment_len = trim(p, he);
  };

  if (format_ == FORMAT_FASTQ) {
    size_t recs = nl_.size() / 4;
    records_.resize(recs);
    size_t out = 0;
    for (size_t i = 0; i < recs; i++) {
      size_t j = i * 4;
      char *hs = line_start(j), *he = line_stop(j);
      if (hs >= he || *hs != '@')
        break;
      SeqView &v = records_[out];
      v.format = FORMAT_FASTQ;
      set_header(v, hs + 1, he);
      char *q = line_start(j + 1);
      v.seq = q;
      v.seq_len = trim(q, line_stop(j + 1));
      q = line_start(j + 3);
      v.quals = q;
      v.quals_len = trim(q, line_stop(j + 3));
      out++;
    }
    records_.resize(out);
    // Whole blocks were inflated, so drop whatever falls outside the range the
    // caller claimed.
    if (range_skip_ > 0 || range_take_ != SIZE_MAX) {
      size_t skip = range_skip_ < records_.size() ? range_skip_ : records_.size();
      records_.erase(records_.begin(), records_.begin() + skip);
      if (records_.size() > range_take_)
        records_.resize(range_take_);
    }
    record_count_ = records_.size();
    return;
  }

  // FASTA: sequence spans every line up to the next header, spliced in place by
  // shifting bytes down over the newlines.
  size_t nlines = nl_.size();
  size_t j = 0;
  while (j < nlines) {
    char *hs = line_start(j), *he = line_stop(j);
    if (hs >= he || *hs != '>')
      break;
    SeqView v;
    v.format = FORMAT_FASTA;
    v.quals = nullptr;
    v.quals_len = 0;
    set_header(v, hs + 1, he);
    j++;
    char *dst = (j < nlines) ? line_start(j) : base + buf_.size();
    v.seq = dst;
    while (j < nlines) {
      char *ls = line_start(j);
      if (*ls == '>')
        break;
      uint32_t len = trim(ls, line_stop(j));
      if (dst != ls)
        memmove(dst, ls, len);
      dst += len;
      j++;
    }
    v.seq_len = (uint32_t) (dst - v.seq);
    records_.push_back(v);
  }
  record_count_ = records_.size();
}

}  // end namespace
