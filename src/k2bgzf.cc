/*
 * k2bgzf -- write a FASTQ file as record-aligned BGZF with a record-aware index.
 *
 * Every emitted block begins exactly at a FASTQ record start, so a reader that
 * seeks to a block never has to scan for a record boundary, and two mate files
 * can be claimed by record range independently.  See bgzf_index.h.
 *
 * Given two mate files, it writes their records interleaved (mate 1, mate 2,
 * mate 1, ...) and ends blocks only between pairs, so every block holds whole
 * pairs.  The index then has version 2, which readers of single-end files
 * reject.
 *
 * The output is ordinary BGZF, so it stays a valid gzip file: gunzip, and any
 * tool that reads .gz, sees the original bytes.  Ending blocks on record
 * boundaries costs about a quarter of a percent of compression ratio, since a
 * block gives up on average half a record of its 64 KB.  The default level
 * makes that back: it lands within a fraction of a percent of gzip -6, and -l
 * 12 comes out several percent smaller than gzip while staying seekable.
 *
 * Blocks are compressed in parallel and written in order, which is why this
 * does not use htslib: bgzf_flush drains the whole thread pool, so forcing a
 * block boundary per record group there serializes the writer.
 */

#include "kraken2_headers.h"
#include <libdeflate.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fcntl.h>
#include <unistd.h>

using std::vector;
using std::string;

namespace {

// A BGZF block is an 18-byte gzip header with the BC extra field, the deflate
// stream, then CRC32 and ISIZE.  The whole thing must fit in 64 KiB.
const size_t BGZF_FRAMING = 26;
const size_t BGZF_MAX_BLOCK = 65536;
const size_t DEFAULT_TARGET = 60000;
const size_t BLOCKS_PER_WAVE = 512;

const unsigned char BGZF_EOF[28] = {
  0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00,
  0x42, 0x43, 0x02, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};

const char K2_BGZF_MAGIC[8] = {'K','2','B','G','Z','F','\1','\0'};

struct IdxHeader {
  char magic[8];
  uint32_t version;
  uint32_t block_target;
  uint64_t n_blocks;
  uint64_t n_records;
  uint64_t total_uncompressed;
};
struct IdxEnt { uint64_t coff, uoff, rec; };

// Hands back whole four-line FASTQ records as spans of its own buffer.
class RecordReader {
  public:
  explicit RecordReader(int fd) : fd_(fd), pos_(0), eof_(false) {
    buf_.reserve(1 << 22);
  }

  bool Next(const char **ptr, size_t *len) {
    while (true) {
      size_t p = pos_;
      int lines = 0;
      while (lines < 4) {
        const void *nl = memchr(&buf_[0] + p, '\n', buf_.size() - p);
        if (! nl)
          break;
        p = (const char *) nl - &buf_[0] + 1;
        lines++;
      }
      if (lines == 4) {
        *ptr = &buf_[0] + pos_;
        *len = p - pos_;
        pos_ = p;
        return true;
      }
      if (eof_) {
        if (pos_ >= buf_.size())
          return false;
        if (lines > 0 && buf_.back() != '\n') {
          buf_.push_back('\n');
          continue;
        }
        errx(EX_DATAERR, "truncated FASTQ record at end of input");
      }
      Refill();
    }
  }

  private:
  void Refill() {
    if (pos_ > 0) {
      buf_.erase(buf_.begin(), buf_.begin() + pos_);
      pos_ = 0;
    }
    size_t base = buf_.size();
    buf_.resize(base + (1 << 22));
    size_t got = 0;
    while (got < (1u << 22)) {
      ssize_t n = read(fd_, &buf_[0] + base + got, (1 << 22) - got);
      if (n <= 0)
        break;
      got += (size_t) n;
    }
    buf_.resize(base + got);
    if (got == 0)
      eof_ = true;
  }

  int fd_;
  vector<char> buf_;
  size_t pos_;
  bool eof_;
};

// Frames one already-compressed deflate stream as a BGZF block.
void WriteFraming(unsigned char *hdr, size_t dlen) {
  size_t total = dlen + BGZF_FRAMING;
  hdr[0] = 0x1f; hdr[1] = 0x8b; hdr[2] = 0x08; hdr[3] = 0x04;
  hdr[4] = hdr[5] = hdr[6] = hdr[7] = 0;   // MTIME
  hdr[8] = 0;                              // XFL
  hdr[9] = 0xff;                           // OS unknown
  hdr[10] = 6; hdr[11] = 0;                // XLEN
  hdr[12] = 'B'; hdr[13] = 'C';
  hdr[14] = 2; hdr[15] = 0;
  hdr[16] = (unsigned char) ((total - 1) & 0xff);
  hdr[17] = (unsigned char) (((total - 1) >> 8) & 0xff);
}

void PutU32(unsigned char *p, uint32_t v) {
  p[0] = (unsigned char) (v & 0xff);
  p[1] = (unsigned char) ((v >> 8) & 0xff);
  p[2] = (unsigned char) ((v >> 16) & 0xff);
  p[3] = (unsigned char) ((v >> 24) & 0xff);
}

}  // anonymous namespace

int main(int argc, char **argv) {
  const char *in_path = NULL, *out_path = NULL, *idx_path = NULL;
  int threads = 1, level = 9;
  size_t target = DEFAULT_TARGET;
  int c;
  while ((c = getopt(argc, argv, "o:i:t:b:l:h")) != -1) {
    switch (c) {
      case 'o': out_path = optarg; break;
      case 'i': idx_path = optarg; break;
      case 't': threads = atoi(optarg); break;
      case 'b': target = (size_t) atoll(optarg); break;
      case 'l': level = atoi(optarg); break;
      default:
        std::cerr << "usage: k2bgzf [-t threads] [-b block_bytes] [-l level]"
                  << " -o out.k2bgz -i out.k2i in.fq [mate2.fq]" << std::endl;
        return 1;
    }
  }
  const char *in2_path = NULL;
  if (optind < argc)
    in_path = argv[optind];
  if (optind + 1 < argc)
    in2_path = argv[optind + 1];
  if (! out_path || ! idx_path)
    errx(EX_USAGE, "-o and -i are both required");
  if (target > BGZF_MAX_BLOCK - BGZF_FRAMING)
    target = BGZF_MAX_BLOCK - BGZF_FRAMING;
  if (threads < 1)
    threads = 1;

  int fd = (! in_path || ! strcmp(in_path, "-")) ? 0 : open(in_path, O_RDONLY);
  if (fd < 0)
    errx(EX_NOINPUT, "cannot open %s", in_path);
  FILE *out = fopen(out_path, "wb");
  if (! out)
    errx(EX_CANTCREAT, "cannot write %s", out_path);

  RecordReader rr(fd);
  int fd2 = -1;
  if (in2_path && (fd2 = open(in2_path, O_RDONLY)) < 0)
    errx(EX_NOINPUT, "cannot open %s", in2_path);
  RecordReader rr2(fd2);
  const int per_unit = in2_path ? 2 : 1;
  vector<IdxEnt> ents;
  uint64_t coff = 0, uoff = 0, nrec = 0;
  size_t max_rec = 0;

  // One wave of blocks: the raw bytes, where each block starts and ends, and
  // the compressed result.
  vector<char> stage;
  vector<size_t> blk_off, blk_len;
  vector<uint64_t> blk_rec;
  vector<vector<unsigned char> > cblk;
  stage.reserve(BLOCKS_PER_WAVE * target + target);

  bool more = true;
  while (more) {
    stage.clear();
    blk_off.clear();
    blk_len.clear();
    blk_rec.clear();
    size_t cur_off = 0, cur_len = 0;
    uint64_t cur_rec = nrec;

    // A unit is one record, or a pair of records when there are two mate
    // files; blocks end only between units.
    const char *urec[2];
    size_t ulen[2];
    while (blk_off.size() < BLOCKS_PER_WAVE) {
      more = rr.Next(&urec[0], &ulen[0]);
      if (per_unit == 2) {
        bool more2 = rr2.Next(&urec[1], &ulen[1]);
        if (more != more2)
          errx(EX_DATAERR, "the mate files have different numbers of records");
      }
      if (! more)
        break;
      size_t total = 0;
      for (int m = 0; m < per_unit; m++) {
        if (ulen[m] > max_rec)
          max_rec = ulen[m];
        total += ulen[m];
      }
      if (total > target)
        errx(EX_DATAERR,
             "record %llu is %zu bytes%s, larger than the %zu-byte block target; "
             "BGZF blocks cannot exceed 64 KB, so this file cannot be record "
             "aligned",
             (unsigned long long) nrec, total, per_unit == 2 ? " with its mate" : "",
             target);
      if (cur_len > 0 && cur_len + total > target) {
        blk_off.push_back(cur_off);
        blk_len.push_back(cur_len);
        blk_rec.push_back(cur_rec);
        cur_off += cur_len;
        cur_len = 0;
        cur_rec = nrec;
      }
      for (int m = 0; m < per_unit; m++)
        stage.insert(stage.end(), urec[m], urec[m] + ulen[m]);
      cur_len += total;
      nrec += per_unit;
    }
    if (cur_len > 0) {
      blk_off.push_back(cur_off);
      blk_len.push_back(cur_len);
      blk_rec.push_back(cur_rec);
    }
    if (blk_off.empty())
      break;

    cblk.resize(blk_off.size());
    bool failed = false;
    #pragma omp parallel num_threads(threads)
    {
      libdeflate_compressor *co = libdeflate_alloc_compressor(level);
      if (! co) {
        #pragma omp critical(k2bgzf_fail)
        failed = true;
      }
      else {
        #pragma omp for schedule(dynamic, 4)
        for (long i = 0; i < (long) blk_off.size(); i++) {
          vector<unsigned char> &o = cblk[i];
          o.resize(BGZF_MAX_BLOCK);
          size_t dlen = libdeflate_deflate_compress(
              co, &stage[0] + blk_off[i], blk_len[i], &o[0] + 18,
              BGZF_MAX_BLOCK - BGZF_FRAMING);
          if (dlen == 0) {
            #pragma omp critical(k2bgzf_fail)
            failed = true;
            continue;
          }
          WriteFraming(&o[0], dlen);
          PutU32(&o[0] + 18 + dlen,
                 libdeflate_crc32(0, &stage[0] + blk_off[i], blk_len[i]));
          PutU32(&o[0] + 18 + dlen + 4, (uint32_t) blk_len[i]);
          o.resize(18 + dlen + 8);
        }
        libdeflate_free_compressor(co);
      }
    }
    if (failed)
      errx(EX_SOFTWARE, "deflate failed");

    for (size_t i = 0; i < cblk.size(); i++) {
      IdxEnt e;
      e.coff = coff;
      e.uoff = uoff;
      e.rec = blk_rec[i];
      ents.push_back(e);
      if (fwrite(&cblk[i][0], 1, cblk[i].size(), out) != cblk[i].size())
        errx(EX_IOERR, "short write to %s", out_path);
      coff += cblk[i].size();
      uoff += blk_len[i];
    }
  }

  if (fwrite(BGZF_EOF, 1, sizeof BGZF_EOF, out) != sizeof BGZF_EOF)
    errx(EX_IOERR, "short write to %s", out_path);
  if (fclose(out) != 0)
    errx(EX_IOERR, "closing %s failed", out_path);
  if (fd > 0)
    close(fd);
  if (fd2 > 0)
    close(fd2);

  IdxEnt sentinel;
  sentinel.coff = coff;
  sentinel.uoff = uoff;
  sentinel.rec = nrec;
  ents.push_back(sentinel);

  FILE *ix = fopen(idx_path, "wb");
  if (! ix)
    errx(EX_CANTCREAT, "cannot write %s", idx_path);
  IdxHeader h;
  memcpy(h.magic, K2_BGZF_MAGIC, sizeof h.magic);
  h.version = per_unit == 2 ? 2 : 1;
  h.block_target = (uint32_t) target;
  h.n_blocks = ents.size() - 1;
  h.n_records = nrec;
  h.total_uncompressed = uoff;
  if (fwrite(&h, sizeof h, 1, ix) != 1 ||
      fwrite(&ents[0], sizeof(IdxEnt), ents.size(), ix) != ents.size() ||
      fclose(ix) != 0)
    errx(EX_IOERR, "writing %s failed", idx_path);

  std::cerr << "k2bgzf: " << nrec << " records, " << h.n_blocks << " blocks, "
            << uoff << " uncompressed bytes, " << coff << " compressed, "
            << "longest record " << max_rec << " bytes" << std::endl;
  return 0;
}
