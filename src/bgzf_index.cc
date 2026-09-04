/*
 * See bgzf_index.h.
 */

#include "bgzf_index.h"
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace kraken2 {

static const char K2_BGZF_MAGIC[8] = {'K','2','B','G','Z','F','\1','\0'};

struct IdxHeader {
  char magic[8];
  uint32_t version;
  uint32_t block_target;
  uint64_t n_blocks;
  uint64_t n_records;
  uint64_t total_uncompressed;
};

BgzfIndex::BgzfIndex() : n_records_(0), fd_(-1) { }

BgzfIndex::~BgzfIndex() {
  if (fd_ >= 0)
    close(fd_);
}

bool BgzfIndex::Open(const std::string &bgz_path) {
  std::string idx_path = bgz_path;
  const std::string suffix = ".k2bgz";
  if (idx_path.size() > suffix.size() &&
      idx_path.compare(idx_path.size() - suffix.size(), suffix.size(), suffix) == 0)
    idx_path.replace(idx_path.size() - suffix.size(), suffix.size(), ".k2i");
  else
    idx_path += ".k2i";

  FILE *f = fopen(idx_path.c_str(), "rb");
  if (! f)
    return false;
  IdxHeader h;
  if (fread(&h, sizeof h, 1, f) != 1 ||
      memcmp(h.magic, K2_BGZF_MAGIC, sizeof h.magic) != 0 || h.version != 1) {
    fclose(f);
    return false;
  }
  ents_.resize((size_t) h.n_blocks + 1);
  if (fread(&ents_[0], sizeof(Entry), ents_.size(), f) != ents_.size()) {
    fclose(f);
    ents_.clear();
    return false;
  }
  fclose(f);

  recs_.resize(ents_.size());
  for (size_t i = 0; i < ents_.size(); i++)
    recs_[i] = ents_[i].rec;
  n_records_ = h.n_records;

  fd_ = open(bgz_path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    ents_.clear();
    recs_.clear();
    return false;
  }
  return true;
}

void BgzfIndex::Range(uint64_t r0, uint64_t r1, uint64_t *coff, size_t *clen,
                      size_t *ulen, size_t *skip) const {
  size_t b0 = (size_t) (std::upper_bound(recs_.begin(), recs_.end(), r0) -
                        recs_.begin() - 1);
  size_t b1 = (size_t) (std::upper_bound(recs_.begin(), recs_.end(), r1 - 1) -
                        recs_.begin() - 1);
  *coff = ents_[b0].coff;
  *clen = (size_t) (ents_[b1 + 1].coff - ents_[b0].coff);
  *ulen = (size_t) (ents_[b1 + 1].uoff - ents_[b0].uoff);
  *skip = (size_t) (r0 - ents_[b0].rec);
}

}  // end namespace
