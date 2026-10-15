// End to end: record-aligned BGZF in, one classification call per read out,
// with nothing touching the host in between.
//
// Every earlier program measured a stage.  This one runs the whole thing as a
// single pipeline, which is the only way to find out what the stage figures add
// up to in practice: compressed blocks are sent to the card, expanded there,
// split into reads there, classified there, and only one identifier per read
// comes back.  The reads themselves never exist in host memory.
//
// Correctness is still settled against classify's own output, so a faster
// pipeline that disagrees is reported as a failure rather than a speed-up.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <nvcomp/gzip.h>
#include "kernels.cuh"
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include "../src/mmscanner.h"

#define CK(x) do { cudaError_t e=(x); if (e!=cudaSuccess) { \
  fprintf(stderr,"cuda %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)
#define NK(x) do { nvcompStatus_t st=(x); if (st!=nvcompSuccess) { \
  fprintf(stderr,"nvcomp %d at %d\n",(int)st,__LINE__); exit(1);} } while(0)

struct IdxHeader {
  char magic[8]; uint32_t version, block_target;
  uint64_t n_blocks, n_records, total_uncompressed;
};
struct IdxEnt { uint64_t coff, uoff, rec; };

// Kraken 2's own scanner, over the handful of reads carrying an ambiguous
// base.  Using it rather than reproducing its refill behaviour is the point:
// it emits the ambiguous spans exactly, including the token-per-character it
// produces while recovering, which is what the device version gets wrong.
struct HostFallback {
  k2t::Table table;
  const uint64_t *parent, *extid; uint64_t mahv;
  kraken2::MinimizerScanner *scanner;

  bool is_anc(uint64_t a, uint64_t b) const {
    if (!a || !b) return false;
    while (b > a) b = parent[b];
    return b == a;
  }
  uint64_t lca_h(uint64_t a, uint64_t b) const {
    if (!a || !b) return a ? a : b;
    while (a != b) { if (a > b) a = parent[a]; else b = parent[b]; }
    return a;
  }

  std::string line(const char *hdr, uint32_t hlen, const char *seq, uint32_t slen) {
    static const uint32_t AMBIG = 0xFFFFFFFDu;   // TAXID_MAX - 2, narrowed
    std::vector<uint32_t> taxa;
    std::map<uint32_t, uint64_t> hits;
    scanner->LoadSequence(seq, slen);
    uint64_t last_min = UINT64_MAX; uint32_t last_taxon = 0, *mpv;
    uint64_t *mp;
    (void) mpv;
    while ((mp = scanner->NextMinimizer()) != nullptr) {
      if (scanner->is_ambiguous()) { taxa.push_back(AMBIG); continue; }
      uint32_t taxon;
      if (*mp != last_min) {
        last_min = *mp;
        if (mahv && murmur(*mp) < mahv) { taxon = 0; last_taxon = 0; }
        else { taxon = table.get(*mp); last_taxon = taxon; }
      } else taxon = last_taxon;
      if (taxon) hits[taxon]++;
      taxa.push_back(taxon);
    }
    uint64_t max_taxon = 0; uint32_t max_score = 0;
    for (auto &a : hits) {
      uint32_t sc = 0;
      for (auto &b : hits) if (is_anc(b.first, a.first)) sc += (uint32_t) b.second;
      if (sc > max_score) { max_score = sc; max_taxon = a.first; }
      else if (sc == max_score) max_taxon = lca_h(max_taxon, a.first);
    }
    uint64_t ext = max_taxon ? extid[max_taxon] : 0;

    char buf[8192]; int n = 0;
    n += snprintf(buf+n, sizeof buf-n, "%c\t", max_taxon ? 'C' : 'U');
    n += snprintf(buf+n, sizeof buf-n, "%.*s\t%llu\t%u\t",
                  (int) hlen, hdr, (unsigned long long) ext, slen);
    if (taxa.empty()) n += snprintf(buf+n, sizeof buf-n, "0:0");
    else {
      uint32_t last = taxa[0]; int run = 1;
      for (size_t i = 1; i < taxa.size(); i++) {
        if (taxa[i] == last) { run++; continue; }
        if (last == AMBIG) n += snprintf(buf+n, sizeof buf-n, "A:%d ", run);
        else n += snprintf(buf+n, sizeof buf-n, "%llu:%d ",
                           (unsigned long long) (last ? extid[last] : 0), run);
        last = taxa[i]; run = 1;
      }
      // Kraken 2's last hit-list entry carries no trailing space, except in
      // the ambiguous branch, which emits one.  Reproduced deliberately.
      if (last == AMBIG) n += snprintf(buf+n, sizeof buf-n, "A:%d ", run);
      else n += snprintf(buf+n, sizeof buf-n, "%llu:%d",
                         (unsigned long long) (last ? extid[last] : 0), run);
    }
    n += snprintf(buf+n, sizeof buf-n, "\n");
    return std::string(buf, n);
  }
};

static double now_s() {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    fprintf(stderr,
      "usage: k2gpu reads.k2bgz reads.k2i hash.k2d opts.k2d taxo.k2d classify.out\n"
      "             [blocks_per_chunk]\n");
    return 1;
  }
  size_t chunk_blocks = argc > 7 ? (size_t) atoll(argv[7]) : 4096;
  // Stopping after a few chunks keeps a run under compute-sanitizer short
  // enough to be usable.
  size_t max_chunks = argc > 8 ? (size_t) atoll(argv[8]) : (size_t) -1;
  // Writing the answer is part of the job, so it is timed with the rest.
  const char *out_path = getenv("K2GPU_OUT");
  // K2GPU_SKIP_PROBE=1 replaces every table lookup by a miss, so the change in
  // the classify kernel's time is the lookup's share.  Output is then wrong
  // by design and the call check fails.
  auto env_int = [](const char *n, int d) { return getenv(n) ? atoi(getenv(n)) : d; };
  int skip_probe = env_int("K2GPU_SKIP_PROBE", 0);
  // K2GPU_GROUPS=G streams a blocked table through the device in G groups of
  // whole partitions instead of holding it resident (FUTURE.md's
  // partition-outer loop).  K2GPU_BIN=1 scans once and probes minimizers
  // grouped by partition; otherwise each group pass rescans the reads.
  // K2GPU_SERIAL=1 turns off the overlap of copies with probes.
  uint32_t n_groups = (uint32_t) env_int("K2GPU_GROUPS", 0);
  bool streamed = n_groups > 0;
  bool binned = env_int("K2GPU_BIN", 0) != 0;
  bool serial = env_int("K2GPU_SERIAL", 0) != 0;
  double bin_frac = getenv("K2GPU_BIN_FRAC") ? atof(getenv("K2GPU_BIN_FRAC")) : 0.5;

  // ---- index ----
  IdxHeader ih;
  std::vector<IdxEnt> ent;
  {
    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open index %s\n", argv[2]); return 1; }
    if (fread(&ih, sizeof ih, 1, f) != 1 || memcmp(ih.magic, "K2BGZF\1\0", 8) != 0) {
      fprintf(stderr, "%s is not a k2bgzf index\n", argv[2]); return 1;
    }
    ent.resize(ih.n_blocks + 1);
    if (fread(ent.data(), sizeof(IdxEnt), ent.size(), f) != ent.size()) return 1;
    fclose(f);
  }
  int bgz = open(argv[1], O_RDONLY);
  if (bgz < 0) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  printf("input: %" PRIu64 " blocks, %" PRIu64 " reads, %.2f GB uncompressed, "
         "%.2f GB on disk\n", ih.n_blocks, ih.n_records,
         ih.total_uncompressed / 1e9, ent[ih.n_blocks].coff / 1e9);

  // ---- index options ----
  uint64_t o[8];
  { int fd = open(argv[4], O_RDONLY);
    if (fd < 0 || read(fd, o, sizeof o) < (ssize_t) sizeof o) { perror("opts"); return 1; }
    close(fd); }
  int k = (int) o[0], l = (int) o[1];
  uint64_t ssm = o[2], tog = o[3], mahv = o[5];

  // ---- hash table ----
  // The file's first 8 bytes say which table it is, as in classify.  The
  // device copy and the host view (used by the fallback path) share one
  // k2t::Table description and differ only in where the pointers point.
  // Mapping the file costs nothing until a page is touched, and the fallback
  // touches only a handful.
  double t_load0 = now_s();
  k2t::Table d_tab{}, h_tab{};
  size_t table_bytes = 0;
  int hf = open(argv[3], O_RDONLY);
  struct stat hst;
  if (hf < 0 || fstat(hf, &hst) != 0) { perror("hash"); return 1; }
  char *hmap = (char *) mmap(nullptr, hst.st_size, PROT_READ, MAP_PRIVATE, hf, 0);
  if (hmap == MAP_FAILED) { perror("mmap hash"); return 1; }
  // Copies [off, off + n) of the file to the device in 256 MB pieces through
  // a pinned buffer.
  const size_t LOAD_CH = 256u << 20;
  char *load_pin; CK(cudaHostAlloc(&load_pin, LOAD_CH, cudaHostAllocDefault));
  auto upload = [&](void *dst, size_t off, size_t n) {
    for (size_t done = 0; done < n; ) {
      size_t want = std::min(LOAD_CH, n - done), got = 0;
      while (got < want) {
        ssize_t r = pread(hf, load_pin + got, want - got, off + done + got);
        if (r <= 0) { fprintf(stderr, "short read in %s\n", argv[3]); exit(1); }
        got += r;
      }
      CK(cudaMemcpy((char *) dst + done, load_pin, got, cudaMemcpyHostToDevice));
      done += got;
    }
  };
  uint64_t magic; memcpy(&magic, hmap, 8);
  size_t blk_bucket_off = 0;
  if (magic == k2t::BLOCKED_MAGIC) {
    k2t::BlockedHeader bh; memcpy(&bh, hmap, sizeof bh);
    if (bh.version != 1 || bh.bucket_bytes != 64) {
      fprintf(stderr, "unsupported blocked table version\n"); return 1;
    }
    h_tab.kind = k2t::Table::BLOCKED;
    h_tab.G.init(bh);
    if ((uint32_t) h_tab.G.S != bh.slots || (uint32_t) h_tab.G.sf != bh.spill_bits ||
        (uint32_t) h_tab.G.B != bh.key_bits) {
      fprintf(stderr, "blocked table header is inconsistent\n"); return 1;
    }
    size_t nbk = (size_t) (h_tab.G.P * h_tab.G.nb) * 64;
    blk_bucket_off = k2t::BLOCKED_HEADER_BYTES;
    size_t off_o = blk_bucket_off + nbk;
    size_t off_k = off_o + (h_tab.G.P + 1) * 8;
    size_t off_v = off_k + bh.stash_entries * 8;
    h_tab.buckets = (const uint8_t *) (hmap + blk_bucket_off);
    h_tab.part_base = 0;
    h_tab.stash_off = (const uint64_t *) (hmap + off_o);
    h_tab.stash_key = (const uint64_t *) (hmap + off_k);
    h_tab.stash_val = (const uint32_t *) (hmap + off_v);
    d_tab = h_tab;
    uint64_t *dso, *dsk = nullptr; uint32_t *dsv = nullptr;
    CK(cudaMalloc(&dso, (h_tab.G.P + 1) * 8));
    upload(dso, off_o, (h_tab.G.P + 1) * 8);
    if (bh.stash_entries) {
      CK(cudaMalloc(&dsk, bh.stash_entries * 8)); upload(dsk, off_k, bh.stash_entries * 8);
      CK(cudaMalloc(&dsv, bh.stash_entries * 4)); upload(dsv, off_v, bh.stash_entries * 4);
    }
    d_tab.stash_off = dso; d_tab.stash_key = dsk; d_tab.stash_val = dsv;
    table_bytes = nbk;
    printf("table: blocked, %" PRIu64 " partitions of %" PRIu64 " buckets, "
           "%d slots of %d bits, %d-bit spill filter, stash %" PRIu64 "\n",
           h_tab.G.P, h_tab.G.nb, h_tab.G.S, h_tab.G.w, h_tab.G.sf, bh.stash_entries);
  } else {
    uint64_t hdr[4]; memcpy(hdr, hmap, sizeof hdr);
    h_tab.kind = k2t::Table::CHT;
    h_tab.capacity = hdr[0]; h_tab.key_bits = (int) hdr[2]; h_tab.value_bits = (int) hdr[3];
    h_tab.cells = (const uint32_t *) (hmap + 32);
    d_tab = h_tab;
    table_bytes = h_tab.capacity * 4;
    printf("table: compact hash, capacity %" PRIu64 ", %d key bits, %d value bits\n",
           h_tab.capacity, h_tab.key_bits, h_tab.value_bits);
  }
  // A resident run holds the whole table on the device.  A streamed run holds
  // the buckets in pinned host memory and one or two device slots of one
  // group each; the stash, which is small, stays resident.
  int n_slots = 0;
  uint8_t *d_slot[2] = {nullptr, nullptr};
  const uint8_t *h_buckets = nullptr;
  if (streamed) {
    if (h_tab.kind != k2t::Table::BLOCKED || h_tab.G.P % n_groups != 0 || h_tab.G.v > 16) {
      fprintf(stderr, "streaming needs a blocked table whose partition count is a multiple "
              "of K2GPU_GROUPS and whose values fit 16 bits\n");
      return 1;
    }
    uint8_t *hb; CK(cudaHostAlloc(&hb, table_bytes, cudaHostAllocDefault));
    for (size_t done = 0; done < table_bytes; ) {
      ssize_t r = pread(hf, hb + done, table_bytes - done, blk_bucket_off + done);
      if (r <= 0) { fprintf(stderr, "short read in %s\n", argv[3]); return 1; }
      done += r;
    }
    h_buckets = hb;
    n_slots = n_groups == 1 ? 1 : 2;
    size_t group_bytes = table_bytes / n_groups;
    for (int s = 0; s < n_slots; s++) CK(cudaMalloc(&d_slot[s], group_bytes));
    // Host-to-device rates for the same bytes of buckets (one slot, at most
    // 1 GB): pinned, as the group copies use, and pageable, from the file
    // mapping already in the page cache, as the earlier measurement used.
    size_t nb_test = std::min(group_bytes, (size_t) 1 << 30);
    for (int rep = 0; rep < 2; rep++) {
      double t1 = now_s();
      CK(cudaMemcpy(d_slot[0], h_buckets, nb_test, cudaMemcpyHostToDevice));
      double t2 = now_s();
      CK(cudaMemcpy(d_slot[0], hmap + blk_bucket_off, nb_test, cudaMemcpyHostToDevice));
      double t3 = now_s();
      if (rep == 1)
        printf("host to device, %.2f GB: pinned %.1f GB/s, pageable %.1f GB/s\n",
               nb_test / 1e9, nb_test / 1e9 / (t2 - t1), nb_test / 1e9 / (t3 - t2));
    }
  } else if (h_tab.kind == k2t::Table::BLOCKED) {
    uint8_t *db; CK(cudaMalloc(&db, table_bytes));
    upload(db, blk_bucket_off, table_bytes);
    d_tab.buckets = db;
  } else {
    uint32_t *dc; CK(cudaMalloc(&dc, table_bytes));
    upload(dc, 32, table_bytes);
    d_tab.cells = dc;
  }
  CK(cudaDeviceSynchronize());
  double t_load = now_s() - t_load0;
  printf("table: %.2f GB %s, loaded in %.2f s\n", table_bytes / 1e9,
         streamed ? "in pinned host memory" : "resident", t_load);

  // ---- taxonomy: parent and external id only ----
  uint64_t *d_parent, *d_extid, node_count;
  std::vector<uint64_t> h_par, h_ext;
  {
    int tf = open(argv[5], O_RDONLY);
    char hdr[32];
    if (tf < 0 || read(tf, hdr, 32) < 32 || memcmp(hdr, "K2TAXDAT", 8) != 0) {
      fprintf(stderr, "%s is not a taxonomy\n", argv[5]); return 1;
    }
    memcpy(&node_count, hdr + 8, 8);
    std::vector<uint64_t> raw(node_count * 7);
    size_t want = raw.size() * 8, got = 0;
    while (got < want) {
      ssize_t n = pread(tf, (char *) raw.data() + got, want - got, 32 + got);
      if (n <= 0) break;
      got += n;
    }
    close(tf);
    h_par.resize(node_count); h_ext.resize(node_count);
    std::vector<uint64_t> &par = h_par, &ext = h_ext;
    for (uint64_t i = 0; i < node_count; i++) { par[i] = raw[i*7]; ext[i] = raw[i*7+5]; }
    CK(cudaMalloc(&d_parent, node_count*8)); CK(cudaMalloc(&d_extid, node_count*8));
    CK(cudaMemcpy(d_parent, par.data(), node_count*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_extid,  ext.data(), node_count*8, cudaMemcpyHostToDevice));
    printf("taxonomy: %" PRIu64 " nodes\n", node_count);
  }

  // ---- per-chunk sizing, from the index rather than guessed ----
  std::vector<size_t> chunk_b0;
  size_t max_c = 0, max_u = 0, max_r = 0, max_rec_per_blk = 0;
  for (size_t b0 = 0; b0 < ih.n_blocks; b0 += chunk_blocks) {
    size_t b1 = std::min(b0 + chunk_blocks, (size_t) ih.n_blocks);
    if (chunk_b0.size() < max_chunks) chunk_b0.push_back(b0);
    max_c = std::max(max_c, (size_t) (ent[b1].coff - ent[b0].coff));
    max_u = std::max(max_u, (size_t) (ent[b1].uoff - ent[b0].uoff));
    max_r = std::max(max_r, (size_t) (ent[b1].rec  - ent[b0].rec));
  }
  auto chunk_end = [&](size_t b0) { return std::min(b0 + chunk_blocks, (size_t) ih.n_blocks); };
  size_t last_b1 = chunk_end(chunk_b0.back());
  uint64_t n_reads = ent[last_b1].rec;
  for (size_t b = 0; b < ih.n_blocks; b++)
    max_rec_per_blk = std::max(max_rec_per_blk, (size_t) (ent[b+1].rec - ent[b].rec));
  size_t max_ublk = 0;
  for (size_t b = 0; b < ih.n_blocks; b++)
    max_ublk = std::max(max_ublk, (size_t) (ent[b+1].uoff - ent[b].uoff));

  // A streamed run keeps every read on the device across the group loop, so
  // the expanded input and the per-read spans are sized for the whole input;
  // a resident run reuses one chunk's worth.
  size_t ubuf_bytes = streamed ? (size_t) ent[last_b1].uoff : max_u;
  size_t span_reads = streamed ? (size_t) n_reads : max_r;
  char *h_pin;  CK(cudaHostAlloc(&h_pin, max_c, cudaHostAllocDefault));
  char *d_cbuf; CK(cudaMalloc(&d_cbuf, max_c));
  char *d_ubuf; CK(cudaMalloc(&d_ubuf, ubuf_bytes));
  uint32_t *d_off, *d_len, *d_hoff, *d_hlen;
  CK(cudaMalloc(&d_off, span_reads*4)); CK(cudaMalloc(&d_len, span_reads*4));
  CK(cudaMalloc(&d_hoff, span_reads*4)); CK(cudaMalloc(&d_hlen, span_reads*4));
  uint32_t *d_call, *d_llen, *d_loff, *d_sums;
  CK(cudaMalloc(&d_call, max_r*4));
  CK(cudaMalloc(&d_llen, max_r*4)); CK(cudaMalloc(&d_loff, max_r*4));
  uint32_t *d_over = nullptr;
  uint8_t *d_amb = nullptr;
  uint32_t scan_blocks = (uint32_t) ((max_r + 255) / 256);
  CK(cudaMalloc(&d_sums, (scan_blocks + 1) * 4));

  // Lines are formatted into fixed-width slots, then packed.  The width is the
  // worst case: an identifier, the call, the length, and one hit-list run per
  // reported minimizer.
  const uint32_t TEXT_STRIDE = 2048;
  char *d_text = nullptr, *d_packed = nullptr, *h_out = nullptr;
  if (out_path) {
    CK(cudaMalloc(&d_text, (size_t) max_r * TEXT_STRIDE));
    CK(cudaMalloc(&d_packed, (size_t) max_r * TEXT_STRIDE / 4));
    CK(cudaHostAlloc(&h_out, (size_t) max_r * TEXT_STRIDE / 4, cudaHostAllocDefault));
    CK(cudaMalloc(&d_over, 4)); CK(cudaMemset(d_over, 0, 4));
    CK(cudaMalloc(&d_amb, max_r));
  }
  uint64_t *d_buoff, *d_brec;
  CK(cudaMalloc(&d_buoff, (chunk_blocks+1)*8));
  CK(cudaMalloc(&d_brec,  (chunk_blocks+1)*8));
  void **d_cptr, **d_uptr; size_t *d_cb, *d_ub, *d_ac; nvcompStatus_t *d_st;
  CK(cudaMalloc(&d_cptr, chunk_blocks*sizeof(void*)));
  CK(cudaMalloc(&d_uptr, chunk_blocks*sizeof(void*)));
  CK(cudaMalloc(&d_cb, chunk_blocks*8)); CK(cudaMalloc(&d_ub, chunk_blocks*8));
  CK(cudaMalloc(&d_ac, chunk_blocks*8)); CK(cudaMalloc(&d_st, chunk_blocks*sizeof(nvcompStatus_t)));
  std::vector<void*> h_cptr(chunk_blocks), h_uptr(chunk_blocks);
  std::vector<size_t> h_cb(chunk_blocks), h_ub(chunk_blocks);
  std::vector<uint32_t> h_call(ih.n_records);

  nvcompBatchedGzipDecompressOpts_t nopts = nvcompBatchedGzipDecompressDefaultOpts;
  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  size_t temp_bytes = 0;
  NK(nvcompBatchedGzipDecompressGetTempSizeAsync(chunk_blocks, max_ublk, nopts,
                                                 &temp_bytes, max_u));
  void *d_temp = nullptr;
  if (temp_bytes) CK(cudaMalloc(&d_temp, temp_bytes));

  size_t shmem = (4 * max_rec_per_blk + 16) * sizeof(uint32_t);
  printf("chunk: %zu blocks, %.0f MB compressed, %.0f MB expanded, %zu reads; "
         "parse shared %.1f KB\n", chunk_blocks, max_c/1e6, max_u/1e6, max_r, shmem/1024.0);

  int out_fd = -1;
  size_t out_bytes = 0, n_fallback = 0;
  kraken2::MinimizerScanner fb_scanner(k, l, ssm, true, tog, (int) o[6]);
  HostFallback fb_ctx{h_tab, h_par.data(), h_ext.data(), mahv, &fb_scanner};
  if (out_path) {
    out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    printf("writing output to %s\n", out_path);
  }

  // ---- stage timing ----
  // Device stages are timed by event pairs on the stream that runs them, host
  // stages by the clock.  A timer is read only after a synchronization that
  // follows its stop event.
  struct Timer {
    const char *name; cudaEvent_t a, b; double ms; bool pending;
    void start(cudaStream_t s) { CK(cudaEventRecord(a, s)); }
    void stop(cudaStream_t s) { CK(cudaEventRecord(b, s)); pending = true; }
    void collect() {
      if (!pending) return;
      float t; CK(cudaEventElapsedTime(&t, a, b)); ms += t; pending = false;
    }
  };
  enum { T_H2D, T_INFLATE, T_PARSE, T_BIN, T_CLASSIFY, T_OUTPUT, N_T };
  Timer tm[N_T] = {{"copy in"}, {"inflate"}, {"parse"}, {"bin minimizers by partition"},
                   {streamed ? "finish (scan+resolve+format)" : "classify (scan+probe+resolve+format)"},
                   {"pack + copy out"}};
  for (auto &t : tm) { CK(cudaEventCreate(&t.a)); CK(cudaEventCreate(&t.b)); t.ms = 0; t.pending = false; }
  auto collect_all = [&]() { for (auto &t : tm) t.collect(); };
  double t_read = 0, t_write = 0;

  // Reads chunk [b0, b1) from disk, expands it into ubuf and splits it into
  // reads; the spans are written at off.. (indexed from the chunk's first read).
  auto load_chunk = [&](size_t b0, char *ubuf, uint32_t *off, uint32_t *len,
                        uint32_t *hoff, uint32_t *hlen) {
    size_t b1 = chunk_end(b0), nb = b1 - b0;
    size_t clen = ent[b1].coff - ent[b0].coff;
    double tr = now_s();
    size_t got = 0;
    while (got < clen) {
      ssize_t n = pread(bgz, h_pin + got, clen - got, ent[b0].coff + got);
      if (n <= 0) break;
      got += n;
    }
    t_read += now_s() - tr;
    tm[T_H2D].start(stream);
    CK(cudaMemcpyAsync(d_cbuf, h_pin, clen, cudaMemcpyHostToDevice, stream));
    for (size_t i = 0; i < nb; i++) {
      h_cptr[i] = d_cbuf + (ent[b0+i].coff - ent[b0].coff);
      h_uptr[i] = ubuf + (ent[b0+i].uoff - ent[b0].uoff);
      h_cb[i]   = ent[b0+i+1].coff - ent[b0+i].coff;
      h_ub[i]   = ent[b0+i+1].uoff - ent[b0+i].uoff;
    }
    CK(cudaMemcpyAsync(d_cptr, h_cptr.data(), nb*sizeof(void*), cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_uptr, h_uptr.data(), nb*sizeof(void*), cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_cb, h_cb.data(), nb*8, cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_ub, h_ub.data(), nb*8, cudaMemcpyHostToDevice, stream));
    std::vector<uint64_t> buoff(nb+1), brec(nb+1);
    for (size_t i = 0; i <= nb; i++) { buoff[i] = ent[b0+i].uoff; brec[i] = ent[b0+i].rec; }
    CK(cudaMemcpyAsync(d_buoff, buoff.data(), (nb+1)*8, cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(d_brec,  brec.data(),  (nb+1)*8, cudaMemcpyHostToDevice, stream));
    tm[T_H2D].stop(stream);

    tm[T_INFLATE].start(stream);
    NK(nvcompBatchedGzipDecompressAsync((const void *const *) d_cptr, d_cb, d_ub,
        d_ac, nb, d_temp, temp_bytes, (void *const *) d_uptr, nopts, d_st, stream));
    tm[T_INFLATE].stop(stream);

    // Decompression reports per block, and ignoring that turns a decode
    // failure into a wild read three kernels later.  Checked on the first
    // chunk, which is where a wrong argument would show up.
    if (b0 == 0) {
      CK(cudaStreamSynchronize(stream));
      std::vector<nvcompStatus_t> st(nb);
      std::vector<size_t> ac(nb);
      CK(cudaMemcpy(st.data(), d_st, nb*sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(ac.data(), d_ac, nb*8, cudaMemcpyDeviceToHost));
      size_t nbad = 0, nshort = 0;
      for (size_t i = 0; i < nb; i++) {
        if (st[i] != nvcompSuccess) nbad++;
        if (ac[i] != h_ub[i]) nshort++;
      }
      printf("decompress: %zu blocks, %zu failed, %zu wrong size "
             "(first block %zu of %zu bytes)\n", nb, nbad, nshort, ac[0], h_ub[0]);
      if (nbad || nshort) { fprintf(stderr, "decompression did not agree with the index\n"); exit(2); }
    }

    tm[T_PARSE].start(stream);
    parse_blocks<<<(unsigned) std::min(nb, (size_t) 8192), 256, shmem, stream>>>(
        ubuf, d_buoff, d_brec, ent[b0].uoff, (uint32_t) nb, ent[b0].rec,
        off, len, hoff, hlen);
    tm[T_PARSE].stop(stream);
    // The pinned staging buffer is reused by the next chunk.
    CK(cudaStreamSynchronize(stream));
    collect_all();
  };

  auto emit = [&](const char *p, size_t nb) {
    size_t w = 0;
    while (w < nb) {
      ssize_t got = write(out_fd, p + w, nb - w);
      if (got <= 0) { perror("write"); exit(1); }
      w += got;
    }
    out_bytes += nb;
  };

  // Classifies (CLASSIFY) or finishes (FINISH) one loaded chunk, writes its
  // lines, and keeps its calls for the check against classify.
  uint64_t done_reads = 0;
  auto emit_chunk = [&](size_t b0, const char *ubuf, const uint32_t *off, const uint32_t *len,
                        const uint32_t *hoff, const uint32_t *hlen, StreamArgs sa) {
    size_t nrec = ent[chunk_end(b0)].rec - ent[b0].rec;
    if (out_path) CK(cudaMemsetAsync(d_amb, 0, nrec, stream));
    tm[T_CLASSIFY].start(stream);
    fused<<<8192, THREADS, 0, stream>>>(
        ubuf, off, len, (uint32_t) nrec, k, l, ssm, tog, mahv,
        d_tab, skip_probe, sa,
        nullptr, nullptr, nullptr, nullptr, d_parent, d_extid, d_call,
        hoff, hlen, d_text, TEXT_STRIDE, d_llen, d_over, d_amb);
    tm[T_CLASSIFY].stop(stream);

    if (out_path) {
      tm[T_OUTPUT].start(stream);
      uint32_t nblk = (uint32_t) ((nrec + 255) / 256);
      scan_pass1<<<nblk, 256, 0, stream>>>(d_llen, d_loff, d_sums, (uint32_t) nrec);
      scan_pass2<<<1, 1, 0, stream>>>(d_sums, nblk);
      scan_pass3<<<nblk, 256, 0, stream>>>(d_loff, d_sums, (uint32_t) nrec);
      compact_text<<<8192, 128, 0, stream>>>(d_text, TEXT_STRIDE, d_llen, d_loff,
                                             d_packed, (uint32_t) nrec);
      uint32_t last_off, last_len;
      CK(cudaMemcpyAsync(&last_off, d_loff + nrec - 1, 4, cudaMemcpyDeviceToHost, stream));
      CK(cudaMemcpyAsync(&last_len, d_llen + nrec - 1, 4, cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));
      size_t tbytes = (size_t) last_off + last_len;
      size_t cap = (size_t) max_r * TEXT_STRIDE / 4;
      if (tbytes > cap) {
        fprintf(stderr, "packed output %zu bytes exceeds the %zu reserved\n", tbytes, cap);
        exit(1);
      }
      CK(cudaMemcpyAsync(h_out, d_packed, tbytes, cudaMemcpyDeviceToHost, stream));
      tm[T_OUTPUT].stop(stream);
      CK(cudaStreamSynchronize(stream));
      double tw = now_s();
      // Reads the kernel declined contribute no bytes, so their line belongs
      // exactly at their offset.  Splice the host's version in there.
      std::vector<uint8_t> amb(nrec);
      CK(cudaMemcpy(amb.data(), d_amb, nrec, cudaMemcpyDeviceToHost));
      std::vector<uint32_t> fb;
      for (size_t i = 0; i < nrec; i++) if (amb[i]) fb.push_back((uint32_t) i);
      if (fb.empty()) emit(h_out, tbytes);
      else {
        std::vector<uint32_t> loff(nrec), soff(nrec), slen(nrec), ho(nrec), hl(nrec);
        CK(cudaMemcpy(loff.data(), d_loff, nrec*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(soff.data(), off,  nrec*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(slen.data(), len,  nrec*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ho.data(), hoff, nrec*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hl.data(), hlen, nrec*4, cudaMemcpyDeviceToHost));
        size_t pos = 0;
        std::vector<char> rb;
        for (uint32_t r : fb) {
          emit(h_out + pos, loff[r] - pos);
          pos = loff[r];
          rb.resize(hl[r] + slen[r]);
          CK(cudaMemcpy(rb.data(), ubuf + ho[r], hl[r], cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(rb.data() + hl[r], ubuf + soff[r], slen[r], cudaMemcpyDeviceToHost));
          std::string ln = fb_ctx.line(rb.data(), hl[r], rb.data() + hl[r], slen[r]);
          emit(ln.data(), ln.size());
          n_fallback++;
        }
        emit(h_out + pos, tbytes - pos);
      }
      t_write += now_s() - tw;
    }
    CK(cudaMemcpyAsync(h_call.data() + done_reads, d_call, nrec*4,
                       cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    collect_all();
    done_reads += nrec;
  };

  // ---- the pipeline ----
  CK(cudaDeviceSynchronize());
  double t0 = now_s();
  double t_in = 0, t_bin = 0, t_groups = 0, t_fin = 0;
  double copy_ms = 0, probe_ms = 0;
  size_t copy_bytes = 0;
  uint32_t tok_stride = 0;
  double lookups_per_read = 0;

  if (! streamed) {
    for (size_t b0 : chunk_b0) {
      load_chunk(b0, d_ubuf, d_off, d_len, d_hoff, d_hlen);
      emit_chunk(b0, d_ubuf, d_off, d_len, d_hoff, d_hlen, StreamArgs{});
    }
  } else {
    // Every chunk is loaded first and stays on the device.
    for (size_t b0 : chunk_b0) {
      uint64_t r0 = ent[b0].rec;
      load_chunk(b0, d_ubuf + ent[b0].uoff, d_off + r0, d_len + r0, d_hoff + r0, d_hlen + r0);
    }
    uint32_t *d_max; CK(cudaMalloc(&d_max, 4)); CK(cudaMemset(d_max, 0, 4));
    max_len<<<1024, 256, 0, stream>>>(d_len, (uint32_t) n_reads, d_max);
    uint32_t mlen; CK(cudaMemcpy(&mlen, d_max, 4, cudaMemcpyDeviceToHost));
    if (mlen > (uint32_t) MAX_LEN) mlen = MAX_LEN;
    tok_stride = mlen >= (uint32_t) k ? mlen - k + 1 : 1;
    if ((double) n_reads * tok_stride >= 4294967296.0) {
      fprintf(stderr, "too many k-mers for 32-bit token indices; use fewer chunks\n"); return 1;
    }
    // The state that survives the group loop: one taxon per k-mer position.
    uint16_t *d_tok;
    CK(cudaMalloc(&d_tok, (size_t) n_reads * tok_stride * 2));
    t_in = now_s() - t0;

    // Binned variant: the scan runs once and files each lookup under its
    // partition, one set of bins per chunk.  A bin holds up to bin_cap
    // entries; the cap assumes at most bin_frac lookups per k-mer position
    // and the run stops if any bin overflows.
    std::vector<uint64_t *> bh(chunk_b0.size(), nullptr);
    std::vector<uint32_t *> bi(chunk_b0.size(), nullptr), bc(chunk_b0.size(), nullptr);
    std::vector<uint32_t> bcap(chunk_b0.size(), 0);
    if (binned) {
      double tb = now_s();
      uint64_t total = 0;
      uint32_t P = (uint32_t) d_tab.G.P;
      for (size_t c = 0; c < chunk_b0.size(); c++) {
        size_t b0 = chunk_b0[c];
        uint64_t r0 = ent[b0].rec, nrec = ent[chunk_end(b0)].rec - r0;
        double mean = (double) nrec * tok_stride * bin_frac / P;
        bcap[c] = (uint32_t) (mean + 8 * sqrt(mean) + 1024);
        CK(cudaMalloc(&bh[c], (size_t) P * bcap[c] * 8));
        CK(cudaMalloc(&bi[c], (size_t) P * bcap[c] * 4));
        CK(cudaMalloc(&bc[c], P * 4)); CK(cudaMemset(bc[c], 0, P * 4));
        StreamArgs sa{BIN, d_tok, r0 * tok_stride, tok_stride, 0, 0, bh[c], bi[c], bc[c], bcap[c]};
        tm[T_BIN].start(stream);
        fused<<<8192, THREADS, 0, stream>>>(
            d_ubuf + ent[b0].uoff, d_off + r0, d_len + r0, (uint32_t) nrec, k, l, ssm, tog, mahv,
            d_tab, 0, sa, nullptr, nullptr, nullptr, nullptr, d_parent, d_extid, d_call,
            nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr);
        tm[T_BIN].stop(stream);
        std::vector<uint32_t> cnt(P);
        CK(cudaMemcpyAsync(cnt.data(), bc[c], P * 4, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        collect_all();
        for (uint32_t b = 0; b < P; b++) {
          if (cnt[b] > bcap[c]) {
            fprintf(stderr, "bin %u of chunk %zu overflowed (%u > %u); raise K2GPU_BIN_FRAC\n",
                    b, c, cnt[b], bcap[c]);
            return 1;
          }
          total += cnt[b];
        }
      }
      lookups_per_read = (double) total / n_reads;
      t_bin = now_s() - tb;
    }

    // The group loop.  Group g is partitions [g P / G, (g + 1) P / G), copied
    // from pinned host memory into slot g mod 2 on its own stream while the
    // previous group is probed.  A slot is refilled only after the probes
    // that read it are done.  With K2GPU_SERIAL the copy runs on the compute
    // stream instead, so nothing overlaps.
    double tg = now_s();
    uint32_t P = (uint32_t) d_tab.G.P, per = P / n_groups;
    size_t group_bytes = (size_t) per * d_tab.G.nb * 64;
    cudaStream_t cs = stream;
    if (! serial) CK(cudaStreamCreate(&cs));
    std::vector<cudaEvent_t> c0(n_groups), c1(n_groups), p0(n_groups), p1(n_groups);
    for (uint32_t g = 0; g < n_groups; g++) {
      CK(cudaEventCreate(&c0[g])); CK(cudaEventCreate(&c1[g]));
      CK(cudaEventCreate(&p0[g])); CK(cudaEventCreate(&p1[g]));
    }
    auto issue_copy = [&](uint32_t g) {
      int s = g % n_slots;
      if (g >= (uint32_t) n_slots) CK(cudaStreamWaitEvent(cs, p1[g - n_slots], 0));
      CK(cudaEventRecord(c0[g], cs));
      CK(cudaMemcpyAsync(d_slot[s], h_buckets + (size_t) g * group_bytes, group_bytes,
                         cudaMemcpyHostToDevice, cs));
      CK(cudaEventRecord(c1[g], cs));
    };
    for (uint32_t g = 0; g < (uint32_t) n_slots && g < n_groups; g++) issue_copy(g);
    for (uint32_t g = 0; g < n_groups; g++) {
      int s = g % n_slots;
      CK(cudaStreamWaitEvent(stream, c1[g], 0));
      k2t::Table gt = d_tab;
      gt.buckets = d_slot[s];
      gt.part_base = (uint64_t) g * per;
      CK(cudaEventRecord(p0[g], stream));
      for (size_t c = 0; c < chunk_b0.size(); c++) {
        size_t b0 = chunk_b0[c];
        uint64_t r0 = ent[b0].rec, nrec = ent[chunk_end(b0)].rec - r0;
        if (binned) {
          uint64_t n = (uint64_t) per * bcap[c];
          unsigned blocks = (unsigned) std::min<uint64_t>((n + 255) / 256, 1u << 20);
          probe_bins<<<blocks, 256, 0, stream>>>(gt, bh[c], bi[c], bc[c], bcap[c],
                                                 g * per, (g + 1) * per, d_tok);
        } else {
          StreamArgs sa{PROBE_GROUP, d_tok, r0 * tok_stride, tok_stride, g * per, (g + 1) * per,
                        nullptr, nullptr, nullptr, 0};
          fused<<<8192, THREADS, 0, stream>>>(
              d_ubuf + ent[b0].uoff, d_off + r0, d_len + r0, (uint32_t) nrec, k, l, ssm, tog, mahv,
              gt, 0, sa, nullptr, nullptr, nullptr, nullptr, d_parent, d_extid, d_call,
              nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr);
        }
      }
      CK(cudaEventRecord(p1[g], stream));
      if (g + n_slots < n_groups) issue_copy(g + n_slots);
    }
    CK(cudaDeviceSynchronize());
    t_groups = now_s() - tg;
    for (uint32_t g = 0; g < n_groups; g++) {
      float a, b;
      CK(cudaEventElapsedTime(&a, c0[g], c1[g])); copy_ms += a;
      CK(cudaEventElapsedTime(&b, p0[g], p1[g])); probe_ms += b;
    }
    copy_bytes = (size_t) n_groups * group_bytes;

    double tf = now_s();
    for (size_t b0 : chunk_b0) {
      uint64_t r0 = ent[b0].rec;
      StreamArgs sa{FINISH, d_tok, r0 * tok_stride, tok_stride, 0, 0, nullptr, nullptr, nullptr, 0};
      emit_chunk(b0, d_ubuf + ent[b0].uoff, d_off + r0, d_len + r0, d_hoff + r0, d_hlen + r0, sa);
    }
    t_fin = now_s() - tf;
  }
  double el = now_s() - t0;

  if (out_fd >= 0) { close(out_fd); }
  printf("\nend to end   %.3f s   %.2f Gbase/s   %.1f Mread/s   %.2f Mread total\n",
         el, ih.total_uncompressed * 0.4 / el / 1e9, done_reads / el / 1e6,
         done_reads / 1e6);
  printf("             (bases are about 40%% of the FASTQ bytes)\n");
  printf("stage times (s):\n  %-40s %8.3f\n", "read input (host)", t_read);
  for (int i = 0; i < N_T; i++)
    if (tm[i].ms > 0) printf("  %-40s %8.3f\n", tm[i].name, tm[i].ms / 1e3);
  printf("  %-40s %8.3f\n", "write output + fallback (host)", t_write);
  if (streamed) {
    printf("streamed: %u groups of %u partitions (%.3f GB each), %s%s\n", n_groups,
           (uint32_t) d_tab.G.P / n_groups, table_bytes / 1e9 / n_groups,
           binned ? "binned" : "recomputed", serial ? ", serial" : ", overlapped");
    printf("  %-40s %8.3f\n", "load all chunks (wall)", t_in);
    if (binned) printf("  %-40s %8.3f   (%.1f lookups per read)\n", "bin (wall)", t_bin, lookups_per_read);
    printf("  %-40s %8.3f\n", "group loop (wall)", t_groups);
    printf("  %-40s %8.3f   (%.2f GB, %.1f GB/s)\n", "  table copies (sum)", copy_ms / 1e3,
           copy_bytes / 1e9, copy_bytes / 1e9 / (copy_ms / 1e3));
    printf("  %-40s %8.3f\n", "  probes (sum)", probe_ms / 1e3);
    printf("  %-40s %8.3f\n", "finish (wall)", t_fin);
    printf("per-read state: %u bytes of k-mer taxa", tok_stride * 2);
    if (binned) printf(" + %.0f bytes of binned minimizers", lookups_per_read * 12);
    printf("\n");
  }
  if (out_bytes) {
    uint32_t over = 0;
    CK(cudaMemcpy(&over, d_over, 4, cudaMemcpyDeviceToHost));
    printf("             wrote %.2f GB of output, %u truncated, "
           "%zu reads via the host fallback\n", out_bytes / 1e9, over, n_fallback);
    if (over) { fprintf(stderr, "some hit lists did not fit the slot\n"); return 1; }
  }

  // ---- correctness, against classify's own calls ----
  FILE *rf = fopen(argv[6], "rb");
  if (!rf) { fprintf(stderr, "cannot open %s\n", argv[6]); return 1; }
  size_t bad = 0, seen = 0, classified = 0;
  char line[8192];
  for (uint64_t i = 0; i < done_reads && fgets(line, sizeof line, rf); i++) {
    unsigned long want = 0;
    char *p1 = strchr(line, '\t');
    char *p2 = p1 ? strchr(p1 + 1, '\t') : nullptr;
    if (p2) want = strtoul(p2 + 1, nullptr, 10);
    seen++;
    if (want) classified++;
    if ((unsigned long) h_call[i] != want) {
      if (bad < 3) fprintf(stderr, "read %" PRIu64 ": classify %lu, gpu %u\n", i, want, h_call[i]);
      bad++;
    }
  }
  fclose(rf);
  printf("verified %zu calls against classify (%.2f%% classified): %zu mismatched\n",
         seen, seen ? 100.0*classified/seen : 0.0, bad);
  return bad ? 2 : 0;
}
