// End to end: record-aligned BGZF in, one classification call per read out,
// with nothing touching the host in between.
//
// Every earlier program measured a stage.  This one runs the whole thing as a
// single pipeline, which is the only way to find out what the stage figures add
// up to in practice: compressed blocks are sent to the card, expanded there,
// split into reads there, classified there, and only one identifier per read
// comes back.  The reads themselves never exist in host memory, with one
// exception: a unit the kernel cannot handle (a read longer than MAX_LEN, a
// line longer than its output slot, or more distinct taxa than the kernel
// counts) is fetched from the device and classified on the host with the
// same scanner, resolution and formatting code (classify_core.h), and its
// line is merged into the chunk's output at its place.
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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <memory>
#include <sched.h>
#include <unordered_map>

#define CK(x) do { cudaError_t e=(x); if (e!=cudaSuccess) { \
  fprintf(stderr,"cuda %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)
#define NK(x) do { nvcompStatus_t st=(x); if (st!=nvcompSuccess) { \
  fprintf(stderr,"nvcomp %d at %d\n",(int)st,__LINE__); exit(1);} } while(0)

struct IdxHeader {
  char magic[8]; uint32_t version, block_target;
  uint64_t n_blocks, n_records, total_uncompressed;
};
struct IdxEnt { uint64_t coff, uoff, rec; };

static double now_s() {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

// What the host needs to classify a unit the device left to it: the scan
// parameters, the host view of the table and the taxonomy columns.
struct HostCtx {
  k2t::Table tab;
  int k, l; uint64_t ssm, tog, mahv; int skip_probe;
  const uint64_t *parent, *extid;
};

// Classifies one unit as classify does, with the shared scanner, resolution
// and formatting of classify_core.h, but with no bound on read length, on the
// number of distinct taxa or on the length of the line.  Every mate goes
// through k2c::Scanner, which is Kraken 2's own scanner restated, so a mate
// without ambiguous bases gets the tokens the device's parallel scan would.
// Returns the call as an external ID and leaves the line in *line.
static uint32_t classify_host(const HostCtx &c, int mates, const std::string seq[2],
                              const std::string &hdr, std::vector<uint32_t> &tok,
                              std::string *line) {
  tok.clear();
  for (int m = 0; m < mates; m++) {
    k2c::Scanner sc;
    sc.init(c.k, c.l, c.ssm, c.tog);
    sc.load(seq[m].data(), (int) seq[m].size());
    uint64_t mz, last_min = ~0ULL; bool amb;
    uint32_t last_taxon = 0;
    while (sc.next(&mz, &amb)) {
      uint32_t taxon;
      if (amb) taxon = k2c::AMBIG;
      else if (mz != last_min) {
        last_min = mz;
        if (c.mahv && k2t::murmur(mz) < c.mahv) taxon = 0;
        else taxon = c.skip_probe ? 0u : c.tab.get(mz);
        last_taxon = taxon;
      } else taxon = last_taxon;
      tok.push_back(taxon);
    }
    if (mates == 2 && m == 0) tok.push_back(k2c::MATE);
  }
  std::unordered_map<uint32_t, uint32_t> cnt;
  for (uint32_t t : tok) if (t && t < k2c::AMBIG) cnt[t]++;
  std::vector<uint32_t> tx, ct;
  for (auto &e : cnt) { tx.push_back(e.first); ct.push_back(e.second); }
  uint32_t call = k2c::resolve(c.parent, tx.data(), ct.data(), (int) tx.size());
  if (line) {
    // A run is at most two 20-digit numbers and three separators, and the
    // formatter checks the end only between runs, hence the margin.
    size_t cap = hdr.size() + 256 + (tok.size() + 1) * 48;
    line->resize(cap);
    bool over = false;
    int n = k2c::format_line(&(*line)[0], &(*line)[0] + cap - 64, call, c.extid,
                             hdr.data(), (uint32_t) hdr.size(), mates == 2,
                             (uint32_t) seq[0].size(), mates == 2 ? (uint32_t) seq[1].size() : 0u,
                             tok.data(), (int) tok.size(), &over);
    if (over) { fprintf(stderr, "host line buffer too small\n"); exit(1); }
    line->resize(n);
  }
  return (uint32_t) (call ? c.extid[call] : 0);
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
  // K2GPU_BIN_NOWRITE=1 keeps the binned lookups but not the stores of
  // their results, to separate the two costs.  Output is wrong.
  int bin_nowrite = env_int("K2GPU_BIN_NOWRITE", 0);
  // K2GPU_BATCH_RECORDS bounds how many records a streamed run keeps on the
  // device at once (default 20M).
  uint64_t batch_records = getenv("K2GPU_BATCH_RECORDS")
      ? (uint64_t) atoll(getenv("K2GPU_BATCH_RECORDS")) : 20000000ULL;

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
    k2t::BlockedHeader bh;
    if (const char *e = k2t::map_blocked(hmap, hst.st_size, h_tab, bh)) {
      fprintf(stderr, "%s: %s\n", argv[3], e); return 1;
    }
    // K2GPU_SCAN: 0 reads each bucket into registers once (the default), 1
    // does the same and stops at the first remainder >= the query's, 2 reloads
    // each slot.
    h_tab.scan_mode = env_int("K2GPU_SCAN", k2t::Table::REGS_FULL);
    k2t::BlockedLayout L = k2t::blocked_layout(bh);
    size_t nbk = L.stash_off - L.buckets;
    blk_bucket_off = L.buckets;
    d_tab = h_tab;
    // The stash and the code-to-taxon array are small and stay resident in
    // both modes.
    uint64_t *dso, *dsk = nullptr; uint32_t *dsv = nullptr, *dct = nullptr;
    CK(cudaMalloc(&dso, (h_tab.G.P + 1) * 8));
    upload(dso, L.stash_off, (h_tab.G.P + 1) * 8);
    if (bh.stash_entries) {
      CK(cudaMalloc(&dsk, bh.stash_entries * 8)); upload(dsk, L.stash_key, bh.stash_entries * 8);
      CK(cudaMalloc(&dsv, bh.stash_entries * 4)); upload(dsv, L.stash_val, bh.stash_entries * 4);
    }
    if (h_tab.code_taxa) {
      CK(cudaMalloc(&dct, (bh.taxon_codes + 1) * 4));
      upload(dct, L.code_taxa, (bh.taxon_codes + 1) * 4);
    }
    d_tab.stash_off = dso; d_tab.stash_key = dsk; d_tab.stash_val = dsv;
    d_tab.code_taxa = dct;
    table_bytes = nbk;
    printf("table: blocked version %u, %" PRIu64 " partitions of %" PRIu64 " buckets, "
           "%d slots of %d bits, %d-bit spill filter, stash %" PRIu64 ", %" PRIu64
           " taxon codes\n", bh.version, h_tab.G.P, h_tab.G.nb, h_tab.G.S, h_tab.G.w,
           h_tab.G.sf, bh.stash_entries, bh.taxon_codes);
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
              "of K2GPU_GROUPS and whose stored values fit 16 bits\n");
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

  // ---- chunks, and batches of chunks for a streamed table ----
  // A version 2 index holds interleaved mate pairs, every block a whole
  // number of pairs; a unit is then a pair of consecutive records.
  int mates = ih.version == 2 ? 2 : 1;
  std::vector<size_t> chunk_b0;
  size_t max_c = 0, max_u = 0, max_r = 0, max_rec_per_blk = 0;
  for (size_t b0 = 0; b0 < ih.n_blocks; b0 += chunk_blocks) {
    size_t b1 = std::min(b0 + chunk_blocks, (size_t) ih.n_blocks);
    if (chunk_b0.size() < max_chunks) chunk_b0.push_back(b0);
    max_c = std::max(max_c, (size_t) (ent[b1].coff - ent[b0].coff));
    max_u = std::max(max_u, (size_t) (ent[b1].uoff - ent[b0].uoff));
    max_r = std::max(max_r, (size_t) (ent[b1].rec  - ent[b0].rec));
  }
  for (size_t b = 0; b <= ih.n_blocks; b++)
    if (ent[b].rec % mates) { fprintf(stderr, "block %zu does not start at a pair\n", b); return 1; }
  auto chunk_end = [&](size_t b0) { return std::min(b0 + chunk_blocks, (size_t) ih.n_blocks); };
  const size_t nchunks = chunk_b0.size();
  auto rec0 = [&](size_t i) { return (uint64_t) ent[chunk_b0[i]].rec; };
  auto nrec = [&](size_t i) { return (uint64_t) (ent[chunk_end(chunk_b0[i])].rec - ent[chunk_b0[i]].rec); };
  uint64_t n_recs = ent[chunk_end(chunk_b0.back())].rec, n_units = n_recs / mates;
  size_t max_units = max_r / mates;
  for (size_t b = 0; b < ih.n_blocks; b++)
    max_rec_per_blk = std::max(max_rec_per_blk, (size_t) (ent[b+1].rec - ent[b].rec));
  size_t max_ublk = 0;
  for (size_t b = 0; b < ih.n_blocks; b++)
    max_ublk = std::max(max_ublk, (size_t) (ent[b+1].uoff - ent[b].uoff));
  printf("input: %s\n", mates == 2 ? "interleaved pairs" : "single-end reads");

  // A streamed table needs every read of a batch on the device for the
  // whole group loop, so the reads go through in batches of whole chunks of
  // at most batch_records records; each batch pays one copy of the table.
  std::vector<size_t> batch_first;
  size_t max_batch_recs = 0, max_batch_u = 0;
  if (streamed) {
    for (size_t i = 0; i < nchunks; i++) {
      if (batch_first.empty() ||
          rec0(i) + nrec(i) - rec0(batch_first.back()) > batch_records)
        batch_first.push_back(i);
    }
    batch_first.push_back(nchunks);
    for (size_t b = 0; b + 1 < batch_first.size(); b++) {
      size_t i0 = batch_first[b], i1 = batch_first[b + 1];
      max_batch_recs = std::max(max_batch_recs, (size_t) (rec0(i1 - 1) + nrec(i1 - 1) - rec0(i0)));
      max_batch_u = std::max(max_batch_u, (size_t) (ent[chunk_end(chunk_b0[i1 - 1])].uoff
                                                    - ent[chunk_b0[i0]].uoff));
    }
    printf("streamed reads: %zu batches of at most %zu records\n",
           batch_first.size() - 1, max_batch_recs);
  }

  // ---- device and pinned buffers ----
  // Two sets, each with its own stream, so one chunk can be expanded and
  // classified while the previous one is copied back and written.
  const uint32_t TEXT_STRIDE = mates == 2 ? 4096 : 2048;
  const size_t out_cap = (size_t) max_units * TEXT_STRIDE / 4;
  nvcompBatchedGzipDecompressOpts_t nopts = nvcompBatchedGzipDecompressDefaultOpts;
  size_t temp_bytes = 0;
  NK(nvcompBatchedGzipDecompressGetTempSizeAsync(chunk_blocks, max_ublk, nopts,
                                                 &temp_bytes, max_u));
  struct Timer {
    cudaEvent_t a, b; double ms; bool pending;
    void init() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); ms = 0; pending = false; }
    void start(cudaStream_t s) { CK(cudaEventRecord(a, s)); }
    void stop(cudaStream_t s) { CK(cudaEventRecord(b, s)); pending = true; }
    void collect() {
      if (!pending) return;
      CK(cudaEventSynchronize(b));
      float t; CK(cudaEventElapsedTime(&t, a, b)); ms += t; pending = false;
    }
  };
  enum { T_H2D, T_INFLATE, T_PARSE, T_CLASSIFY, T_OUTPUT, N_T };
  static const char *stage_name[N_T] = {"copy in", "inflate", "parse",
      "classify or finish (scan+resolve+format)", "pack output"};
  struct Set {
    cudaStream_t st;
    char *cbuf, *ubuf; uint32_t *off, *len, *hoff, *hlen;
    uint32_t *call, *llen, *loff, *sums; char *text, *packed;
    void **cptr, **uptr; size_t *cb, *ub, *ac; nvcompStatus_t *stat; uint64_t *buoff, *brec;
    void *temp;
    void **h_cptr, **h_uptr; size_t *h_cb, *h_ub; uint64_t *h_buoff, *h_brec; uint32_t *h_last;
    cudaEvent_t staged, sized; bool staged_rec;
    Timer tm[N_T];
    // Units of the chunk the device left to the host, and where the chunk's
    // reads and spans are, so the host can fetch them.
    uint32_t *hunits, *nhost, *h_nhost;
    const char *c_ubuf; const uint32_t *c_off, *c_len, *c_hoff, *c_hlen;
  } set[2];
  for (Set &S : set) {
    CK(cudaStreamCreate(&S.st));
    CK(cudaMalloc(&S.cbuf, max_c));
    S.ubuf = nullptr; S.off = S.len = S.hoff = S.hlen = nullptr;
    if (!streamed) {
      CK(cudaMalloc(&S.ubuf, max_u));
      CK(cudaMalloc(&S.off, max_r*4)); CK(cudaMalloc(&S.len, max_r*4));
      CK(cudaMalloc(&S.hoff, max_r*4)); CK(cudaMalloc(&S.hlen, max_r*4));
    }
    CK(cudaMalloc(&S.call, max_units*4));
    CK(cudaMalloc(&S.llen, max_units*4)); CK(cudaMalloc(&S.loff, max_units*4));
    CK(cudaMalloc(&S.sums, ((max_units + 255) / 256 + 1) * 4));
    S.text = S.packed = nullptr;
    if (out_path) {
      CK(cudaMalloc(&S.text, (size_t) max_units * TEXT_STRIDE));
      CK(cudaMalloc(&S.packed, out_cap));
    }
    CK(cudaMalloc(&S.cptr, chunk_blocks*sizeof(void*))); CK(cudaMalloc(&S.uptr, chunk_blocks*sizeof(void*)));
    CK(cudaMalloc(&S.cb, chunk_blocks*8)); CK(cudaMalloc(&S.ub, chunk_blocks*8));
    CK(cudaMalloc(&S.ac, chunk_blocks*8)); CK(cudaMalloc(&S.stat, chunk_blocks*sizeof(nvcompStatus_t)));
    CK(cudaMalloc(&S.buoff, (chunk_blocks+1)*8)); CK(cudaMalloc(&S.brec, (chunk_blocks+1)*8));
    S.temp = nullptr;
    if (temp_bytes) CK(cudaMalloc(&S.temp, temp_bytes));
    CK(cudaHostAlloc(&S.h_cptr, chunk_blocks*sizeof(void*), 0));
    CK(cudaHostAlloc(&S.h_uptr, chunk_blocks*sizeof(void*), 0));
    CK(cudaHostAlloc(&S.h_cb, chunk_blocks*8, 0)); CK(cudaHostAlloc(&S.h_ub, chunk_blocks*8, 0));
    CK(cudaHostAlloc(&S.h_buoff, (chunk_blocks+1)*8, 0)); CK(cudaHostAlloc(&S.h_brec, (chunk_blocks+1)*8, 0));
    CK(cudaHostAlloc(&S.h_last, 8, 0));
    CK(cudaEventCreateWithFlags(&S.staged, cudaEventDisableTiming)); S.staged_rec = false;
    CK(cudaEventCreateWithFlags(&S.sized, cudaEventDisableTiming));
    for (auto &t : S.tm) t.init();
    CK(cudaMalloc(&S.hunits, max_units*4)); CK(cudaMalloc(&S.nhost, 4));
    CK(cudaHostAlloc(&S.h_nhost, 4, 0));
  }
  uint32_t *d_over;
  CK(cudaMalloc(&d_over, 4)); CK(cudaMemset(d_over, 0, 4));
  uint32_t *h_call; CK(cudaHostAlloc(&h_call, (size_t) n_units * 4, 0));

  // Streamed: the batch's reads, spans and per-token state.
  char *d_bubuf = nullptr; uint32_t *d_boff = nullptr, *d_blen = nullptr, *d_bhoff = nullptr, *d_bhlen = nullptr;
  if (streamed) {
    CK(cudaMalloc(&d_bubuf, max_batch_u));
    CK(cudaMalloc(&d_boff, max_batch_recs*4)); CK(cudaMalloc(&d_blen, max_batch_recs*4));
    CK(cudaMalloc(&d_bhoff, max_batch_recs*4)); CK(cudaMalloc(&d_bhlen, max_batch_recs*4));
  }

  size_t shmem = (4 * max_rec_per_blk + 16) * sizeof(uint32_t);
  printf("chunk: %zu blocks, %.0f MB compressed, %.0f MB expanded, %zu records; %zu chunks\n",
         chunk_blocks, max_c/1e6, max_u/1e6, max_r, nchunks);

  int out_fd = -1;
  size_t out_bytes = 0;
  if (out_path) {
    out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    printf("writing output to %s\n", out_path);
  }

  // ---- host threads: input reader and output writer ----
  // The reader fills a ring of pinned buffers with the compressed bytes of
  // each chunk, in order, ahead of the device.  A slot is refilled only after
  // the copy out of it has finished.  The writer writes each chunk's packed
  // output from a ring of pinned buffers once its copy back has finished.
  std::mutex mu;
  std::condition_variable cv;
  const int NR = 3, NW = 3;
  struct InSlot { char *buf; long chunk; cudaEvent_t used; bool recorded; } in[NR];
  for (auto &sl : in) {
    CK(cudaHostAlloc(&sl.buf, max_c, 0)); sl.chunk = -1; sl.recorded = false;
    CK(cudaEventCreateWithFlags(&sl.used, cudaEventDisableTiming));
  }
  char *outbuf[NW]; cudaEvent_t outev[NW]; bool out_busy[NW];
  for (int w = 0; w < NW; w++) {
    outbuf[w] = nullptr; out_busy[w] = false;
    if (out_path) CK(cudaHostAlloc(&outbuf[w], out_cap, 0));
    CK(cudaEventCreateWithFlags(&outev[w], cudaEventDisableTiming));
  }
  // A job writes either a pinned buffer w or, for a chunk the host had a
  // hand in, its own merged bytes.
  struct OutJob { int w; size_t bytes; std::shared_ptr<std::vector<char>> data; };
  std::deque<OutJob> wq;
  bool wdone = false;
  double t_read = 0, t_write = 0, t_wait_in = 0, t_wait_out = 0;

  std::thread reader([&]() {
    for (size_t i = 0; i < nchunks; i++) {
      InSlot &sl = in[i % NR];
      { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return sl.chunk == -1; }); }
      if (sl.recorded) CK(cudaEventSynchronize(sl.used));
      size_t b0 = chunk_b0[i], clen = ent[chunk_end(b0)].coff - ent[b0].coff;
      double t = now_s();
      for (size_t got = 0; got < clen; ) {
        ssize_t n = pread(bgz, sl.buf + got, clen - got, ent[b0].coff + got);
        if (n <= 0) { fprintf(stderr, "short read of input\n"); exit(1); }
        got += n;
      }
      t_read += now_s() - t;
      { std::lock_guard<std::mutex> lk(mu); sl.chunk = (long) i; }
      cv.notify_all();
    }
  });
  std::thread writer([&]() {
    while (true) {
      OutJob j;
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !wq.empty() || wdone; });
        if (wq.empty()) break;
        j = wq.front(); wq.pop_front();
      }
      if (j.w >= 0) CK(cudaEventSynchronize(outev[j.w]));
      const char *src = j.w >= 0 ? outbuf[j.w] : j.data->data();
      double t = now_s();
      for (size_t w = 0; w < j.bytes; ) {
        ssize_t got = write(out_fd, src + w, j.bytes - w);
        if (got <= 0) { perror("write"); exit(1); }
        w += got;
      }
      t_write += now_s() - t;
      { std::lock_guard<std::mutex> lk(mu); out_bytes += j.bytes; if (j.w >= 0) out_busy[j.w] = false; }
      cv.notify_all();
    }
  });

  // ---- per-chunk work ----
  // Copies chunk i into the device, expands it into ubuf and splits it into
  // records; spans are written at off.. (indexed from the chunk's first record).
  auto enqueue_load = [&](size_t i, Set &S, char *ubuf, uint32_t *off, uint32_t *len,
                          uint32_t *hoff, uint32_t *hlen) {
    for (auto &t : S.tm) t.collect();
    size_t b0 = chunk_b0[i], b1 = chunk_end(b0), nb = b1 - b0;
    size_t clen = ent[b1].coff - ent[b0].coff;
    InSlot &sl = in[i % NR];
    double tw = now_s();
    { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return sl.chunk == (long) i; }); }
    t_wait_in += now_s() - tw;
    if (S.staged_rec) CK(cudaEventSynchronize(S.staged));
    S.tm[T_H2D].start(S.st);
    CK(cudaMemcpyAsync(S.cbuf, sl.buf, clen, cudaMemcpyHostToDevice, S.st));
    CK(cudaEventRecord(sl.used, S.st)); sl.recorded = true;
    { std::lock_guard<std::mutex> lk(mu); sl.chunk = -1; }
    cv.notify_all();
    for (size_t k = 0; k < nb; k++) {
      S.h_cptr[k] = S.cbuf + (ent[b0+k].coff - ent[b0].coff);
      S.h_uptr[k] = ubuf + (ent[b0+k].uoff - ent[b0].uoff);
      S.h_cb[k]   = ent[b0+k+1].coff - ent[b0+k].coff;
      S.h_ub[k]   = ent[b0+k+1].uoff - ent[b0+k].uoff;
    }
    for (size_t k = 0; k <= nb; k++) { S.h_buoff[k] = ent[b0+k].uoff; S.h_brec[k] = ent[b0+k].rec; }
    CK(cudaMemcpyAsync(S.cptr, S.h_cptr, nb*sizeof(void*), cudaMemcpyHostToDevice, S.st));
    CK(cudaMemcpyAsync(S.uptr, S.h_uptr, nb*sizeof(void*), cudaMemcpyHostToDevice, S.st));
    CK(cudaMemcpyAsync(S.cb, S.h_cb, nb*8, cudaMemcpyHostToDevice, S.st));
    CK(cudaMemcpyAsync(S.ub, S.h_ub, nb*8, cudaMemcpyHostToDevice, S.st));
    CK(cudaMemcpyAsync(S.buoff, S.h_buoff, (nb+1)*8, cudaMemcpyHostToDevice, S.st));
    CK(cudaMemcpyAsync(S.brec, S.h_brec, (nb+1)*8, cudaMemcpyHostToDevice, S.st));
    CK(cudaEventRecord(S.staged, S.st)); S.staged_rec = true;
    S.tm[T_H2D].stop(S.st);

    S.tm[T_INFLATE].start(S.st);
    NK(nvcompBatchedGzipDecompressAsync((const void *const *) S.cptr, S.cb, S.ub,
        S.ac, nb, S.temp, temp_bytes, (void *const *) S.uptr, nopts, S.stat, S.st));
    S.tm[T_INFLATE].stop(S.st);
    // Decompression reports per block, and ignoring that turns a decode
    // failure into a wild read three kernels later.  Checked on the first
    // chunk, which is where a wrong argument would show up.
    if (i == 0) {
      CK(cudaStreamSynchronize(S.st));
      std::vector<nvcompStatus_t> st(nb);
      std::vector<size_t> ac(nb);
      CK(cudaMemcpy(st.data(), S.stat, nb*sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(ac.data(), S.ac, nb*8, cudaMemcpyDeviceToHost));
      size_t nbad = 0, nshort = 0;
      for (size_t k = 0; k < nb; k++) {
        if (st[k] != nvcompSuccess) nbad++;
        if (ac[k] != S.h_ub[k]) nshort++;
      }
      printf("decompress: %zu blocks, %zu failed, %zu wrong size\n", nb, nbad, nshort);
      if (nbad || nshort) { fprintf(stderr, "decompression did not agree with the index\n"); exit(2); }
    }
    S.tm[T_PARSE].start(S.st);
    parse_blocks<<<(unsigned) std::min(nb, (size_t) 8192), 256, shmem, S.st>>>(
        ubuf, S.buoff, S.brec, ent[b0].uoff, (uint32_t) nb, ent[b0].rec, off, len, hoff, hlen);
    S.tm[T_PARSE].stop(S.st);
  };

  // Classifies (CLASSIFY) or finishes (FINISH) a loaded chunk and packs its
  // lines; the calls go straight to their place in h_call.
  auto enqueue_classify = [&](size_t i, Set &S, const char *ubuf, const uint32_t *off,
                              const uint32_t *len, const uint32_t *hoff, const uint32_t *hlen,
                              StreamArgs sa) {
    uint64_t nu = nrec(i) / mates, u0 = rec0(i) / mates;
    S.c_ubuf = ubuf; S.c_off = off; S.c_len = len; S.c_hoff = hoff; S.c_hlen = hlen;
    CK(cudaMemsetAsync(S.nhost, 0, 4, S.st));
    S.tm[T_CLASSIFY].start(S.st);
    fused<<<8192, THREADS, 0, S.st>>>(
        ubuf, off, len, (uint32_t) nu, mates, k, l, ssm, tog, mahv,
        d_tab, skip_probe, sa, d_parent, d_extid, S.call,
        hoff, hlen, S.text, TEXT_STRIDE, S.llen, d_over, S.hunits, S.nhost);
    S.tm[T_CLASSIFY].stop(S.st);
    CK(cudaMemcpyAsync(S.h_nhost, S.nhost, 4, cudaMemcpyDeviceToHost, S.st));
    if (out_path) {
      S.tm[T_OUTPUT].start(S.st);
      uint32_t nblk = (uint32_t) ((nu + 255) / 256);
      scan_pass1<<<nblk, 256, 0, S.st>>>(S.llen, S.loff, S.sums, (uint32_t) nu);
      scan_pass2<<<1, 1, 0, S.st>>>(S.sums, nblk);
      scan_pass3<<<nblk, 256, 0, S.st>>>(S.loff, S.sums, (uint32_t) nu);
      compact_text<<<8192, 128, 0, S.st>>>(S.text, TEXT_STRIDE, S.llen, S.loff, S.packed, (uint32_t) nu);
      CK(cudaMemcpyAsync(&S.h_last[0], S.loff + nu - 1, 4, cudaMemcpyDeviceToHost, S.st));
      CK(cudaMemcpyAsync(&S.h_last[1], S.llen + nu - 1, 4, cudaMemcpyDeviceToHost, S.st));
      S.tm[T_OUTPUT].stop(S.st);
    }
    CK(cudaMemcpyAsync(h_call + u0, S.call, nu*4, cudaMemcpyDeviceToHost, S.st));
    CK(cudaEventRecord(S.sized, S.st));
  };

  // Classifies on the host the units of chunk i the device left to it (reads
  // longer than MAX_LEN, lines too long for their slot, more distinct taxa
  // than the device keeps), fills in their calls, and, with output, returns
  // the chunk's lines with theirs merged in at their places.
  HostCtx hc{h_tab, k, l, ssm, tog, mahv, skip_probe, h_par.data(), h_ext.data()};
  // The pool is as large as the CPUs this process may run on, which under
  // Slurm is the allocation rather than the node.
  int host_cpus = (int) std::thread::hardware_concurrency();
  { cpu_set_t cs; if (sched_getaffinity(0, sizeof cs, &cs) == 0) host_cpus = CPU_COUNT(&cs); }
  int host_threads = env_int("K2GPU_HOST_THREADS", host_cpus);
  if (host_threads < 1) host_threads = 1;
  uint64_t host_units = 0, host_bases = 0;
  double t_host = 0;
  auto host_route = [&](size_t i, Set &S, uint32_t nh) -> std::shared_ptr<std::vector<char>> {
    double th = now_s();
    uint64_t nr = nrec(i), nu = nr / mates, u0 = rec0(i) / mates;
    std::vector<uint32_t> units(nh), off(nr), len(nr), hoff(nr), hlen(nr), loff;
    CK(cudaMemcpyAsync(units.data(), S.hunits, nh * 4, cudaMemcpyDeviceToHost, S.st));
    CK(cudaMemcpyAsync(off.data(), S.c_off, nr * 4, cudaMemcpyDeviceToHost, S.st));
    CK(cudaMemcpyAsync(len.data(), S.c_len, nr * 4, cudaMemcpyDeviceToHost, S.st));
    CK(cudaMemcpyAsync(hoff.data(), S.c_hoff, nr * 4, cudaMemcpyDeviceToHost, S.st));
    CK(cudaMemcpyAsync(hlen.data(), S.c_hlen, nr * 4, cudaMemcpyDeviceToHost, S.st));
    CK(cudaStreamSynchronize(S.st));
    std::sort(units.begin(), units.end());
    struct Job { std::string seq[2], hdr, line; uint32_t call; };
    std::vector<Job> jobs(nh);
    for (uint32_t j = 0; j < nh; j++) {
      uint32_t r = units[j];
      for (int m = 0; m < mates; m++) {
        uint32_t rec = r * mates + m;
        jobs[j].seq[m].resize(len[rec]);
        if (len[rec]) CK(cudaMemcpyAsync(&jobs[j].seq[m][0], S.c_ubuf + off[rec], len[rec],
                                         cudaMemcpyDeviceToHost, S.st));
        host_bases += len[rec];
      }
      uint32_t rec0 = r * mates;
      jobs[j].hdr.resize(hlen[rec0]);
      if (hlen[rec0]) CK(cudaMemcpyAsync(&jobs[j].hdr[0], S.c_ubuf + hoff[rec0], hlen[rec0],
                                         cudaMemcpyDeviceToHost, S.st));
    }
    std::vector<char> packed;
    if (out_path) {
      loff.resize(nu);
      CK(cudaMemcpyAsync(loff.data(), S.loff, nu * 4, cudaMemcpyDeviceToHost, S.st));
      packed.resize((size_t) S.h_last[0] + S.h_last[1]);
      if (!packed.empty())
        CK(cudaMemcpyAsync(packed.data(), S.packed, packed.size(), cudaMemcpyDeviceToHost, S.st));
    }
    CK(cudaStreamSynchronize(S.st));
    int nt = (int) std::min<uint32_t>(nh, (uint32_t) host_threads);
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; t++)
      pool.emplace_back([&, t]() {
        std::vector<uint32_t> tok;
        for (uint32_t j = t; j < nh; j += nt)
          jobs[j].call = classify_host(hc, mates, jobs[j].seq, jobs[j].hdr, tok,
                                       out_path ? &jobs[j].line : nullptr);
      });
    for (auto &th : pool) th.join();
    for (uint32_t j = 0; j < nh; j++) h_call[u0 + units[j]] = jobs[j].call;
    host_units += nh;
    std::shared_ptr<std::vector<char>> merged;
    if (out_path) {
      // A unit the host classified has an empty line on the device, so its
      // line goes at the offset the device gave it.
      merged = std::make_shared<std::vector<char>>();
      size_t extra = 0;
      for (auto &jb : jobs) extra += jb.line.size();
      merged->reserve(packed.size() + extra);
      size_t at = 0;
      for (uint32_t j = 0; j < nh; j++) {
        size_t pos = loff[units[j]];
        merged->insert(merged->end(), packed.begin() + at, packed.begin() + pos);
        merged->insert(merged->end(), jobs[j].line.begin(), jobs[j].line.end());
        at = pos;
      }
      merged->insert(merged->end(), packed.begin() + at, packed.end());
    }
    t_host += now_s() - th;
    return merged;
  };

  // Hands chunk i's packed lines to the writer, in order.
  uint64_t done_units = 0;
  auto finalize = [&](size_t i, Set &S) {
    double tw = now_s();
    CK(cudaEventSynchronize(S.sized));
    if (*S.h_nhost) {
      t_wait_out += now_s() - tw;
      std::shared_ptr<std::vector<char>> merged = host_route(i, S, *S.h_nhost);
      if (out_path) {
        size_t nb = merged->size();
        { std::lock_guard<std::mutex> lk(mu); wq.push_back({-1, nb, std::move(merged)}); }
        cv.notify_all();
      }
    } else if (out_path) {
      size_t tbytes = (size_t) S.h_last[0] + S.h_last[1];
      if (tbytes > out_cap) {
        fprintf(stderr, "packed output %zu bytes exceeds the %zu reserved\n", tbytes, out_cap);
        exit(1);
      }
      static int next_w = 0;
      int w = next_w; next_w = (next_w + 1) % NW;
      { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return !out_busy[w]; }); out_busy[w] = true; }
      t_wait_out += now_s() - tw;
      CK(cudaMemcpyAsync(outbuf[w], S.packed, tbytes, cudaMemcpyDeviceToHost, S.st));
      CK(cudaEventRecord(outev[w], S.st));
      { std::lock_guard<std::mutex> lk(mu); wq.push_back({w, tbytes, nullptr}); }
      cv.notify_all();
    } else t_wait_out += now_s() - tw;
    done_units += nrec(i) / mates;
  };

  // ---- the pipeline ----
  CK(cudaDeviceSynchronize());
  double t0 = now_s();
  double t_in = 0, t_bin = 0, t_groups = 0, t_fin = 0, bin_ms = 0;
  double copy_ms = 0, probe_ms = 0;
  size_t copy_bytes = 0;
  uint32_t mate_stride = 0, tok_stride = 0;
  uint64_t total_lookups = 0;

  if (! streamed) {
    for (size_t i = 0; i < nchunks; i++) {
      Set &S = set[i % 2];
      enqueue_load(i, S, S.ubuf, S.off, S.len, S.hoff, S.hlen);
      enqueue_classify(i, S, S.ubuf, S.off, S.len, S.hoff, S.hlen, StreamArgs{});
      if (i) finalize(i - 1, set[(i - 1) % 2]);
    }
    finalize(nchunks - 1, set[(nchunks - 1) % 2]);
  } else {
    uint32_t P = (uint32_t) d_tab.G.P, per = P / n_groups;
    size_t group_bytes = (size_t) per * d_tab.G.nb * 64;
    cudaStream_t cmp = set[0].st, cs = cmp;
    if (! serial) CK(cudaStreamCreate(&cs));
    uint16_t *d_tok = nullptr; uint32_t *d_ref = nullptr; size_t tok_cap = 0;
    uint64_t *d_bh = nullptr; uint16_t *d_bv = nullptr; uint32_t *d_bc = nullptr;
    size_t bin_cap_total = 0, bc_cap = 0;
    uint32_t *d_max; CK(cudaMalloc(&d_max, 4));
    std::vector<cudaEvent_t> c0(n_groups), c1(n_groups), p0(n_groups), p1(n_groups);
    for (uint32_t g = 0; g < n_groups; g++) {
      CK(cudaEventCreate(&c0[g])); CK(cudaEventCreate(&c1[g]));
      CK(cudaEventCreate(&p0[g])); CK(cudaEventCreate(&p1[g]));
    }
    cudaEvent_t b0ev, b1ev; CK(cudaEventCreate(&b0ev)); CK(cudaEventCreate(&b1ev));

    for (size_t bt = 0; bt + 1 < batch_first.size(); bt++) {
      size_t i0 = batch_first[bt], i1 = batch_first[bt + 1];
      uint64_t R0 = rec0(i0);
      uint64_t brecs = rec0(i1 - 1) + nrec(i1 - 1) - R0, bunits = brecs / mates;
      uint64_t uoff0 = ent[chunk_b0[i0]].uoff;
      auto bptrs = [&](size_t i, char *&ub, uint32_t *&o, uint32_t *&ln, uint32_t *&ho, uint32_t *&hl) {
        uint64_t r = rec0(i) - R0;
        ub = d_bubuf + (ent[chunk_b0[i]].uoff - uoff0);
        o = d_boff + r; ln = d_blen + r; ho = d_bhoff + r; hl = d_bhlen + r;
      };

      // Load every chunk of the batch.
      double ta = now_s();
      for (size_t i = i0; i < i1; i++) {
        char *ub; uint32_t *o, *ln, *ho, *hl; bptrs(i, ub, o, ln, ho, hl);
        enqueue_load(i, set[i % 2], ub, o, ln, ho, hl);
      }
      CK(cudaDeviceSynchronize());
      CK(cudaMemset(d_max, 0, 4));
      max_len<<<1024, 256>>>(d_blen, (uint32_t) brecs, d_max);
      uint32_t mlen; CK(cudaMemcpy(&mlen, d_max, 4, cudaMemcpyDeviceToHost));
      if (mlen > (uint32_t) MAX_LEN) mlen = MAX_LEN;
      // A mate yields at most len - k + 1 tokens, with or without ambiguous bases.
      mate_stride = mlen >= (uint32_t) k ? mlen - k + 1 : 1;
      tok_stride = mate_stride * mates;
      size_t need = (size_t) bunits * tok_stride;
      if (need >= 4294967296ULL) { fprintf(stderr, "batch too large for 32-bit token indices\n"); return 1; }
      if (need > tok_cap) {
        if (d_tok) CK(cudaFree(d_tok));
        if (d_ref) CK(cudaFree(d_ref));
        d_tok = nullptr; d_ref = nullptr;
        if (binned) CK(cudaMalloc(&d_ref, need * 4)); else CK(cudaMalloc(&d_tok, need * 2));
        tok_cap = need;
      }
      t_in += now_s() - ta;

      // Bins: one set per chunk, filed in one array.
      std::vector<uint64_t> bbase(i1 - i0 + 1, 0);
      std::vector<uint32_t> bcap(i1 - i0, 0);
      if (binned) {
        double tb = now_s();
        for (size_t i = i0; i < i1; i++) {
          double mean = (double) (nrec(i) / mates) * tok_stride * bin_frac / P;
          bcap[i - i0] = (uint32_t) (mean + 8 * sqrt(mean) + 1024);
          bbase[i - i0 + 1] = bbase[i - i0] + (uint64_t) P * bcap[i - i0];
        }
        if (bbase.back() >= 4294967296ULL) { fprintf(stderr, "bins exceed 32-bit entry indices\n"); return 1; }
        if (bbase.back() > bin_cap_total) {
          if (d_bh) { CK(cudaFree(d_bh)); CK(cudaFree(d_bv)); }
          CK(cudaMalloc(&d_bh, bbase.back() * 8)); CK(cudaMalloc(&d_bv, bbase.back() * 2));
          bin_cap_total = bbase.back();
        }
        if ((i1 - i0) * P > bc_cap) {
          if (d_bc) CK(cudaFree(d_bc));
          CK(cudaMalloc(&d_bc, (i1 - i0) * P * 4)); bc_cap = (i1 - i0) * P;
        }
        CK(cudaMemsetAsync(d_bc, 0, (i1 - i0) * P * 4, cmp));
        CK(cudaEventRecord(b0ev, cmp));
        for (size_t i = i0; i < i1; i++) {
          char *ub; uint32_t *o, *ln, *ho, *hl; bptrs(i, ub, o, ln, ho, hl);
          StreamArgs sa{BIN, nullptr, d_ref, (rec0(i) - R0) / mates * tok_stride, tok_stride, mate_stride,
                        0, 0, d_bh, d_bv, d_bc + (i - i0) * P, bcap[i - i0], bbase[i - i0]};
          fused<<<8192, THREADS, 0, cmp>>>(ub, o, ln, (uint32_t) (nrec(i) / mates), mates, k, l, ssm, tog, mahv,
              d_tab, 0, sa, d_parent, d_extid, set[0].call, nullptr, nullptr, nullptr, 0, nullptr, nullptr,
              set[0].hunits, set[0].nhost);
        }
        CK(cudaEventRecord(b1ev, cmp));
        std::vector<uint32_t> cnt((i1 - i0) * P);
        CK(cudaMemcpyAsync(cnt.data(), d_bc, cnt.size() * 4, cudaMemcpyDeviceToHost, cmp));
        CK(cudaStreamSynchronize(cmp));
        float ms; CK(cudaEventElapsedTime(&ms, b0ev, b1ev)); bin_ms += ms;
        for (size_t c = 0; c < i1 - i0; c++)
          for (uint32_t b = 0; b < P; b++) {
            uint32_t v = cnt[c * P + b];
            if (v > bcap[c]) {
              fprintf(stderr, "a bin overflowed (%u > %u); raise K2GPU_BIN_FRAC\n", v, bcap[c]);
              return 1;
            }
            total_lookups += v;
          }
        t_bin += now_s() - tb;
      }

      // The group loop.  Group g is partitions [g P / G, (g + 1) P / G),
      // copied from pinned host memory into slot g mod 2 on its own stream
      // while the previous group is probed.  A slot is refilled only after
      // the probes that read it are done.  With K2GPU_SERIAL the copies run
      // on the compute stream, so nothing overlaps.
      double tg = now_s();
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
        CK(cudaStreamWaitEvent(cmp, c1[g], 0));
        k2t::Table gt = d_tab;
        gt.buckets = d_slot[s];
        gt.part_base = (uint64_t) g * per;
        CK(cudaEventRecord(p0[g], cmp));
        for (size_t i = i0; i < i1; i++) {
          if (binned) {
            size_t c = i - i0;
            uint64_t n = (uint64_t) per * bcap[c];
            unsigned blocks = (unsigned) std::min<uint64_t>((n + 255) / 256, 1u << 20);
            probe_bins<<<blocks, 256, 0, cmp>>>(gt, d_bh, d_bv, d_bc + c * P, bcap[c], bbase[c],
                                                g * per, (g + 1) * per, bin_nowrite);
          } else {
            char *ub; uint32_t *o, *ln, *ho, *hl; bptrs(i, ub, o, ln, ho, hl);
            StreamArgs sa{PROBE_GROUP, d_tok, nullptr, (rec0(i) - R0) / mates * tok_stride, tok_stride,
                          mate_stride, g * per, (g + 1) * per, nullptr, nullptr, nullptr, 0, 0};
            fused<<<8192, THREADS, 0, cmp>>>(ub, o, ln, (uint32_t) (nrec(i) / mates), mates, k, l, ssm, tog, mahv,
                gt, 0, sa, d_parent, d_extid, set[0].call, nullptr, nullptr, nullptr, 0, nullptr, nullptr,
                set[0].hunits, set[0].nhost);
          }
        }
        CK(cudaEventRecord(p1[g], cmp));
        if (g + n_slots < n_groups) issue_copy(g + n_slots);
      }
      CK(cudaDeviceSynchronize());
      t_groups += now_s() - tg;
      for (uint32_t g = 0; g < n_groups; g++) {
        float a, b;
        CK(cudaEventElapsedTime(&a, c0[g], c1[g])); copy_ms += a;
        CK(cudaEventElapsedTime(&b, p0[g], p1[g])); probe_ms += b;
      }
      copy_bytes += (size_t) n_groups * group_bytes;

      // Finish the batch: resolve, format and write each chunk.
      double tf = now_s();
      for (size_t i = i0; i < i1; i++) {
        char *ub; uint32_t *o, *ln, *ho, *hl; bptrs(i, ub, o, ln, ho, hl);
        StreamArgs sa{FINISH, d_tok, d_ref, (rec0(i) - R0) / mates * tok_stride, tok_stride, mate_stride,
                      0, 0, nullptr, d_bv, nullptr, 0, 0};
        enqueue_classify(i, set[i % 2], ub, o, ln, ho, hl, sa);
        if (i > i0) finalize(i - 1, set[(i - 1) % 2]);
      }
      finalize(i1 - 1, set[(i1 - 1) % 2]);
      CK(cudaDeviceSynchronize());
      t_fin += now_s() - tf;
    }
  }
  CK(cudaDeviceSynchronize());
  { std::lock_guard<std::mutex> lk(mu); wdone = true; }
  cv.notify_all();
  writer.join();
  reader.join();
  double el = now_s() - t0;
  for (Set &S : set) for (auto &t : S.tm) t.collect();

  if (out_fd >= 0) { close(out_fd); }
  printf("\nend to end   %.3f s   %.2f Gbase/s   %.1f Munit/s   %.2f Munit total\n",
         el, ih.total_uncompressed * 0.4 / el / 1e9, done_units / el / 1e6, done_units / 1e6);
  printf("             (bases are about 40%% of the FASTQ bytes; a unit is a %s)\n",
         mates == 2 ? "pair" : "read");
  printf("device stage times (s, summed over chunks; the two streams overlap):\n");
  for (int t = 0; t < N_T; t++) {
    double ms = set[0].tm[t].ms + set[1].tm[t].ms;
    if (ms > 0) printf("  %-44s %8.3f\n", stage_name[t], ms / 1e3);
  }
  printf("host classification: %" PRIu64 " of %" PRIu64 " units (%.4f%%), %.2f Mbase, "
         "%.3f s on up to %d threads\n", host_units, done_units,
         done_units ? 100.0 * host_units / done_units : 0.0, host_bases / 1e6, t_host, host_threads);
  printf("host threads (s):\n");
  printf("  %-44s %8.3f\n", "reader: reading input", t_read);
  printf("  %-44s %8.3f\n", "writer: writing output", t_write);
  printf("  %-44s %8.3f\n", "main: waiting for input", t_wait_in);
  printf("  %-44s %8.3f\n", "main: waiting for a chunk or an output buffer", t_wait_out);
  if (streamed) {
    size_t nb = batch_first.size() - 1;
    printf("streamed: %zu batches; %u groups of %u partitions (%.3f GB each), %s%s\n", nb, n_groups,
           (uint32_t) d_tab.G.P / n_groups, table_bytes / 1e9 / n_groups,
           binned ? "binned" : "recomputed", serial ? ", serial" : ", overlapped");
    printf("  %-44s %8.3f\n", "load all chunks of each batch (wall)", t_in);
    if (binned) printf("  %-44s %8.3f   (device %.3f; %.1f lookups per unit)\n", "bin (wall)", t_bin,
                       bin_ms / 1e3, (double) total_lookups / done_units);
    printf("  %-44s %8.3f\n", "group loops (wall)", t_groups);
    printf("  %-44s %8.3f   (%.2f GB, %.1f GB/s)\n", "  table copies (sum)", copy_ms / 1e3,
           copy_bytes / 1e9, copy_bytes / 1e9 / (copy_ms / 1e3));
    printf("  %-44s %8.3f\n", "  probes (sum)", probe_ms / 1e3);
    printf("  %-44s %8.3f\n", "finish (wall)", t_fin);
    if (binned)
      printf("per-unit state: %u bytes of token references + %.0f bytes of binned "
             "minimizers and results\n", tok_stride * 4, (double) total_lookups / done_units * 10);
    else
      printf("per-unit state: %u bytes of token taxa\n", tok_stride * 2);
  }
  {
    uint32_t over = 0;
    CK(cudaMemcpy(&over, d_over, 4, cudaMemcpyDeviceToHost));
    if (out_bytes)
      printf("             wrote %.2f GB of output, %u truncated or too long\n", out_bytes / 1e9, over);
    if (over) { fprintf(stderr, "some reads were too long or hit lists did not fit the slot\n"); return 1; }
  }

  // ---- correctness, against classify's own calls ----
  FILE *rf = fopen(argv[6], "rb");
  if (!rf) { fprintf(stderr, "cannot open %s\n", argv[6]); return 1; }
  size_t bad = 0, seen = 0, classified = 0;
  // A long read's line can run to megabytes, so lines are read whole.
  char *line = nullptr; size_t line_cap = 0;
  for (uint64_t i = 0; i < done_units && getline(&line, &line_cap, rf) > 0; i++) {
    unsigned long want = 0;
    char *p1 = strchr(line, '\t');
    char *p2 = p1 ? strchr(p1 + 1, '\t') : nullptr;
    if (p2) want = strtoul(p2 + 1, nullptr, 10);
    seen++;
    if (want) classified++;
    if ((unsigned long) h_call[i] != want) {
      if (bad < 3) fprintf(stderr, "unit %" PRIu64 ": classify %lu, gpu %u\n", i, want, h_call[i]);
      bad++;
    }
  }
  fclose(rf); free(line);
  printf("verified %zu calls against classify (%.2f%% classified): %zu mismatched\n",
         seen, seen ? 100.0*classified/seen : 0.0, bad);
  return bad ? 2 : 0;
}
