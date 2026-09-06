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
  const uint32_t *table; uint64_t capacity; int key_bits, value_bits;
  const uint64_t *parent, *extid; uint64_t mahv;
  kraken2::MinimizerScanner *scanner;

  uint32_t probe_host(uint64_t key) const {
    uint64_t hc = murmur(key);
    uint32_t compacted = (uint32_t) (hc >> (64 - key_bits));
    uint64_t idx = hc % capacity, first = idx;
    const uint32_t vmask = (1u << value_bits) - 1;
    while (true) {
      uint32_t cell = table[idx];
      uint32_t v = cell & vmask;
      if (!v) return 0;
      if ((cell >> value_bits) == compacted) return v;
      if (++idx == capacity) idx = 0;
      if (idx == first) return 0;
    }
  }
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
        else { taxon = probe_host(*mp); last_taxon = taxon; }
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
  size_t chunks_done = 0;

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
  uint64_t capacity, key_bits, value_bits;
  uint32_t *d_table = nullptr;
  {
    int hf = open(argv[3], O_RDONLY);
    uint64_t hdr[4];
    if (hf < 0 || read(hf, hdr, sizeof hdr) < (ssize_t) sizeof hdr) { perror("hash"); return 1; }
    capacity = hdr[0]; key_bits = hdr[2]; value_bits = hdr[3];
    size_t tb = capacity * 4;
    CK(cudaMalloc(&d_table, tb));
    const size_t CH = 256u << 20;
    std::vector<char> buf(CH);
    size_t done = 0;
    while (done < tb) {
      size_t want = std::min(CH, tb - done), got = 0;
      while (got < want) {
        ssize_t n = pread(hf, buf.data() + got, want - got, 32 + done + got);
        if (n <= 0) break;
        got += n;
      }
      CK(cudaMemcpy((char *) d_table + done, buf.data(), got, cudaMemcpyHostToDevice));
      done += got;
      if (got < want) break;
    }
    close(hf);
    printf("table: %.2f GB resident\n", done / 1e9);
  }

  // A host view of the same table for the fallback path.  Mapping it costs
  // nothing until a page is touched, and only a handful ever are.
  const uint32_t *h_table = nullptr;
  {
    int hf = open(argv[3], O_RDONLY);
    struct stat st; fstat(hf, &st);
    char *m = (char *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, hf, 0);
    if (m == MAP_FAILED) { perror("mmap hash"); return 1; }
    h_table = (const uint32_t *) (m + 32);
    close(hf);
  }

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
  size_t max_c = 0, max_u = 0, max_r = 0, max_rec_per_blk = 0;
  for (size_t b0 = 0; b0 < ih.n_blocks; b0 += chunk_blocks) {
    size_t b1 = std::min(b0 + chunk_blocks, (size_t) ih.n_blocks);
    max_c = std::max(max_c, (size_t) (ent[b1].coff - ent[b0].coff));
    max_u = std::max(max_u, (size_t) (ent[b1].uoff - ent[b0].uoff));
    max_r = std::max(max_r, (size_t) (ent[b1].rec  - ent[b0].rec));
  }
  for (size_t b = 0; b < ih.n_blocks; b++)
    max_rec_per_blk = std::max(max_rec_per_blk, (size_t) (ent[b+1].rec - ent[b].rec));
  size_t max_ublk = 0;
  for (size_t b = 0; b < ih.n_blocks; b++)
    max_ublk = std::max(max_ublk, (size_t) (ent[b+1].uoff - ent[b].uoff));

  char *h_pin;  CK(cudaHostAlloc(&h_pin, max_c, cudaHostAllocDefault));
  char *d_cbuf; CK(cudaMalloc(&d_cbuf, max_c));
  char *d_ubuf; CK(cudaMalloc(&d_ubuf, max_u));
  uint32_t *d_off, *d_len, *d_call, *d_hoff, *d_hlen, *d_llen, *d_loff, *d_sums;
  CK(cudaMalloc(&d_off, max_r*4)); CK(cudaMalloc(&d_len, max_r*4));
  CK(cudaMalloc(&d_call, max_r*4));
  CK(cudaMalloc(&d_hoff, max_r*4)); CK(cudaMalloc(&d_hlen, max_r*4));
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
  HostFallback fb_ctx{h_table, capacity, (int) key_bits, (int) value_bits,
                      h_par.data(), h_ext.data(), mahv, &fb_scanner};
  if (out_path) {
    out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    printf("writing output to %s\n", out_path);
  }

  // ---- the pipeline ----
  CK(cudaDeviceSynchronize());
  double t0 = now_s();
  uint64_t done_reads = 0;

  for (size_t b0 = 0; b0 < ih.n_blocks; b0 += chunk_blocks) {
    size_t b1 = std::min(b0 + chunk_blocks, (size_t) ih.n_blocks);
    size_t nb = b1 - b0;
    size_t clen = ent[b1].coff - ent[b0].coff;
    size_t nrec = ent[b1].rec - ent[b0].rec;

    size_t got = 0;
    while (got < clen) {
      ssize_t n = pread(bgz, h_pin + got, clen - got, ent[b0].coff + got);
      if (n <= 0) break;
      got += n;
    }
    CK(cudaMemcpyAsync(d_cbuf, h_pin, clen, cudaMemcpyHostToDevice, stream));

    for (size_t i = 0; i < nb; i++) {
      h_cptr[i] = d_cbuf + (ent[b0+i].coff - ent[b0].coff);
      h_uptr[i] = d_ubuf + (ent[b0+i].uoff - ent[b0].uoff);
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

    NK(nvcompBatchedGzipDecompressAsync((const void *const *) d_cptr, d_cb, d_ub,
        d_ac, nb, d_temp, temp_bytes, (void *const *) d_uptr, nopts, d_st, stream));

    // Decompression reports per block, and ignoring that turns a decode failure
    // into a wild read three kernels later.  Checked on the first chunk, which
    // is where a wrong argument would show up.
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
      if (nbad || nshort) { fprintf(stderr, "decompression did not agree with the index\n"); return 2; }
    }

    parse_blocks<<<(unsigned) std::min(nb, (size_t) 8192), 256, shmem, stream>>>(
        d_ubuf, d_buoff, d_brec, ent[b0].uoff, (uint32_t) nb, ent[b0].rec,
        d_off, d_len, d_hoff, d_hlen);

    if (out_path) CK(cudaMemsetAsync(d_amb, 0, nrec, stream));
    fused<<<8192, THREADS, 0, stream>>>(
        d_ubuf, d_off, d_len, (uint32_t) nrec, k, l, ssm, tog, mahv,
        d_table, capacity, (int) key_bits, (int) value_bits,
        nullptr, nullptr, nullptr, nullptr, d_parent, d_extid, d_call,
        d_hoff, d_hlen, d_text, TEXT_STRIDE, d_llen, d_over, d_amb);

    if (out_path) {
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
        return 1;
      }
      CK(cudaMemcpyAsync(h_out, d_packed, tbytes, cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));
      // Reads the kernel declined contribute no bytes, so their line belongs
      // exactly at their offset.  Splice the host's version in there.
      std::vector<uint8_t> amb(nrec);
      CK(cudaMemcpy(amb.data(), d_amb, nrec, cudaMemcpyDeviceToHost));
      std::vector<uint32_t> fb;
      for (size_t i = 0; i < nrec; i++) if (amb[i]) fb.push_back((uint32_t) i);

      auto emit = [&](const char *p, size_t nb) {
        size_t w = 0;
        while (w < nb) {
          ssize_t got = write(out_fd, p + w, nb - w);
          if (got <= 0) { perror("write"); exit(1); }
          w += got;
        }
        out_bytes += nb;
      };

      if (fb.empty()) emit(h_out, tbytes);
      else {
        std::vector<uint32_t> loff(nrec), soff(nrec), slen(nrec), hoff(nrec), hlen(nrec);
        CK(cudaMemcpy(loff.data(), d_loff, nrec*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(soff.data(), d_off,  nrec*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(slen.data(), d_len,  nrec*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hoff.data(), d_hoff, nrec*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hlen.data(), d_hlen, nrec*4, cudaMemcpyDeviceToHost));
        size_t pos = 0;
        std::vector<char> rb;
        for (uint32_t r : fb) {
          emit(h_out + pos, loff[r] - pos);
          pos = loff[r];
          rb.resize(hlen[r] + slen[r]);
          CK(cudaMemcpy(rb.data(), d_ubuf + hoff[r], hlen[r], cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(rb.data() + hlen[r], d_ubuf + soff[r], slen[r], cudaMemcpyDeviceToHost));
          std::string ln = fb_ctx.line(rb.data(), hlen[r], rb.data() + hlen[r], slen[r]);
          emit(ln.data(), ln.size());
          n_fallback++;
        }
        emit(h_out + pos, tbytes - pos);
      }
    }

    CK(cudaMemcpyAsync(h_call.data() + done_reads, d_call, nrec*4,
                       cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    done_reads += nrec;
    if (++chunks_done >= max_chunks) break;
  }
  double el = now_s() - t0;

  if (out_fd >= 0) { close(out_fd); }
  printf("\nend to end   %.3f s   %.2f Gbase/s   %.1f Mread/s   %.2f Mread total\n",
         el, ih.total_uncompressed * 0.4 / el / 1e9, done_reads / el / 1e6,
         done_reads / 1e6);
  printf("             (bases are about 40%% of the FASTQ bytes)\n");
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
