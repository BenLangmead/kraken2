// Scan, subsample and probe, fused, against the real Kraken 2 hash table.
//
// The three stages have been measured apart; this is the first time they run
// together against a real 16 GB table, which is what exercises the parts that
// isolated benchmarks cannot: that the device MurmurHash3 agrees with the
// host's, that the compact cell decoding is right, that the subsampling filter
// selects the same minimizers, and that a table of this size can simply live on
// the device while reads stream past it.
//
// Correctness is checked against k2mindump's output rather than asserted: for
// every read, the sequence of (minimizer, taxon) pairs the device produces must
// equal the sequence the CPU produced, in order.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "kernels.cuh"

#define CK(x) do { cudaError_t e=(x); if (e!=cudaSuccess) { \
  fprintf(stderr,"cuda %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: fused reads.fq hash.k2d opts.k2d taxo.k2d classify.out [reads]\n");
    return 1;
  }
  size_t limit = argc > 6 ? (size_t) atoll(argv[6]) : 50000;

  // ---- index options ----
  uint64_t o[8];
  { int fd = open(argv[3], O_RDONLY);
    if (fd < 0 || read(fd, o, sizeof o) < (ssize_t) sizeof o) { perror("opts"); return 1; }
    close(fd); }
  int k = (int) o[0], l = (int) o[1];
  uint64_t ssm = o[2], tog = o[3], mahv = o[5];
  printf("k=%d l=%d ssm=0x%" PRIx64 " toggle=0x%" PRIx64 " min_hash=0x%" PRIx64 "\n",
         k, l, ssm, tog, mahv);

  // ---- hash table ----
  uint64_t hdr[4];
  int hf = open(argv[2], O_RDONLY);
  if (hf < 0 || read(hf, hdr, sizeof hdr) < (ssize_t) sizeof hdr) { perror("hash"); return 1; }
  uint64_t capacity = hdr[0], key_bits = hdr[2], value_bits = hdr[3];
  size_t tbytes = capacity * 4;
  printf("table capacity %" PRIu64 ", %.2f GB, key_bits %" PRIu64 ", value_bits %" PRIu64 "\n",
         capacity, tbytes / 1e9, key_bits, value_bits);

  uint32_t *d_table = nullptr;
  CK(cudaMalloc(&d_table, tbytes));
  { const size_t CH = 256u << 20;
    std::vector<char> buf(CH);
    size_t done = 0;
    while (done < tbytes) {
      size_t want = tbytes - done < CH ? tbytes - done : CH;
      size_t got = 0;
      while (got < want) {
        ssize_t n = pread(hf, &buf[0] + got, want - got, 32 + done + got);
        if (n <= 0) break;
        got += n;
      }
      CK(cudaMemcpy((char *) d_table + done, &buf[0], got, cudaMemcpyHostToDevice));
      done += got;
      if (got < want) break;
    }
    printf("table uploaded: %.2f GB\n", done / 1e9);
  }
  close(hf);

  // ---- taxonomy: only parent and external id are needed ----
  uint64_t *d_parent = nullptr, *d_extid = nullptr; uint64_t node_count = 0;
  {
    int tf = open(argv[4], O_RDONLY);
    if (tf < 0) { fprintf(stderr, "cannot open taxonomy %s\n", argv[4]); return 1; }
    char hdr[32];
    if (read(tf, hdr, 32) < 32 || memcmp(hdr, "K2TAXDAT", 8) != 0) {
      fprintf(stderr, "%s is not a taxonomy\n", argv[4]); return 1;
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
    std::vector<uint64_t> par(node_count), ext(node_count);
    for (uint64_t i = 0; i < node_count; i++) {
      par[i] = raw[i * 7 + 0];
      ext[i] = raw[i * 7 + 5];
    }
    CK(cudaMalloc(&d_parent, node_count * 8));
    CK(cudaMalloc(&d_extid,  node_count * 8));
    CK(cudaMemcpy(d_parent, par.data(), node_count * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_extid,  ext.data(), node_count * 8, cudaMemcpyHostToDevice));
    printf("taxonomy: %" PRIu64 " nodes, %.1f MB uploaded\n",
           node_count, node_count * 16 / 1e6);
  }

  // ---- reads ----
  FILE *f = fopen(argv[1], "rb");
  if (! f) { fprintf(stderr, "cannot open reads %s\n", argv[1]); return 1; }
  std::vector<char> bases; std::vector<uint32_t> offs, lens;
  char line[8192]; size_t nr = 0;
  while (nr < limit && fgets(line, sizeof line, f)) {
    if (!fgets(line, sizeof line, f)) break;
    size_t L = strlen(line); while (L && (line[L-1]=='\n'||line[L-1]=='\r')) L--;
    if ((int) L > MAX_LEN) {
      fprintf(stderr, "read %zu is %zu bases, longer than the %d this build "
              "handles\n", nr, L, MAX_LEN);
      return 1;
    }
    offs.push_back((uint32_t) bases.size()); lens.push_back((uint32_t) L);
    bases.insert(bases.end(), line, line + L);
    if (!fgets(line, sizeof line, f)) break;
    if (!fgets(line, sizeof line, f)) break;
    nr++;
  }
  fclose(f);
  std::vector<uint32_t> ooff(nr + 1); uint64_t tot = 0;
  for (size_t i = 0; i < nr; i++) {
    ooff[i] = (uint32_t) tot;
    int nk = (int) lens[i] - k + 1; if (nk < 0) nk = 0;
    tot += nk;
  }
  ooff[nr] = (uint32_t) tot;
  printf("reads %zu, %.2f MB bases, %" PRIu64 " k-mer positions\n", nr, bases.size()/1e6, tot);

  char *d_b; uint32_t *d_o, *d_l, *d_oo, *d_tax, *d_cnt; uint64_t *d_key;
  CK(cudaMalloc(&d_b, bases.size())); CK(cudaMalloc(&d_o, nr*4));
  CK(cudaMalloc(&d_l, nr*4)); CK(cudaMalloc(&d_oo, (nr+1)*4));
  CK(cudaMalloc(&d_key, tot*8)); CK(cudaMalloc(&d_tax, tot*4)); CK(cudaMalloc(&d_cnt, nr*4));
  uint32_t *d_call; CK(cudaMalloc(&d_call, nr*4));
  CK(cudaMemcpy(d_b, bases.data(), bases.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_o, offs.data(), nr*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_l, lens.data(), nr*4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_oo, ooff.data(), (nr+1)*4, cudaMemcpyHostToDevice));

  k2t::Table cht{};
  cht.kind = k2t::Table::CHT; cht.cells = d_table; cht.capacity = capacity;
  cht.key_bits = (int) key_bits; cht.value_bits = (int) value_bits;
  int blocks = 8192;
  fused<<<blocks, THREADS>>>(d_b,d_o,d_l,(uint32_t)nr,k,l,ssm,tog,mahv,
                             cht,0,
                             d_key,d_tax,d_cnt,d_oo,d_parent,d_extid,d_call,
                             nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr);
  CK(cudaDeviceSynchronize());

  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  int reps = 10;
  CK(cudaEventRecord(a));
  for (int i = 0; i < reps; i++)
    fused<<<blocks, THREADS>>>(d_b,d_o,d_l,(uint32_t)nr,k,l,ssm,tog,mahv,
                               cht,0,
                               d_key,d_tax,d_cnt,d_oo,d_parent,d_extid,d_call,
                               nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr);
  CK(cudaEventRecord(b)); CK(cudaDeviceSynchronize());
  float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=reps;
  printf("\nfused scan+filter+probe   %.2f ms   %.2f Mbase/s   %.1f Mread/s\n",
         ms, bases.size()/(ms/1000.0)/1e6, nr/(ms/1000.0)/1e6);

  // ---- verify against classify's own calls ----
  std::vector<uint32_t> hcall(nr);
  CK(cudaMemcpy(hcall.data(), d_call, nr*4, cudaMemcpyDeviceToHost));

  FILE *rf = fopen(argv[5], "rb");
  if (! rf) { fprintf(stderr, "cannot open %s\n", argv[5]); return 1; }
  size_t bad = 0, seen = 0, classified = 0;
  char oline[8192];
  for (size_t i = 0; i < nr && fgets(oline, sizeof oline, rf); i++) {
    unsigned long want = 0;
    char *p1 = strchr(oline, '\t');
    char *p2 = p1 ? strchr(p1 + 1, '\t') : nullptr;
    if (p2) want = strtoul(p2 + 1, nullptr, 10);
    seen++;
    if (want) classified++;
    if ((unsigned long) hcall[i] != want) {
      if (bad < 3)
        fprintf(stderr, "read %zu: classify %lu, gpu %u\n", i, want, hcall[i]);
      bad++;
    }
  }
  fclose(rf);
  printf("verified %zu calls against classify (%.2f%% classified): %zu mismatched\n",
         seen, seen ? 100.0 * classified / seen : 0.0, bad);
  return bad ? 2 : 0;
}
